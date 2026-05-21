"""Single-process test project.

POST /api/process — full pipeline:
  1. Download the room image.
  2. Roboflow SAM3 ``concept_segment`` with a fixed list of furniture prompts.
  3. For each detected object: crop to bbox + apply mask as alpha → transparent
     PNG → upload to R2 → get a public URL.
  4. Run SearchAPI.io Google Lens (search_type=products) against each public URL
     in parallel.
  5. Keep only candidates that look genuinely purchasable
     (has a price AND is not flagged out-of-stock).

GET / — serves the frontend (../frontend/index.html and friends).

Env vars required — copy .env.example to .env and fill in:
  ROBOFLOW_API_KEY
  SEARCH_API_KEY
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
  R2_BUCKET_NAME, R2_PUBLIC_URL
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pycocotools import mask as coco_mask
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("test_project")


# ─────────────────────────────────────────────────────────────────────
# Config (from env)
# ─────────────────────────────────────────────────────────────────────

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"
ROBOFLOW_CONFIDENCE = float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.35"))

SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY", "")
LENS_ENDPOINT = "https://www.searchapi.io/api/v1/search"

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")

# The set of object types SAM3 looks for. Edit freely — each prompt gives at
# most one detection per image; cheap/safe to add more.
DEFAULT_PROMPTS = [
    "sofa",
    "armchair",
    "chair",
    "coffee table",
    "side table",
    "dining table",
    "floor lamp",
    "table lamp",
    "rug",
    "artwork",
    "plant",
    "vase",
    "bookshelf",
    "cabinet",
    "tv",
    "bed",
    "pillow",
    "curtain",
    "mirror",
]

# Lens config
LENS_COUNTRY = os.environ.get("LENS_COUNTRY", "us")
LENS_LANGUAGE = os.environ.get("LENS_LANGUAGE", "en")
LENS_MAX_PARALLEL = int(os.environ.get("LENS_MAX_PARALLEL", "6"))

# Gemini — vision LLM that emits SAM3-ready prompts by looking at the room.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


# ─────────────────────────────────────────────────────────────────────
# Gemini client (lazy, thread-safe)
# ─────────────────────────────────────────────────────────────────────

_gemini_client = None
_gemini_client_lock = threading.Lock()


def _gemini_get_client():
    """Lazy thread-safe ``google.genai`` client."""
    global _gemini_client
    if _gemini_client is None:
        with _gemini_client_lock:
            if _gemini_client is None:
                if not GOOGLE_API_KEY:
                    raise RuntimeError("GOOGLE_API_KEY missing in .env")
                from google import genai

                _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


# Instruction we send to Gemini. The output format is constrained to a JSON
# array of 3-4 word SAM3 prompts — colour first because SAM3 matches colour
# most reliably, then material/shape, then category.
DISCOVERY_INSTRUCTION = """\
You are looking at a photo of a room. List every distinct piece of furniture
and decor visible.

For each item, output a SAM3-ready segmentation prompt with 3-4 words
formatted as:  <colour> <material-or-texture> <category>

Examples of valid prompts:
  "gray linen sofa"
  "round oak coffee table"
  "tall brass floor lamp"
  "blue patterned wool rug"
  "framed abstract artwork"
  "green potted plant"

Rules:
  - Colour FIRST — SAM3 matches colour most reliably.
  - 3-4 words total per prompt. No more, no less.
  - Skip walls, floor, and ceiling — handled separately.
  - Skip tiny clutter (books, glasses, throw pillows on a sofa).
  - If two similar items are visible (e.g. two matching chairs), list once.
  - Return ONLY a JSON array of strings. No prose, no markdown fences.

Example output:
["gray linen sofa", "round oak coffee table", "tall brass floor lamp"]
"""


def _parse_prompt_list(text: str) -> List[str]:
    """Pull a JSON array of strings out of Gemini's reply, tolerantly."""
    text = (text or "").strip()
    # Strip markdown code fences if Gemini added them.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    return [str(p).strip() for p in data if isinstance(p, str) and p.strip()]


def discover_sam3_prompts(room_image: Image.Image) -> List[str]:
    """Ask Gemini Flash Lite to enumerate visible objects as SAM3 prompts.

    Returns a list of 3-4 word prompts ordered by visual prominence. Returns
    an empty list if Gemini cannot produce a usable response (caller falls
    back to ``DEFAULT_PROMPTS``).
    """
    from google.genai import types

    client = _gemini_get_client()
    contents = [room_image, DISCOVERY_INSTRUCTION]
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.0,
        ),
    )
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    parts = candidates[0].content.parts if candidates[0].content else []
    text = "\n".join(p.text for p in parts if getattr(p, "text", None))
    return _parse_prompt_list(text)


# ─────────────────────────────────────────────────────────────────────
# R2 client (boto3 S3-compatible)
# ─────────────────────────────────────────────────────────────────────


def _r2_client():
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        raise RuntimeError(
            "R2 env vars missing. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_URL in .env"
        )
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_png_to_r2(png_bytes: bytes, key: str) -> str:
    """Upload PNG bytes to R2 under ``key`` and return its public URL."""
    client = _r2_client()
    client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
        CacheControl="public, max-age=3600",
    )
    return f"{R2_PUBLIC_URL}/{key}"


# ─────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────


def download_image(url: str) -> Image.Image:
    """Fetch an image URL into a PIL.Image (RGB)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def encode_jpeg_b64(image: Image.Image, quality: int = 90) -> str:
    """JPEG-encode and base64 a PIL image, for the Roboflow payload."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def rle_to_mask(rle: Dict[str, Any], height: int, width: int) -> Optional[np.ndarray]:
    """Decode an RLE mask payload into a uint8 [H, W] mask (0 / 255)."""
    if not isinstance(rle, dict) or "counts" not in rle:
        return None
    size = rle.get("size") or [height, width]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    try:
        decoded = coco_mask.decode({"size": size, "counts": counts})
    except Exception as e:
        logger.warning(f"RLE decode failed: {e}")
        return None
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return (decoded.astype(np.uint8)) * 255


def bbox_of_mask(mask: np.ndarray, padding: int = 6) -> Optional[tuple]:
    """Return (left, top, right, bottom) of the non-zero region, padded."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = mask.shape
    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, w)
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding + 1, h)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def cutout_png(image: Image.Image, mask: np.ndarray, bbox: tuple) -> bytes:
    """Build a transparent-background PNG of the masked object cropped to bbox.

    ``image`` is the original RGB room. ``mask`` is the full-image binary mask.
    ``bbox`` is (left, top, right, bottom).
    """
    left, top, right, bottom = bbox
    rgb_crop = image.crop(bbox)
    mask_crop = Image.fromarray(mask[top:bottom, left:right], mode="L")
    rgba = Image.new("RGBA", rgb_crop.size, (0, 0, 0, 0))
    rgba.paste(rgb_crop, (0, 0), mask=mask_crop)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Roboflow SAM3 — concept_segment
# ─────────────────────────────────────────────────────────────────────


def call_sam3(image_b64: str, prompts: List[str]) -> List[Dict[str, Any]]:
    """POST to Roboflow's SAM3 ``concept_segment`` endpoint.

    Returns a list of ``prompt_results``: one entry per prompt, in input order.
    Each entry has either ``rle`` + ``score`` or is empty.
    """
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY missing in .env")
    payload = {
        "image": {"type": "base64", "value": image_b64},
        "prompts": [{"type": "text", "text": p} for p in prompts],
        "format": "rle",
        "output_prob_thresh": ROBOFLOW_CONFIDENCE,
    }
    resp = requests.post(
        f"{ROBOFLOW_ENDPOINT}?api_key={ROBOFLOW_API_KEY}",
        json=payload,
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Roboflow SAM3 returned {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json().get("prompt_results", []) or []


# ─────────────────────────────────────────────────────────────────────
# Google Lens (SearchAPI.io) — search_type=products
# ─────────────────────────────────────────────────────────────────────


def call_google_lens(image_url: str) -> List[Dict[str, Any]]:
    """Call Google Lens products search for one image URL.

    Returns the raw ``visual_matches`` list (may be empty).
    """
    if not SEARCH_API_KEY:
        raise RuntimeError("SEARCH_API_KEY missing in .env")
    params = {
        "engine": "google_lens",
        "search_type": "products",
        "url": image_url,
        "country": LENS_COUNTRY,
        "hl": LENS_LANGUAGE,
        "api_key": SEARCH_API_KEY,
    }
    resp = requests.get(LENS_ENDPOINT, params=params, timeout=120)
    if not resp.ok:
        logger.warning(f"Lens {resp.status_code} for {image_url}: {resp.text[:200]}")
        return []
    return resp.json().get("visual_matches", []) or []


# ─────────────────────────────────────────────────────────────────────
# Purchasability filter (mirrors test_google_lens_products.py)
# ─────────────────────────────────────────────────────────────────────


def _is_out_of_stock(stock_info: Any) -> bool:
    if not stock_info:
        return False
    text = str(stock_info).lower()
    return any(p in text for p in ("out of stock", "sold out", "unavailable"))


def _has_price(m: Dict[str, Any]) -> bool:
    return m.get("extracted_price") is not None or bool(m.get("price"))


def filter_in_stock(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only matches that are priced AND not flagged out-of-stock."""
    out = []
    for m in matches:
        if not _has_price(m):
            continue
        if _is_out_of_stock(m.get("stock_information")):
            continue
        out.append(
            {
                "title": m.get("title"),
                "source": m.get("source"),
                "link": m.get("link"),
                "thumbnail": m.get("thumbnail"),
                "price": m.get("price"),
                "extracted_price": m.get("extracted_price"),
                "currency": m.get("currency"),
                "stock_information": m.get("stock_information"),
                "rating": m.get("rating"),
                "reviews": m.get("reviews"),
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────


class ProcessRequest(BaseModel):
    """JSON body for POST /api/process."""

    image_url: str
    # Optional override. When set, skips the Gemini discovery step.
    prompts: Optional[List[str]] = None


def process_pipeline(
    image_url: str, override_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run the full discovery → segment → upload → lens → filter pipeline."""
    logger.info(f"Downloading room: {image_url}")
    room_img = download_image(image_url)
    w, h = room_img.size

    # Step 0: Gemini Flash Lite enumerates visible objects as SAM3 prompts.
    # If the caller forced a prompt list we use that as-is. Otherwise we ask
    # Gemini; on any failure we fall back to the hardcoded DEFAULT_PROMPTS so
    # the pipeline still runs.
    prompts: List[str]
    prompt_source: str
    if override_prompts:
        prompts = override_prompts
        prompt_source = "caller"
        logger.info(f"Using {len(prompts)} caller-supplied prompts")
    else:
        try:
            discovered = discover_sam3_prompts(room_img)
            if discovered:
                prompts = discovered
                prompt_source = "gemini"
                logger.info(
                    f"Gemini ({GEMINI_MODEL}) discovered {len(prompts)} prompts: "
                    f"{prompts}"
                )
            else:
                prompts = DEFAULT_PROMPTS
                prompt_source = "default-empty-discovery"
                logger.warning(
                    "Gemini returned no usable prompts — using DEFAULT_PROMPTS"
                )
        except Exception as e:
            prompts = DEFAULT_PROMPTS
            prompt_source = "default-error"
            logger.warning(f"Gemini discovery failed ({e}) — using DEFAULT_PROMPTS")

    image_b64 = encode_jpeg_b64(room_img)
    logger.info(f"SAM3: {len(prompts)} prompts")
    prompt_results = call_sam3(image_b64, prompts)

    # Step 1: per-prompt detection → mask → cutout PNG → upload to R2
    objects = []  # accumulates per-detected-object dicts
    masked_urls = []  # the URLs to fan out to Lens

    for label, result in zip(prompts, prompt_results):
        rle = result.get("rle") or (
            result.get("annotations", [{}])[0].get("rle")
            if result.get("annotations")
            else None
        )
        if not rle:
            continue
        mask = rle_to_mask(rle, h, w)
        if mask is None or mask.max() == 0:
            continue
        bbox = bbox_of_mask(mask)
        if bbox is None:
            continue
        png = cutout_png(room_img, mask, bbox)
        key = f"test-project/cutouts/{uuid.uuid4().hex}.png"
        try:
            url = upload_png_to_r2(png, key)
        except Exception as e:
            logger.warning(f"R2 upload failed for {label}: {e}")
            continue
        score = result.get("score")
        objects.append(
            {
                "label": label,
                "masked_image_url": url,
                "confidence": score,
                "candidates": [],  # filled below
            }
        )
        masked_urls.append(url)
        logger.info(f"  → '{label}' uploaded: {url}")

    # Step 2: fan out Lens calls in parallel
    if objects:
        with ThreadPoolExecutor(max_workers=LENS_MAX_PARALLEL) as pool:
            lens_results = list(pool.map(call_google_lens, masked_urls))
        for obj, raw_matches in zip(objects, lens_results):
            obj["candidates"] = filter_in_stock(raw_matches)
            obj["raw_match_count"] = len(raw_matches)

    summary = {
        "image_url": image_url,
        "prompt_source": prompt_source,  # "gemini" | "caller" | "default-*"
        "prompts_used": prompts,
        "objects_detected": len(objects),
        "total_purchasable_candidates": sum(len(o["candidates"]) for o in objects),
    }
    logger.info(f"Done: {summary}")
    return {"summary": summary, "objects": objects}


# ─────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Lens-Roboflow Test Project")

# CORS — fully open. The frontend will be hosted separately (Vercel) and call
# this backend (Railway) cross-origin. ``allow_origins=["*"]`` only works when
# ``allow_credentials`` is False, which it is — we don't need cookies/auth here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


@app.post("/api/process")
async def process_endpoint(body: ProcessRequest) -> Dict[str, Any]:
    """Run the discovery → segment → upload → lens → filter pipeline."""
    try:
        return await asyncio.to_thread(
            process_pipeline, body.image_url, body.prompts
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=502, detail=f"upstream HTTP error: {e}"
        ) from e
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
def health() -> Dict[str, str]:
    """Liveness probe + config check."""
    return {
        "status": "ok",
        "roboflow_key": "set" if ROBOFLOW_API_KEY else "missing",
        "search_api_key": "set" if SEARCH_API_KEY else "missing",
        "google_api_key": "set" if GOOGLE_API_KEY else "missing",
        "r2": "set" if R2_BUCKET_NAME and R2_PUBLIC_URL else "missing",
        "gemini_model": GEMINI_MODEL,
    }


# Serve the frontend at "/" — has to be mounted last so /api/* still routes.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
