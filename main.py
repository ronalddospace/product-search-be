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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def upload_room_to_r2(image_bytes: bytes, filename: str) -> str:
    """Upload the user-uploaded room image to R2 and return its public URL."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[ext]
    key = f"poc-room/room_{uuid.uuid4().hex}{ext}"
    client = _r2_client()
    client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
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


def dot_pct_of_mask(mask: np.ndarray) -> tuple:
    """Return the mask centroid as (x_pct, y_pct) — used for FE overlay dots."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (50.0, 50.0)
    h, w = mask.shape
    return (float(xs.mean()) / w * 100.0, float(ys.mean()) / h * 100.0)


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


def _normalize_rle_counts(rle: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of an RLE dict (counts as str, size as list)."""
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle.get("size") or []), "counts": counts}


def _resolve_prompts(
    room_img: Image.Image, override: Optional[List[str]]
) -> tuple:
    """Pick SAM3 prompts to use, with Gemini discovery + safe fallback.

    Returns ``(prompts, prompt_source)`` where ``prompt_source`` is one of
    ``"caller"`` / ``"gemini"`` / ``"default-empty-discovery"`` /
    ``"default-error"``.
    """
    if override:
        logger.info(f"Using {len(override)} caller-supplied prompts")
        return override, "caller"
    try:
        discovered = discover_sam3_prompts(room_img)
    except Exception as e:
        logger.warning(f"Gemini discovery failed ({e}) — using DEFAULT_PROMPTS")
        return DEFAULT_PROMPTS, "default-error"
    if discovered:
        logger.info(
            f"Gemini ({GEMINI_MODEL}) discovered {len(discovered)} prompts: "
            f"{discovered}"
        )
        return discovered, "gemini"
    logger.warning("Gemini returned no usable prompts — using DEFAULT_PROMPTS")
    return DEFAULT_PROMPTS, "default-empty-discovery"


# ───────────────────────── Phase A: detection ─────────────────────────


class DetectRequest(BaseModel):
    """Optional JSON overrides for POST /api/detect."""

    # Override Gemini discovery with a fixed prompt list (used for debugging).
    prompts: Optional[List[str]] = None


def detect_objects(
    room_img: Image.Image, override_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Gemini → SAM3 → return one entry per detected object.

    Each ``object`` carries the raw RLE mask, its pixel bbox, a centroid dot
    (% of image), and SAM3's confidence. No cropping or R2 upload at this
    phase — that happens during search.
    """
    h, w = room_img.height, room_img.width
    prompts, prompt_source = _resolve_prompts(room_img, override_prompts)

    image_b64 = encode_jpeg_b64(room_img)
    logger.info(f"SAM3: {len(prompts)} prompts")
    sam_results = call_sam3(image_b64, prompts)

    objects: List[Dict[str, Any]] = []
    for label, result in zip(prompts, sam_results):
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
        dot_x_pct, dot_y_pct = dot_pct_of_mask(mask)
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,
                "confidence": result.get("score"),
                "mask_rle": _normalize_rle_counts(rle),
                "bbox": list(bbox),  # [left, top, right, bottom] in pixels
                "dot": [dot_x_pct, dot_y_pct],  # centroid in % of image
            }
        )
        logger.info(
            f"  → '{label}' bbox={bbox} dot=({dot_x_pct:.1f}%, {dot_y_pct:.1f}%)"
        )

    return {
        "prompt_source": prompt_source,
        "prompts_used": prompts,
        "objects": objects,
    }


# ───────────────────────── Phase B: search ─────────────────────────


class DetectedObjectIn(BaseModel):
    """One detected object echoed back from the FE."""

    id: str
    label: str
    confidence: Optional[float] = None
    mask_rle: Dict[str, Any]
    bbox: List[int]
    dot: List[float]


class SearchRequest(BaseModel):
    """JSON body for POST /api/search."""

    room_image_url: str
    objects: List[DetectedObjectIn]


def search_for_objects(
    room_img: Image.Image, objects: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Crop + R2 upload + Lens + filter per detected object."""
    h, w = room_img.height, room_img.width
    results: List[Dict[str, Any]] = []
    masked_urls: List[str] = []

    for obj in objects:
        mask = rle_to_mask(obj.get("mask_rle") or {}, h, w)
        if mask is None or mask.max() == 0:
            logger.warning(f"  → '{obj.get('label')}' mask empty; skipping")
            continue
        bbox = tuple(obj.get("bbox") or []) or bbox_of_mask(mask)
        if not bbox:
            continue
        png = cutout_png(room_img, mask, tuple(bbox))
        key = f"poc-object/{uuid.uuid4().hex}.png"
        try:
            url = upload_png_to_r2(png, key)
        except Exception as e:
            logger.warning(f"R2 upload failed for {obj.get('label')}: {e}")
            continue
        results.append(
            {
                **obj,
                "masked_image_url": url,
                "candidates": [],
                "raw_match_count": 0,
            }
        )
        masked_urls.append(url)
        logger.info(f"  → '{obj.get('label')}' uploaded: {url}")

    if masked_urls:
        with ThreadPoolExecutor(max_workers=LENS_MAX_PARALLEL) as pool:
            lens_results = list(pool.map(call_google_lens, masked_urls))
        for obj, raw_matches in zip(results, lens_results):
            obj["candidates"] = filter_in_stock(raw_matches)
            obj["raw_match_count"] = len(raw_matches)

    return results


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


def _pipeline_error_to_http(e: Exception) -> HTTPException:
    """Map pipeline exceptions to clean HTTPExceptions for the endpoints."""
    if isinstance(e, RuntimeError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, requests.HTTPError):
        return HTTPException(status_code=502, detail=f"upstream HTTP error: {e}")
    return HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect")
async def detect_endpoint(
    room_image: UploadFile = File(...),
    prompts: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a room image, run Gemini → SAM3, return detected objects.

    The original room is also uploaded to R2 so the search phase can fetch
    it back without the FE having to re-send the binary.

    Form fields:
      - ``room_image``: the uploaded photo (required).
      - ``prompts``: optional JSON-encoded list of strings — if set, skips
        the Gemini discovery and forces SAM3 to use exactly these prompts.

    Returns:
      ``{ room_image_url, image_width, image_height, prompt_source,
          prompts_used, objects: [{ id, label, confidence, mask_rle,
                                    bbox, dot }] }``
    """
    raw = await room_image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        pil_img = Image.open(io.BytesIO(raw))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"could not decode image: {e}"
        ) from e

    override_prompts: Optional[List[str]] = None
    if prompts:
        try:
            parsed = json.loads(prompts)
            if isinstance(parsed, list):
                override_prompts = [str(p).strip() for p in parsed if str(p).strip()]
        except Exception:
            logger.warning(f"ignored unparsable prompts form-field: {prompts!r}")

    try:
        room_url = await asyncio.to_thread(
            upload_room_to_r2, raw, room_image.filename or "room.jpg"
        )
        detection = await asyncio.to_thread(detect_objects, pil_img, override_prompts)
    except Exception as e:
        logger.exception("detect failed")
        raise _pipeline_error_to_http(e) from e

    return {
        "room_image_url": room_url,
        "image_width": pil_img.width,
        "image_height": pil_img.height,
        **detection,
    }


@app.post("/api/search")
async def search_endpoint(body: SearchRequest) -> Dict[str, Any]:
    """Given the detection result, crop + R2 + Lens + filter per object."""
    try:
        pil_img = await asyncio.to_thread(download_image, body.room_image_url)
        objects = await asyncio.to_thread(
            search_for_objects,
            pil_img,
            [o.dict() for o in body.objects],
        )
    except Exception as e:
        logger.exception("search failed")
        raise _pipeline_error_to_http(e) from e

    return {
        "room_image_url": body.room_image_url,
        "objects_returned": len(objects),
        "total_purchasable_candidates": sum(len(o["candidates"]) for o in objects),
        "objects": objects,
    }


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
