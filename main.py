"""Single-process test project.

POST /api/process — full pipeline:
  1. Download the room image.
  2. Vision LLM (OpenAI or Gemini, picked via VISION_PROVIDER) emits a
     SAM3-ready prompt list by looking at the room.
  3. Roboflow SAM3 ``concept_segment`` against those prompts.
  4. For each detected object: crop to bbox + apply mask as alpha → transparent
     PNG → upload to R2 → get a public URL.
  5. Run SearchAPI.io Google Lens (search_type=products) against each public URL
     in parallel.
  6. Keep only candidates that look genuinely purchasable
     (has a price AND is not flagged out-of-stock).

GET / — serves the frontend (../frontend/index.html and friends).

Env vars required — copy .env.example to .env and fill in:
  VISION_PROVIDER (openai|gemini)
  OPENAI_API_KEY  (when VISION_PROVIDER=openai)
  GOOGLE_API_KEY  (when VISION_PROVIDER=gemini)
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
import time
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

# Lens config
LENS_COUNTRY = os.environ.get("LENS_COUNTRY", "us")
LENS_LANGUAGE = os.environ.get("LENS_LANGUAGE", "en")
LENS_MAX_PARALLEL = int(os.environ.get("LENS_MAX_PARALLEL", "12"))

# Vision LLM — picks the provider that looks at the room image and emits
# SAM3-ready prompts. "openai" or "gemini".
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "openai").strip().lower()

# Gemini config
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ─────────────────────────────────────────────────────────────────────
# Vision LLM clients (lazy, thread-safe) — Gemini + OpenAI
# ─────────────────────────────────────────────────────────────────────

_gemini_client = None
_gemini_client_lock = threading.Lock()

_openai_client = None
_openai_client_lock = threading.Lock()


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


def _openai_get_client():
    """Lazy thread-safe ``openai.OpenAI`` client."""
    global _openai_client
    if _openai_client is None:
        with _openai_client_lock:
            if _openai_client is None:
                if not OPENAI_API_KEY:
                    raise RuntimeError("OPENAI_API_KEY missing in .env")
                from openai import OpenAI

                _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# Max items the vision LLM is asked to return. Caps two things at once:
#   - cost / latency of the downstream SAM3 + Lens fan-out;
#   - Roboflow's per-call cap of 16 prompts (we stay safely below).
# Production tuning: bump if you want more recall, lower for faster runs.
VISION_MAX_OBJECTS = 12


# Instruction we send to the vision LLM. The model returns a JSON array where
# each entry has BOTH a bare ``object_type`` (the category word — used as the
# Google Lens text query) and a richer ``sam3_prompt`` (colour + material +
# category — fed to Roboflow SAM3 for segmentation). Splitting them lets us
# anchor Lens to the right product category (so a side-profile TV doesn't
# come back as combs/pianos) while still giving SAM3 the colour cue it needs
# to find the right pixels.
DISCOVERY_INSTRUCTION = f"""\
You are looking at a photo of a room. List the most visually prominent
furniture and decor pieces — at MOST {VISION_MAX_OBJECTS} items, ranked by
size and visual prominence. If there are more than {VISION_MAX_OBJECTS}
distinct items in the room, keep the {VISION_MAX_OBJECTS} that dominate
the frame and drop the rest.

For each item, output TWO fields:

  - "object_type"  — the bare product category, 1-2 words, lower-case.
                     Examples: "sofa", "armchair", "coffee table", "floor lamp",
                               "rug", "artwork", "plant", "tv", "pouf",
                               "side table", "chandelier".

  - "sam3_prompt"  — 3-4 word visual description in the form
                     <colour> <material-or-texture> <category>.
                     Examples: "gray linen sofa", "round oak coffee table",
                               "tall brass floor lamp", "beige fabric pouf",
                               "black flat-screen tv".

Rules:
  - Hard limit: at most {VISION_MAX_OBJECTS} items in the output array.
  - Colour FIRST in ``sam3_prompt`` — SAM3 matches colour most reliably.
  - ``object_type`` is the SAME category word that ends the ``sam3_prompt``.
  - Skip walls, floor, and ceiling — handled separately.
  - Skip tiny clutter (books, glasses, throw pillows on a sofa).
  - If two similar items are visible (e.g. two matching chairs), list once.
  - Return ONLY a JSON array of objects. No prose, no markdown fences.

Example output:
[
  {{"object_type": "sofa",         "sam3_prompt": "gray linen sofa"}},
  {{"object_type": "coffee table", "sam3_prompt": "round oak coffee table"}},
  {{"object_type": "floor lamp",   "sam3_prompt": "tall brass floor lamp"}}
]
"""


def _coerce_prompt_item(item: Any) -> Optional[Dict[str, str]]:
    """Normalise one Gemini list entry into ``{object_type, sam3_prompt}``."""
    if isinstance(item, dict):
        ot = str(item.get("object_type") or "").strip()
        sp = str(item.get("sam3_prompt") or "").strip()
        # Fall back to whichever field is present.
        if not sp and ot:
            sp = ot
        if not ot and sp:
            # Use the last word as the category if Gemini forgot object_type.
            ot = sp.rsplit(" ", 1)[-1] if " " in sp else sp
        if not sp:
            return None
        return {"object_type": ot.lower(), "sam3_prompt": sp}
    if isinstance(item, str) and item.strip():
        # Tolerate the old flat-string shape: same value for both fields.
        sp = item.strip()
        ot = sp.rsplit(" ", 1)[-1].lower() if " " in sp else sp.lower()
        return {"object_type": ot, "sam3_prompt": sp}
    return None


def _parse_prompt_list(text: str) -> List[Dict[str, str]]:
    """Pull a JSON list of ``{object_type, sam3_prompt}`` items out of Gemini.

    Tolerates: markdown fences, the old flat-string shape, missing fields.
    """
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
    out: List[Dict[str, str]] = []
    for item in data:
        coerced = _coerce_prompt_item(item)
        if coerced:
            out.append(coerced)
    return out


def _discover_via_gemini(
    room_image: Image.Image,
) -> List[Dict[str, str]]:
    """Ask Gemini to enumerate visible objects. Returns parsed prompt list."""
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
    return _parse_prompt_list(
        "\n".join(p.text for p in parts if getattr(p, "text", None))
    )


def _discover_via_openai(
    room_image: Image.Image,
) -> List[Dict[str, str]]:
    """Ask OpenAI (vision chat) to enumerate visible objects.

    The room image is base64-encoded inline as a ``data:`` URL — keeps the
    call self-contained and avoids needing a public URL just for discovery.
    """
    client = _openai_get_client()
    buf = io.BytesIO()
    room_image.save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DISCOVERY_INSTRUCTION},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    text = (choices[0].message.content or "") if choices[0].message else ""
    return _parse_prompt_list(text)


def discover_objects(
    room_image: Image.Image,
) -> List[Dict[str, str]]:
    """Dispatch to the configured vision provider and return prompt items.

    Returns one dict per object: ``{"object_type": str, "sam3_prompt": str}``.
    The result is hard-capped at ``VISION_MAX_OBJECTS`` entries — the model
    is already instructed to stay under this cap, but we slice defensively in
    case it returns more (keeping the first N, which the prompt asks it to
    order by visual prominence).
    """
    if VISION_PROVIDER == "openai":
        parsed = _discover_via_openai(room_image)
    elif VISION_PROVIDER == "gemini":
        parsed = _discover_via_gemini(room_image)
    else:
        raise RuntimeError(
            f"Unknown VISION_PROVIDER={VISION_PROVIDER!r}; "
            f"expected 'openai' or 'gemini'"
        )
    if len(parsed) > VISION_MAX_OBJECTS:
        logger.info(
            f"{VISION_PROVIDER} returned {len(parsed)} items; trimming to "
            f"{VISION_MAX_OBJECTS} most prominent"
        )
        parsed = parsed[:VISION_MAX_OBJECTS]
    return parsed


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
    """Fetch an image URL into a fully-decoded PIL.Image (RGB).

    PIL's ``Image.open`` is **lazy** — the decoder isn't actually run until
    something touches pixel data (e.g. ``.crop()``). That's a problem in
    ``/api/search`` where multiple worker threads call ``.crop()`` on the
    same image concurrently: each one races to trigger the deferred decode
    and PIL's internal state corrupts, producing "image file is truncated"
    and "unrecognized data stream contents" errors at random.

    Calling ``.load()`` here forces the decode to happen synchronously on the
    download thread, so worker threads later see a stable, fully-resident
    image. ``convert("RGB")`` already loads on most paths but is not
    contractually guaranteed to — the explicit ``.load()`` is the belt.
    """
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.load()
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
    ``bbox`` is (left, top, right, bottom). PNG is saved without
    ``optimize=True`` — that flag triples encoding time for marginal size
    savings, which we don't need.
    """
    left, top, right, bottom = bbox
    rgb_crop = image.crop(bbox)
    mask_crop = Image.fromarray(mask[top:bottom, left:right], mode="L")
    rgba = Image.new("RGBA", rgb_crop.size, (0, 0, 0, 0))
    rgba.paste(rgb_crop, (0, 0), mask=mask_crop)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Roboflow SAM3 — concept_segment
# ─────────────────────────────────────────────────────────────────────

# Roboflow caps each concept_segment call at 16 text prompts. Chunk above
# this size and fan out in parallel — order is preserved when results are
# stitched back together.
SAM3_MAX_PROMPT_BATCH_SIZE = 16


def _call_sam3_batch(
    image_b64: str, prompts: List[str]
) -> List[Dict[str, Any]]:
    """POST one ≤16-prompt batch to Roboflow's SAM3 endpoint."""
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


def call_sam3(image_b64: str, prompts: List[str]) -> List[Dict[str, Any]]:
    """Run Roboflow SAM3 concept_segment, chunking above the 16-prompt cap.

    Returns ``prompt_results`` aligned to the input ``prompts`` list (one
    entry per input prompt, in order). When more than
    ``SAM3_MAX_PROMPT_BATCH_SIZE`` prompts are supplied, the call is split
    into parallel batches via ``ThreadPoolExecutor`` and the per-batch
    responses are concatenated.
    """
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY missing in .env")
    if not prompts:
        return []

    if len(prompts) <= SAM3_MAX_PROMPT_BATCH_SIZE:
        return _call_sam3_batch(image_b64, prompts)

    batches = [
        prompts[i : i + SAM3_MAX_PROMPT_BATCH_SIZE]
        for i in range(0, len(prompts), SAM3_MAX_PROMPT_BATCH_SIZE)
    ]
    logger.info(
        f"SAM3: chunking {len(prompts)} prompts into {len(batches)} "
        f"parallel batches (cap={SAM3_MAX_PROMPT_BATCH_SIZE})"
    )
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        per_batch_results = list(
            pool.map(lambda b: _call_sam3_batch(image_b64, b), batches)
        )
    out: List[Dict[str, Any]] = []
    for results in per_batch_results:
        out.extend(results)
    return out


# ─────────────────────────────────────────────────────────────────────
# Google Lens (SearchAPI.io) — search_type=products
# ─────────────────────────────────────────────────────────────────────


def call_google_lens(
    image_url: str, query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Call Google Lens products search for one image URL.

    When ``query`` is provided it's sent as the ``q`` parameter, which biases
    the result pool toward items semantically matching the text while Lens
    still ranks visually within that pool. This is how we keep a side-profile
    TV from returning combs / pianos, or a pouf from returning glasses —
    Gemini's SAM3 prompt ("black flat-screen tv", "beige fabric pouf", …) is
    a strong category anchor that the visual signal alone can't provide.

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
    if query:
        params["q"] = query
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
    """Pick the prompt list from caller override or vision discovery.

    Returns ``(items, prompt_source)`` where each ``item`` is
    ``{"object_type", "sam3_prompt"}`` and ``prompt_source`` is either
    ``"caller"`` or the active provider name (``"openai"`` / ``"gemini"``).
    Raises ``RuntimeError`` if discovery produces no usable prompts —
    there is no hard-coded fallback.
    """
    if override:
        items = [_coerce_prompt_item(p) for p in override]
        items = [i for i in items if i is not None]
        logger.info(f"Using {len(items)} caller-supplied prompts")
        return items, "caller"
    discovered = discover_objects(room_img)
    if not discovered:
        raise RuntimeError(
            f"{VISION_PROVIDER} discovery returned no usable prompts"
        )
    logger.info(
        f"{VISION_PROVIDER} discovered {len(discovered)} items: {discovered}"
    )
    return discovered, VISION_PROVIDER


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
    (% of image), SAM3's confidence, and BOTH ``object_type`` (used as the
    Lens text query in the search phase) and ``sam3_prompt`` (the colour /
    material / category string that was fed to SAM3). No cropping or R2
    upload at this phase — that happens during search.
    """
    h, w = room_img.height, room_img.width
    items, prompt_source = _resolve_prompts(room_img, override_prompts)

    # SAM3 sees the rich ``sam3_prompt`` string (colour first, category last).
    sam3_prompts = [it["sam3_prompt"] for it in items]
    image_b64 = encode_jpeg_b64(room_img)
    logger.info(f"SAM3: {len(sam3_prompts)} prompts")
    sam_results = call_sam3(image_b64, sam3_prompts)

    objects: List[Dict[str, Any]] = []
    for item, result in zip(items, sam_results):
        object_type = item["object_type"]
        sam3_prompt = item["sam3_prompt"]
        label = sam3_prompt  # what we show in logs + the FE chip fallback
        # Roboflow concept_segment response shape:
        #   prompt_results[i] = {
        #     "predictions": [
        #       { "masks": {size, counts}, "confidence": float, ... },
        #       ...
        #     ]
        #   }
        # Per prompt we pick the highest-confidence prediction whose mask is
        # non-empty (mirrors production sam3.py behaviour).
        predictions = result.get("predictions") or []
        if not predictions:
            logger.info(f"  → '{label}' SAM3 returned no predictions")
            continue
        # Sort by confidence (desc) and accept the first usable one.
        predictions = sorted(
            predictions,
            key=lambda p: float(p.get("confidence", 0.0)),
            reverse=True,
        )
        chosen_rle = None
        chosen_conf = None
        for pred in predictions:
            rle = pred.get("masks") or pred.get("rle")
            if not rle:
                continue
            mask = rle_to_mask(rle, h, w)
            if mask is None or mask.max() == 0:
                continue
            chosen_rle = rle
            chosen_mask = mask
            chosen_conf = float(pred.get("confidence", 0.0))
            break
        if chosen_rle is None:
            logger.info(
                f"  → '{label}' all predictions empty "
                f"(top conf={float(predictions[0].get('confidence', 0)):.2f})"
            )
            continue
        bbox = bbox_of_mask(chosen_mask)
        if bbox is None:
            continue
        dot_x_pct, dot_y_pct = dot_pct_of_mask(chosen_mask)
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,  # kept for backwards compat (== sam3_prompt)
                "object_type": object_type,  # for Lens ``q`` in search phase
                "sam3_prompt": sam3_prompt,  # for SAM3 + display
                "confidence": chosen_conf,
                "mask_rle": _normalize_rle_counts(chosen_rle),
                "bbox": list(bbox),  # [left, top, right, bottom] in pixels
                "dot": [dot_x_pct, dot_y_pct],  # centroid in % of image
            }
        )
        logger.info(
            f"  → '{sam3_prompt}' (type={object_type}) conf={chosen_conf:.2f} "
            f"bbox={bbox} dot=({dot_x_pct:.1f}%, {dot_y_pct:.1f}%)"
        )

    return {
        "prompt_source": prompt_source,
        "prompts_used": items,  # list of {object_type, sam3_prompt} dicts
        "objects": objects,
    }


# ───────────────────────── Phase B: search ─────────────────────────


class DetectedObjectIn(BaseModel):
    """One detected object echoed back from the FE."""

    id: str
    label: str
    object_type: Optional[str] = None  # falls back to ``label`` if missing
    sam3_prompt: Optional[str] = None
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
    """Crop + R2 upload + Lens + filter per detected object.

    All per-object work runs concurrently in a ``ThreadPoolExecutor``: each
    worker does the full crop → R2 upload → Lens fan-out → in-stock filter
    for one object. Because every step inside the worker is either IO-bound
    (R2 / Lens HTTP) or releases the GIL (PIL crop/encode), threads are the
    right tool here. The wall-clock for the whole phase is dominated by the
    slowest single object, not the sum.
    """
    h, w = room_img.height, room_img.width

    def process(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Crop → upload → Lens → filter for one detection. Returns the
        enriched record, or ``None`` if the mask was unusable or upload
        failed (logged but non-fatal — other objects keep going)."""
        t0 = time.monotonic()
        label = obj.get("label")
        # The Lens text-query uses the bare ``object_type`` ("sofa", "tv",
        # "pouf") rather than the full SAM3 prompt ("black flat-screen tv").
        # The category word is enough to anchor Lens to the right product
        # type while leaving room for visual ranking inside that category.
        # ``label`` is kept as the fallback for older payloads.
        lens_query = obj.get("object_type") or obj.get("label") or ""

        mask = rle_to_mask(obj.get("mask_rle") or {}, h, w)
        if mask is None or mask.max() == 0:
            logger.warning(f"  → '{label}' mask empty; skipping")
            return None
        bbox = tuple(obj.get("bbox") or []) or bbox_of_mask(mask)
        if not bbox:
            return None

        try:
            t_crop_start = time.monotonic()
            png = cutout_png(room_img, mask, tuple(bbox))
            t_upload_start = time.monotonic()
            key = f"poc-object/{uuid.uuid4().hex}.png"
            url = upload_png_to_r2(png, key)
            t_lens_start = time.monotonic()
            raw_matches = call_google_lens(url, query=lens_query)
            t_done = time.monotonic()
        except Exception as e:
            logger.warning(f"  → '{label}' failed: {e}")
            return None

        candidates = filter_in_stock(raw_matches)
        logger.info(
            f"  → '{label}' (q='{lens_query}') "
            f"crop={(t_upload_start - t_crop_start) * 1000:.0f}ms "
            f"upload={(t_lens_start - t_upload_start) * 1000:.0f}ms "
            f"lens={(t_done - t_lens_start) * 1000:.0f}ms "
            f"total={(t_done - t0) * 1000:.0f}ms "
            f"→ {len(candidates)}/{len(raw_matches)} purchasable"
        )
        return {
            **obj,
            "masked_image_url": url,
            "candidates": candidates,
            "raw_match_count": len(raw_matches),
        }

    if not objects:
        return []

    # Cap workers at LENS_MAX_PARALLEL but never below 1, never above the
    # actual number of objects (no point spawning idle threads).
    workers = max(1, min(LENS_MAX_PARALLEL, len(objects)))
    t_phase = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # ``pool.map`` preserves the order of ``objects``, so the FE sees
        # detections in the same order they were detected.
        results = [r for r in pool.map(process, objects) if r is not None]
    logger.info(
        f"search phase: {len(results)}/{len(objects)} objects in "
        f"{(time.monotonic() - t_phase) * 1000:.0f}ms "
        f"(workers={workers})"
    )
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

    # R2 room upload and Gemini→SAM3 detection are independent — they take
    # roughly equal time (~500ms vs ~3-6s respectively), so we run them
    # concurrently. ``asyncio.gather`` returns once BOTH finish.
    try:
        room_url, detection = await asyncio.gather(
            asyncio.to_thread(
                upload_room_to_r2, raw, room_image.filename or "room.jpg"
            ),
            asyncio.to_thread(detect_objects, pil_img, override_prompts),
        )
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
        "openai_key": "set" if OPENAI_API_KEY else "missing",
        "r2": "set" if R2_BUCKET_NAME and R2_PUBLIC_URL else "missing",
        "vision_provider": VISION_PROVIDER,
        "gemini_model": GEMINI_MODEL,
        "openai_model": OPENAI_MODEL,
    }


# Serve the frontend at "/" — only when running locally as a sibling of the
# backend directory. In production (e.g. Railway), the backend repo is the
# deploy root and there is no neighbouring ``frontend/`` folder; trying to
# mount a missing path would crash uvicorn at import time.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
    )
    logger.info(f"Mounted static frontend at / from {FRONTEND_DIR}")
else:
    logger.info(
        f"No sibling frontend/ at {FRONTEND_DIR} — running API-only "
        f"(set BACKEND_URL on the FE deploy to this service's URL)"
    )
