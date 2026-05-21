# product-search-be

FastAPI backend for the Room → Objects → Products prototype. Pairs with
[product-search-fe](https://github.com/DoSpace-Inc/product-search-fe).

Two endpoints — a deliberate split so the FE can show detection
results before paying for the search step.

### `POST /api/detect` — multipart upload

Accepts `room_image` (file). Optional `prompts` (JSON-encoded list) to
force a specific SAM3 prompt list and skip Gemini.

```
{
  "room_image_url": "https://<r2>/test-project/rooms/<uuid>.jpg",
  "image_width": 1024, "image_height": 768,
  "prompt_source": "gemini",
  "prompts_used": ["gray linen sofa", "round oak coffee table", ...],
  "objects": [
    {
      "id": "obj_1",
      "label": "gray linen sofa",
      "confidence": 0.87,
      "mask_rle": { "size": [768, 1024], "counts": "..." },
      "bbox": [left, top, right, bottom],
      "dot": [x_pct, y_pct]
    }
  ]
}
```

Pipeline (all server-side):

1. Upload the user's room image to R2 → public URL.
2. Gemini Flash Lite enumerates visible furniture as 3-4 word SAM3
   prompts in `<colour> <material> <category>` order. On failure or
   empty response, falls back to `DEFAULT_PROMPTS`.
3. Roboflow SAM3 `concept_segment` runs all prompts in one call.
4. Each detection's RLE mask is decoded → bbox + centroid dot are
   computed and returned alongside the raw RLE.

### `POST /api/search` — JSON

```
{
  "room_image_url": "https://<r2>/test-project/rooms/<uuid>.jpg",
  "objects": [ ...the array returned by /api/detect... ]
}
```

For each object the backend re-downloads the room, decodes the RLE,
cuts the masked region onto a transparent PNG, uploads it to R2, and
runs Google Lens (SearchAPI.io, `search_type=products`) for every
cutout in parallel. The response is the same `objects` array with two
fields appended: `masked_image_url` and `candidates`.

`candidates` keeps only matches that have a price AND are not flagged
out-of-stock (`stock_information` containing "out of stock", "sold
out", or "unavailable").

CORS is wide open (`allow_origins=["*"]`) so the frontend on Vercel can call
this directly.

## Local dev

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys + R2 creds
uvicorn main:app --reload --port 8000
```

Sanity check: `curl http://localhost:8000/api/health`.

## Deploy to Railway

1. Create a Railway service from this repo.
2. Add every variable from `.env.example` in the Railway dashboard.
3. Railway picks up the `Procfile` automatically. Start command is:
   `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Note the public URL Railway assigns. Use it as the `BACKEND_URL` in
   the frontend repo.

## Env vars

See `.env.example`. The required keys are `GOOGLE_API_KEY` (Gemini),
`ROBOFLOW_API_KEY`, `SEARCH_API_KEY`, and the five `R2_*` values.
Optional tuning knobs: `GEMINI_MODEL` (default `gemini-3.1-flash-lite`),
`ROBOFLOW_CONFIDENCE`, `LENS_COUNTRY`, `LENS_LANGUAGE`,
`LENS_MAX_PARALLEL`.
