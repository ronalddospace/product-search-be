# product-search-be

FastAPI backend for the Room → Objects → Products prototype. Pairs with
[product-search-fe](https://github.com/DoSpace-Inc/product-search-fe).

`POST /api/process` takes a public room-image URL and returns
purchasable Google Lens matches for each detected object:

```
{
  "summary": { "image_url", "objects_detected", "total_purchasable_candidates" },
  "objects": [
    {
      "label": "sofa",
      "masked_image_url": "https://<r2-public>/test-project/cutouts/<uuid>.png",
      "confidence": 0.87,
      "raw_match_count": 41,
      "candidates": [
        { "title", "source", "link", "thumbnail", "price", "currency", ... }
      ]
    }
  ]
}
```

Pipeline (all server-side):

1. Download the room image.
2. Roboflow SAM3 `concept_segment` with a hardcoded prompt list
   (`DEFAULT_PROMPTS` in `main.py`).
3. For each detection: bbox-crop the room + apply the RLE mask as alpha →
   transparent PNG.
4. boto3 → upload the PNG to Cloudflare R2 → public URL.
5. Google Lens (SearchAPI.io, `search_type=products`) against every cutout
   URL in parallel.
6. Keep only candidates that have a price and are not flagged out-of-stock.

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

See `.env.example`. The required keys are `ROBOFLOW_API_KEY`,
`SEARCH_API_KEY`, and the five `R2_*` values. Optional tuning knobs:
`ROBOFLOW_CONFIDENCE`, `LENS_COUNTRY`, `LENS_LANGUAGE`,
`LENS_MAX_PARALLEL`.
