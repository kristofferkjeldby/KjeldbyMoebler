"""Persistent regeneration server for the image review UI (catalog/review_images.py).

catalog/generate_product_images.py loads the SDXL + ControlNet pipelines
fresh on every invocation (~20-30s just for model loading) — fine for a
batch run, unworkable for "click Regenerate, wait a few seconds" one at a
time during review. This server pays that cost once at startup and keeps
the pipelines warm in memory for every subsequent request.

Runs on the GPU pod. Copy alongside config.py, catalog/image_service.py,
catalog/generate_product_images.py, catalog/data/catalog.json, and
catalog/data/image_prompt_vocab.json (same layout as
generate_product_images.py already expects), then:

    python -m catalog.regen_server

Serves on port 8091. The Mac-side review server reaches it through an SSH
tunnel (see catalog/review_images.py's module docstring).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.generate_product_images import generate_product, load_pipelines  # noqa: E402
from config import CATALOG_PATH  # noqa: E402

VOCAB_PATH = Path(__file__).resolve().parent / "data" / "image_prompt_vocab.json"

app = FastAPI()

products_by_sku: dict[str, dict] = {}
vocab: dict = {}
base_pipe = None
cn_pipe = None
rembg_session = None


@app.on_event("startup")
def startup() -> None:
    global products_by_sku, vocab, base_pipe, cn_pipe, rembg_session
    products = json.loads(CATALOG_PATH.read_text())
    products_by_sku = {p["sku"]: p for p in products}
    vocab = json.loads(VOCAB_PATH.read_text())
    print("Loading SDXL + ControlNet pipelines...")
    base_pipe, cn_pipe, rembg_session = load_pipelines()
    print("Ready.")


@app.post("/regenerate/{sku}")
def regenerate(sku: str, seed_offset: int = 0) -> dict:
    product = products_by_sku.get(sku)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku!r}")
    ok, error = generate_product(product, vocab, base_pipe, cn_pipe, rembg_session, seed_offset, force=True)
    return {"ok": ok, "error": error}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091)
