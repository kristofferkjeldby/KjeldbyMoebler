"""Local review UI for generated product images — step through products one
at a time, Accept (persisted to disk, so a review session survives closing
the tool) or Regenerate (calls the pod's persistent regen server, then
syncs just that product's photos back).

Runs entirely on the Mac. Requires an SSH tunnel to the pod's regen server
(catalog/regen_server.py) open first:

    ssh -f -N -L 8091:localhost:8091 -p <port> -i ~/.ssh/id_ed25519 root@<pod-ip>

Then:

    python -m catalog.review_images

and open http://localhost:8092/. Regenerate also needs direct (non-tunneled)
SSH/rsync access to the same pod to pull the freshly generated files back —
set REVIEW_POD_SSH below to match your current pod connection.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.image_service import CATEGORY_FOLDERS, IMAGES_ROOT, PRODUCTS_ROOT  # noqa: E402
from config import CATALOG_PATH  # noqa: E402

# Update these to match whatever pod is currently up (see module docstring)
# — the same host/port/key used for the SSH tunnel and for every manual
# scp/rsync this session.
REVIEW_POD_HOST = "157.157.221.177"
REVIEW_POD_PORT = "30330"
REVIEW_POD_KEY = str(Path.home() / ".ssh" / "id_ed25519")
REGEN_SERVER_URL = "http://localhost:8091"

STATE_PATH = Path(__file__).resolve().parent / "data" / "image_review_state.json"

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGES_ROOT), name="images")

products: list[dict] = json.loads(CATALOG_PATH.read_text())
sku_index = {p["sku"]: i for i, p in enumerate(products)}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def product_payload(product: dict, accepted_count: int) -> dict:
    sku = product["sku"]
    category = product["category"]
    cat_folder = CATEGORY_FOLDERS.get(category, category)
    return {
        "sku": sku,
        "name": product["name"],
        "category": category,
        "image_url": f"/images/products/{cat_folder}/{sku}/default.jpg?v={int(time.time())}",
        "accepted_count": accepted_count,
        "total_count": len(products),
    }


def find_next(after_sku: str | None) -> dict | None:
    state = load_state()
    start = sku_index.get(after_sku, -1) + 1 if after_sku else 0
    accepted_count = sum(1 for v in state.values() if v.get("status") == "accepted")
    for product in products[start:]:
        if state.get(product["sku"], {}).get("status") != "accepted":
            return product_payload(product, accepted_count)
    return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "review.html")


@app.get("/api/next")
def api_next(after: str | None = None) -> dict:
    result = find_next(after)
    if result is None:
        state = load_state()
        accepted_count = sum(1 for v in state.values() if v.get("status") == "accepted")
        return {"done": True, "accepted_count": accepted_count, "total_count": len(products)}
    return result


@app.get("/api/product/{sku}")
def api_product(sku: str) -> dict:
    if sku not in sku_index:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku!r}")
    state = load_state()
    accepted_count = sum(1 for v in state.values() if v.get("status") == "accepted")
    return product_payload(products[sku_index[sku]], accepted_count)


@app.post("/api/accept/{sku}")
def api_accept(sku: str) -> dict:
    if sku not in sku_index:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku!r}")
    state = load_state()
    entry = state.setdefault(sku, {})
    entry["status"] = "accepted"
    save_state(state)
    return {"ok": True}


@app.post("/api/regenerate/{sku}")
def api_regenerate(sku: str) -> dict:
    if sku not in sku_index:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku!r}")
    product = products[sku_index[sku]]

    state = load_state()
    entry = state.setdefault(sku, {})
    # A small incrementing counter (0, 1, 2, ...) starting fresh in this
    # tool's own state file can collide with a seed already baked into the
    # image on disk — e.g. a SKU regenerated via the CLI script's
    # --seed-offset 1 has offset 1 already "used up", but this file has no
    # record of that, so the first click here would also compute offset 1
    # and reproduce the exact same (already-rejected) image. A random
    # offset can't collide with generation history this tool doesn't know
    # about, so every click is guaranteed to actually be a new attempt.
    entry["seed_offset"] = random.randint(1, 1_000_000)
    entry["status"] = "pending"
    save_state(state)

    try:
        resp = requests.post(
            f"{REGEN_SERVER_URL}/regenerate/{sku}",
            params={"seed_offset": entry["seed_offset"]},
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach pod regen server: {exc}") from exc

    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error") or "Regeneration failed on the pod")

    category = product["category"]
    cat_folder = CATEGORY_FOLDERS.get(category, category)
    remote_dir = f"/workspace/repo/app/images/products/{cat_folder}/{sku}/"
    local_dir = PRODUCTS_ROOT / cat_folder / sku
    local_dir.mkdir(parents=True, exist_ok=True)
    rsync = subprocess.run(
        [
            "rsync", "-az", "-e", f"ssh -p {REVIEW_POD_PORT} -i {REVIEW_POD_KEY} -o StrictHostKeyChecking=accept-new",
            f"root@{REVIEW_POD_HOST}:{remote_dir}", str(local_dir) + "/",
        ],
        capture_output=True, text=True,
    )
    if rsync.returncode != 0:
        raise HTTPException(status_code=502, detail=f"rsync failed: {rsync.stderr}")

    accepted_count = sum(1 for v in state.values() if v.get("status") == "accepted")
    return product_payload(product, accepted_count)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8092)
