"""Populate the per-product/per-color image tree under app/images/products/.

There are no real per-product or per-color photos yet — this links the
existing category (and, where available, category+color) source photos
down into every product's own folder, so the full three-layer structure
`ImageService` expects actually exists on disk and the fallback logic can
be exercised end-to-end. Real photos can replace individual files later
with no code changes (just delete the symlink and drop a real file in its
place), since resolution always prefers the most specific file that exists.

Symlinks rather than copies — a few thousand real duplicate copies of ~25
source photos was multiple GB on disk for zero benefit; a symlink serves
identically (a static file server just follows it) at near-zero cost.

Usage:
    python -m catalog.generate_image_catalog
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.image_service import PRODUCTS_ROOT, ImageService, slugify  # noqa: E402
from config import CATALOG_PATH  # noqa: E402

import json  # noqa: E402


def main() -> None:
    products = json.loads(CATALOG_PATH.read_text())
    svc = ImageService()

    written = 0
    for p in products:
        cat_folder = svc.category_folder(p["category"])
        cat_dir = PRODUCTS_ROOT / cat_folder
        category_default = cat_dir / "default.png"
        if not category_default.exists():
            print(f"  ! no category default for {p['category']!r} (sku {p['sku']}), skipping")
            continue

        product_dir = cat_dir / p["sku"]
        product_dir.mkdir(parents=True, exist_ok=True)

        product_default = product_dir / "default.png"
        if not product_default.exists():
            os.symlink(os.path.relpath(category_default, product_dir), product_default)
            written += 1

        for color in p["colors"]:
            color_path = product_dir / f"{slugify(color)}.png"
            if color_path.exists():
                continue
            category_color_override = cat_dir / f"{slugify(color)}.png"
            source = category_color_override if category_color_override.exists() else category_default
            os.symlink(os.path.relpath(source, product_dir), color_path)
            written += 1

    print(f"Wrote {written} image files under {PRODUCTS_ROOT}")


if __name__ == "__main__":
    main()
