"""Generate app/data/image_manifest.json — a static sku -> image-path lookup
for the (static, no backend) frontend, built from the same ImageService
resolution logic used server-side, so both stay in sync automatically.

Usage:
    python -m catalog.generate_image_manifest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.image_service import ImageService, slugify  # noqa: E402
from config import CATALOG_PATH  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "image_manifest.json"


def main() -> None:
    products = json.loads(CATALOG_PATH.read_text())
    svc = ImageService()

    manifest = {}
    for p in products:
        default_path = svc.resolve(p["sku"], p["category"])
        default_url = svc.resolve_url(p["sku"], p["category"])
        default_real = default_path.resolve() if default_path else None

        colors = {}
        for color in p["colors"]:
            color_path = svc.resolve(p["sku"], p["category"], color)
            if color_path is None:
                continue
            # Only keep a manifest entry when a real, distinct file backs
            # this color — comparing the *resolved* (symlink-followed) file
            # rather than the per-color path string, since every color gets
            # its own path even when it's just a symlink to the same
            # underlying default photo.
            if color_path.resolve() != default_real:
                colors[slugify(color)] = str(color_path.relative_to(svc.root.parent))
        manifest[p["sku"]] = {"default": default_url, "colors": colors}

    OUTPUT_PATH.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    print(f"Wrote {len(manifest)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
