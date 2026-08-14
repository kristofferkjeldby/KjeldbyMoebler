"""Product image resolution service.

The image catalog lives under `app/images/products/` in three layers,
from most to least specific:

    products/<category>/<sku>/<color-slug>.png   product + color
    products/<category>/<sku>/default.png         product, no color
    products/<category>/default.png                category only

Most of the tree is currently populated with *duplicates* of a handful of
real source photos (see generate_image_catalog.py) — there's no real photo
per product/color combination yet, just the scaffolding so the lookup
logic, manifest, and frontend wiring are all ready for real photos to be
dropped in later without any code changes.

Usage:
    from catalog.image_service import ImageService
    svc = ImageService()
    svc.resolve(sku="FRN-2401", category="sectional", color="Mørkegrå")
"""
from __future__ import annotations

import re
from pathlib import Path

IMAGES_ROOT = Path(__file__).resolve().parent.parent / "app" / "images"
PRODUCTS_ROOT = IMAGES_ROOT / "products"

# Internal catalog category key -> image folder name. ASCII-only (these
# become file paths / URL segments) and matches the existing category
# photo filenames already in use elsewhere in the app.
CATEGORY_FOLDERS: dict[str, str] = {
    "sofa": "sofa",
    "armchair": "laenestol",
    "loveseat": "to-personers_sofa",
    "sectional": "hjoernesofa",
    "dining_table": "spisebord",
    "coffee_table": "sofabord",
    "side_table": "sidebord",
    "console_table": "konsolbord",
    "bed_frame": "sengeramme",
    "nightstand": "natbord",
    "dresser": "komode",
    "wardrobe": "garderobeskab",
    "bookshelf": "bogreol",
    "desk": "skrivebord",
    "office_chair": "kontorstol",
    "dining_chair": "spisebordstol",
    "bar_stool": "barstol",
    "tv_stand": "tv-bord",
    "outdoor_set": "havemoeblesaet",
    "rug": "taeppe",
    "lighting": "belysning",
    "kitchen_unit": "koekkenelement",
}


def slugify(text: str) -> str:
    """ASCII-transliterate and slugify for use as a filename ("Mørkegrå" ->
    "moerkegraa", "Sort med grå detaljer" -> "sort-med-graa-detaljer")."""
    text = text.lower().replace("æ", "ae").replace("ø", "oe").replace("å", "aa")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


class ImageService:
    def __init__(self, products_root: Path = PRODUCTS_ROOT):
        self.root = products_root

    def category_folder(self, category: str) -> str:
        return CATEGORY_FOLDERS.get(category, category)

    def resolve(self, sku: str, category: str, color: str | None = None) -> Path | None:
        """Most-specific-first fallback: product+color -> product default ->
        category default. Returns None only if even the category default is
        missing (shouldn't happen once the catalog is generated)."""
        cat_folder = self.category_folder(category)
        product_dir = self.root / cat_folder / sku

        if color:
            color_path = product_dir / f"{slugify(color)}.png"
            if color_path.exists():
                return color_path

        product_default = product_dir / "default.png"
        if product_default.exists():
            return product_default

        category_default = self.root / cat_folder / "default.png"
        if category_default.exists():
            return category_default

        return None

    def resolve_url(self, sku: str, category: str, color: str | None = None) -> str | None:
        """Same as `resolve`, but as a path relative to app/images/ (what
        the frontend actually needs, since it serves from that directory)."""
        path = self.resolve(sku, category, color)
        if path is None:
            return None
        return str(path.relative_to(IMAGES_ROOT))
