"""Generate a fake furniture product catalog using Claude Sonnet.

Usage:
    python -m catalog.generate_catalog [--num-products 200]

Two kinds of generation, both run in parallel via a thread pool:

  1. Series: a cohesive matching set (e.g. a dining table + its matching
     chair) — one Claude call per series, returning one item per category in
     the archetype, all sharing a style/material story and a series_id.
  2. Standalone: single-category batches, same as a plain catalog entry.

Everything "mechanical" (SKU, per-store stock/restock dates, discount
percent/price) is generated locally in Python rather than by Claude — it's
arbitrary random data, not something that benefits from the model's
judgment, and computing discount_price ourselves guarantees the arithmetic
is always consistent with normal_price and discount_percent.

Writes catalog/data/catalog.json — a list of product dicts (see
catalog/product_format.py for the exact shape).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 3


def _with_retries(fn, *args, **kwargs):
    """Retry transient failures (dropped connections, timeouts) a few times
    with linear backoff before giving up — a single flaky network blip
    shouldn't lose an entire batch's worth of generated products."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.structured_client import StructuredClient, build_client  # noqa: E402
from config import (  # noqa: E402
    ALL_ATTRIBUTE_FIELDS,
    CATALOG_PATH,
    CATEGORIES,
    CATEGORY_ATTRIBUTE_FIELDS,
    CLAUDE_MODEL,
    DISCOUNT_PERCENT_RANGE,
    DISCOUNT_PROBABILITY,
    GENERATION_MAX_WORKERS,
    NUM_PRODUCTS,
    NUM_SERIES,
    OUT_OF_STOCK_PROBABILITY,
    RESTOCK_DAYS_RANGE,
    SERIES_ARCHETYPES,
    STOCK_QUANTITY_RANGE,
    STORES,
)

BATCH_SIZE = 4
MAX_WORKERS = GENERATION_MAX_WORKERS

ATTRIBUTE_FIELD_TYPES = {
    "seat_height_cm": "number",
    "seat_depth_cm": "number",
    "weight_capacity_kg": "number",
    "seats_count": "integer",
    "extendable": "boolean",
    "bed_size": "string",
    "num_drawers": "integer",
    "num_shelves": "integer",
    "weather_resistant": "boolean",
    "bulb_type": "string",
    "lumens": "number",
    "wattage": "number",
    "dimmable": "boolean",
    "mount_type": "string",
    "unit_type": "string",
    "door_count": "integer",
}
assert set(ATTRIBUTE_FIELD_TYPES) == set(ALL_ATTRIBUTE_FIELDS), "attribute type map is out of sync with config.py"

ATTRIBUTES_SCHEMA = {
    "type": "object",
    "properties": {
        field: {"type": [ATTRIBUTE_FIELD_TYPES[field], "null"]} for field in ALL_ATTRIBUTE_FIELDS
    },
    "required": ALL_ATTRIBUTE_FIELDS,
    "additionalProperties": False,
}

BASE_ITEM_PROPERTIES = {
    "name": {"type": "string", "description": "Distinct, realistic product name"},
    "short_description": {"type": "string", "description": "1-2 sentence marketing description"},
    "normal_price": {"type": "number"},
    "colors": {"type": "array", "items": {"type": "string"}, "description": "2-4 realistic color/finish options"},
    "material": {"type": "string", "description": "e.g. 'Solid oak, linen upholstery'"},
    "dimensions": {
        "type": "object",
        "properties": {
            "width_cm": {"type": "number"},
            "depth_cm": {"type": "number"},
            "height_cm": {"type": "number"},
        },
        "required": ["width_cm", "depth_cm", "height_cm"],
        "additionalProperties": False,
    },
    "weight_kg": {"type": "number"},
    "rating": {"type": "number", "description": "1.0-5.0"},
    "review_count": {"type": "integer"},
    "warranty_years": {"type": "integer"},
    "assembly_required": {"type": "boolean"},
    "room": {
        "type": "string",
        "enum": ["living_room", "bedroom", "dining_room", "home_office", "outdoor", "entryway", "kids_room", "kitchen"],
    },
    "attributes": ATTRIBUTES_SCHEMA,
}
BASE_ITEM_REQUIRED = list(BASE_ITEM_PROPERTIES)


def attribute_guidance(categories: list[str]) -> str:
    lines = []
    for cat in categories:
        fields = CATEGORY_ATTRIBUTE_FIELDS.get(cat, [])
        if fields:
            lines.append(f"  - {cat}: fill {', '.join(fields)}; leave all other attribute fields null")
        else:
            lines.append(f"  - {cat}: leave all attribute fields null")
    return "\n".join(lines)


# --- Standalone batch generation (single category per call) ---

STANDALONE_BATCH_SCHEMA = {
    "type": "object",
    "properties": {"products": {"type": "array", "items": {
        "type": "object", "properties": BASE_ITEM_PROPERTIES, "required": BASE_ITEM_REQUIRED, "additionalProperties": False,
    }}},
    "required": ["products"],
    "additionalProperties": False,
}


def generate_standalone_batch(client: StructuredClient, category: str, count: int) -> list[dict]:
    prompt = (
        f"Generér {count} forskellige, realistiske møbelprodukter i kategorien "
        f"'{category}' til en dansk online møbelforretnings katalog. Skriv navn, "
        f"kortbeskrivelse, materiale og farver på DANSK. Varier prisniveau (billigt til "
        f"eksklusivt), stilarter (moderne, rustik, industriel, minimalistisk, klassisk) "
        f"og materialer. Priser angives i danske kroner (DKK), og skal sammen med mål og "
        f"vægt være internt konsistente med det beskrevne produkt.\n\n"
        f"Vejledning til attributfelter:\n{attribute_guidance([category])}"
    )
    data = client.generate(prompt, STANDALONE_BATCH_SCHEMA, max_tokens=8000)
    for product in data["products"]:
        product["category"] = category
        product["series_id"] = None
        product["series_name"] = None
    return data["products"]


# --- Series generation (multiple categories, one cohesive style, per call) ---

def series_schema(categories: list[str]) -> dict:
    item_properties = {**BASE_ITEM_PROPERTIES, "category": {"type": "string", "enum": categories}}
    return {
        "type": "object",
        "properties": {
            "series_name": {"type": "string", "description": "A cohesive product line name, e.g. 'Oslo Dining Collection'"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": BASE_ITEM_REQUIRED + ["category"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["series_name", "items"],
        "additionalProperties": False,
    }


def generate_series(client: StructuredClient, archetype: dict) -> dict:
    categories = archetype["categories"]
    prompt = (
        f"Design én sammenhængende møbelserie (et matchende sæt) til en dansk online "
        f"møbelforretning, bestående af præcis ét produkt for hver af disse kategorier: "
        f"{', '.join(categories)}. Alle produkter skal dele en konsistent stil, "
        f"materialefortælling og farvepalet, så de tydeligt hører sammen (fx samme "
        f"trætone og formsprog), men have realistiske mål/vægt/pris for deres egen "
        f"kategori. Skriv navn, kortbeskrivelse, materiale, farver og serienavn på DANSK. "
        f"Priser angives i danske kroner (DKK).\n\n"
        f"Vejledning til attributfelter:\n{attribute_guidance(categories)}"
    )
    return client.generate(prompt, series_schema(categories), max_tokens=4000)


# --- Local post-processing: SKU, availability, discount (no Claude call) ---

def generate_availability() -> dict:
    availability = {}
    for store in STORES:
        if random.random() < OUT_OF_STOCK_PROBABILITY:
            restock_in = random.randint(*RESTOCK_DAYS_RANGE)
            availability[store] = {
                "stock_quantity": 0,
                "restock_date": (date.today() + timedelta(days=restock_in)).isoformat(),
            }
        else:
            availability[store] = {
                "stock_quantity": random.randint(*STOCK_QUANTITY_RANGE),
                "restock_date": None,
            }
    return availability


def apply_local_fields(product: dict, index: int) -> dict:
    product["sku"] = f"FRN-{index:04d}"
    product["availability"] = generate_availability()
    if random.random() < DISCOUNT_PROBABILITY:
        pct = random.randint(*DISCOUNT_PERCENT_RANGE)
        product["discount_percent"] = pct
        product["discount_price"] = round(product["normal_price"] * (1 - pct / 100), 2)
    else:
        product["discount_percent"] = None
        product["discount_price"] = None
    # Business-set "which of these do we want to sell first" ranking (0-100,
    # higher = shown first when a query matches more than SHOWN_MAX_RESULTS
    # products) — never rendered into the model's context or the customer
    # UI. This is a placeholder: real values should come from actual
    # business input (margin, aged stock, etc.), not this random catalog.
    # Seeded on the SKU (not the shared `random` stream) so it's
    # reproducible independent of generation order/threading and stable if
    # this script is ever re-run with --resume.
    product["priority"] = random.Random(product["sku"]).randint(0, 100)
    return product


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-products", type=int, default=NUM_PRODUCTS)
    parser.add_argument("--num-series", type=int, default=None, help="Defaults to config.NUM_SERIES, or 0 when --resume is set")
    parser.add_argument(
        "--resume", action="store_true",
        help="Load the existing output file and top it up to --num-products instead of starting fresh "
             "(e.g. after a partial run lost to a dropped connection). Skips series generation by default.",
    )
    parser.add_argument("--backend", choices=["claude", "local"], default="claude")
    parser.add_argument("--local-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL for --backend local")
    parser.add_argument("--local-model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name for --backend local")
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output path (default: {CATALOG_PATH}). Use a different path for --backend local so the "
             "Claude-generated catalog used for training/eval is never overwritten.",
    )
    args = parser.parse_args()
    num_series = args.num_series if args.num_series is not None else (0 if args.resume else NUM_SERIES)
    output_path = args.output or CATALOG_PATH

    model = CLAUDE_MODEL if args.backend == "claude" else args.local_model
    client = build_client(args.backend, model, args.local_base_url)
    print(f"Using backend={args.backend!r} model={model!r} -> output={output_path}")

    existing_products: list[dict] = []
    if args.resume and output_path.exists():
        existing_products = json.loads(output_path.read_text())
        print(f"Resuming: loaded {len(existing_products)} existing products from {output_path}")

    new_products: list[dict] = []

    if num_series > 0:
        print(f"Generating {num_series} product series ({MAX_WORKERS} workers)...")
        archetypes = [random.choice(SERIES_ARCHETYPES) for _ in range(num_series)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_with_retries, generate_series, client, arch): arch for arch in archetypes}
            for future in as_completed(futures):
                try:
                    series = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! series failed after {RETRY_ATTEMPTS} attempts: {exc} — skipping")
                    continue
                series_id = f"series-{uuid.uuid4().hex[:8]}"
                for item in series["items"]:
                    item["series_id"] = series_id
                    item["series_name"] = series["series_name"]
                new_products.extend(series["items"])
                print(f"  + series '{series['series_name']}' ({len(series['items'])} items, {len(new_products)} new so far)")

    remaining = max(0, args.num_products - len(existing_products) - len(new_products))
    print(f"\nGenerating {remaining} standalone products across categories ({MAX_WORKERS} workers)...")
    jobs: list[tuple[str, int]] = []
    left, idx = remaining, 0
    while left > 0:
        category = CATEGORIES[idx % len(CATEGORIES)]
        idx += 1
        batch_count = min(BATCH_SIZE, left)
        jobs.append((category, batch_count))
        left -= batch_count

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_with_retries, generate_standalone_batch, client, cat, n): cat for cat, n in jobs}
        for future in as_completed(futures):
            category = futures[future]
            try:
                batch = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ! batch for '{category}' failed after {RETRY_ATTEMPTS} attempts: {exc} — skipping")
                continue
            new_products.extend(batch)
            print(f"  + {len(batch)} '{category}' products ({len(existing_products) + len(new_products)}/{args.num_products})")

    new_products = new_products[: max(0, args.num_products - len(existing_products))]
    for i, product in enumerate(new_products, start=len(existing_products) + 1):
        apply_local_fields(product, i)

    products = (existing_products + new_products)[: args.num_products]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(products, indent=2))
    print(f"\nWrote {len(products)} products to {output_path} ({len(new_products)} newly generated)")


if __name__ == "__main__":
    main()
