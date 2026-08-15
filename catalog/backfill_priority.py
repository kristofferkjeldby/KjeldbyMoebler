"""One-off backfill: add the `priority` field to the existing catalog.

`generate_catalog.py` now assigns `priority` for any newly-generated catalog,
but the catalog already on disk predates that field. Re-running the full
generator would reshuffle every product (new SKUs, new random attributes),
invalidating the test-question set and every eval baseline collected against
the current catalog — so this patches `priority` into the existing file
in place instead, using the exact same deterministic seeding as
`generate_catalog.py.apply_local_fields()` so a full regeneration later
would reproduce identical values for the same SKUs.

Usage:
    python -m catalog.backfill_priority
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CATALOG_PATH  # noqa: E402


def main() -> None:
    products = json.loads(CATALOG_PATH.read_text())

    already_has_priority = sum(1 for p in products if "priority" in p)
    if already_has_priority == len(products):
        print(f"All {len(products)} products already have a priority — nothing to do.")
        return

    for product in products:
        product["priority"] = random.Random(product["sku"]).randint(0, 100)

    CATALOG_PATH.write_text(json.dumps(products, ensure_ascii=False, indent=2))
    print(f"Backfilled priority for {len(products)} products in {CATALOG_PATH}")


if __name__ == "__main__":
    main()
