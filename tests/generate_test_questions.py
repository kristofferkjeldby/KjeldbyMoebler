"""Generate a 500-question test suite using Claude Sonnet.

Each test question is tied to specific ground-truth SKUs and has a
Claude-generated reference answer, so the judge (tests/judge.py) can score the
fine-tuned model's answer against known-correct facts rather than guessing.

Distribution mirrors the training-data query types so the eval measures the
same skills that were trained: single-product factual, multi-product
comparison, enumeration ("what chairs come in yellow"), store+quantity
("6 dining tables at Odense"), dimension-fit ("60cm deep kitchen unit"),
series-matching ("what goes with this table"), and unanswerable/edge-case.

Usage:
    python -m tests.generate_test_questions [--num-questions 500]

Writes tests/data/test_questions.jsonl:
    {"id": ..., "type": ..., "relevant_skus": [...], "question": ..., "reference_answer": ...}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.product_format import ROOM_LABELS, products_to_context  # noqa: E402
from catalog.structured_client import StructuredClient, build_client, with_retries  # noqa: E402
from config import (  # noqa: E402
    CATALOG_PATH,
    CLAUDE_MODEL,
    DISCOVERY_VIBES,
    SHOWN_MAX_RESULTS,
    GENERATION_MAX_WORKERS,
    NUM_TEST_QUESTIONS,
    STORES,
    TEST_QUESTIONS_PATH,
)
from rag.retriever import ProductRetriever  # noqa: E402

MAX_WORKERS = GENERATION_MAX_WORKERS
DIMENSION_FIELDS = ["depth_cm", "width_cm", "height_cm"]
DIMENSION_LABELS = {"depth_cm": "dyb", "width_cm": "bred", "height_cm": "høj"}

# Appended to every prompt: the catalog text is already Danish, but the
# model's own output language isn't otherwise constrained.
_DA_INSTRUCTION = " Skriv spørgsmålet og svaret på DANSK."

# Randomly assigned per call for tonal variety across the eval set — mirrors
# training/generate_training_data.py's approach, kept as a separate list so
# eval phrasing patterns don't exactly duplicate the training set's.
_STYLE_HINTS = [
    "kort og afslappet, som en sms til en ven",
    "formel og høflig, med 'De'-tiltale",
    "kortfattet og lidt utålmodig",
    "indirekte, som et hint frem for et direkte spørgsmål",
    "sammenlignende, som om kunden også overvejer at købe et andet sted",
    "tøvende og usikker, som om kunden stadig overvejer",
    "meget specifik og teknisk, med præcise tal og mål",
    "som en forælder der indretter et børneværelse",
    "som en der indretter en ny lejlighed på et stramt budget",
    "kort og direkte, uden høflighedsfraser",
    "entusiastisk og begejstret",
    "skrevet i telegramstil med få ord",
]


def _style_instruction() -> str:
    return f" Skriv spørgsmålet i denne tone: {random.choice(_STYLE_HINTS)}."

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {
            "type": "string",
            "description": "The correct answer, grounded strictly in the provided product data",
        },
    },
    "required": ["question", "reference_answer"],
    "additionalProperties": False,
}


# Every generated question is evaluated as a single, stateless turn — no
# prior conversation exists. A bare demonstrative ("this", "these", "denne",
# "disse") with no named product is unanswerable in that setting, even
# though it reads naturally as a human follow-up. Applied to every question
# type since any of them could slip into this pattern, not just single/multi.
_SELF_CONTAINED_INSTRUCTION = (
    " The question will be evaluated with no prior conversation context, so it must be "
    "fully self-contained — identify the product(s) by name or by clearly distinguishing "
    "detail, never with a bare pronoun/demonstrative like 'this', 'that', 'these', 'denne', "
    "'disse', 'den', 'det' with nothing named for it to refer to."
)


def _call(client: StructuredClient, prompt: str) -> dict:
    full_prompt = prompt + _SELF_CONTAINED_INSTRUCTION + _style_instruction() + _DA_INSTRUCTION
    return client.generate(full_prompt, QUESTION_SCHEMA, max_tokens=3000, temperature=0.7)


def make_single(client: StructuredClient, product: dict) -> dict:
    context = products_to_context([product])
    prompt = (
        f"Product data:\n\n{context}\n\n"
        f"Write ONE realistic customer question about this specific product (about price, "
        f"dimensions, colors, material, stock, warranty, or fit for a use case), naming the "
        f"product by its name '{product['name']}' (or its SKU {product['sku']}) since the "
        f"question must stand alone, and its correct reference answer, using only the facts given."
    )
    result = _call(client, prompt)
    return {"type": "single", "relevant_skus": [product["sku"]], **result}


def make_multi(client: StructuredClient, products: list[dict]) -> dict:
    context = products_to_context(products)
    names = ", ".join(f"'{p['name']}'" for p in products)
    prompt = (
        f"Product data for {len(products)} products:\n\n{context}\n\n"
        f"Write ONE realistic customer question that requires comparing these products "
        f"(price, size, stock, rating, etc.), naming each product by its name ({names}) since "
        f"the question must stand alone, and its correct reference answer, using only "
        f"the facts given."
    )
    result = _call(client, prompt)
    return {"type": "multi", "relevant_skus": [p["sku"] for p in products], **result}


def make_enumeration(client: StructuredClient, category: str | None, color_word: str | None, matches: list[dict]) -> dict:
    filter_desc = " and ".join(
        part for part in [
            f"category '{category.replace('_', ' ')}'" if category else None,
            f"color '{color_word}'" if color_word else None,
        ] if part
    )
    context = products_to_context(matches)
    if matches:
        prompt = (
            f"Here is data for ALL {len(matches)} products matching {filter_desc}:\n\n{context}\n\n"
            f"Write ONE natural customer question asking to see/list products matching "
            f"{filter_desc}, and a correct reference answer naming every one of the "
            f"{len(matches)} products above — no more, no fewer."
        )
    else:
        prompt = (
            f"NO products match {filter_desc}. Write ONE natural customer question asking to "
            f"see/list products matching {filter_desc}, and the correct reference answer: a "
            f"brief, honest statement that none are available, with no invented products."
        )
    result = _call(client, prompt)
    return {"type": "enumeration", "relevant_skus": [p["sku"] for p in matches], **result}


def make_store_stock(client: StructuredClient, category: str, store: str, quantity: int, matches: list[dict]) -> dict:
    category_phrase = category.replace("_", " ")
    context = products_to_context(matches)
    if matches:
        prompt = (
            f"Here is data for {category_phrase} products with at least {quantity} units in "
            f"stock at the {store} store:\n\n{context}\n\n"
            f"Write ONE natural customer question asking whether {quantity} {category_phrase}s "
            f"are available at {store}, and a correct reference answer naming every qualifying "
            f"product above with its {store} stock count."
        )
    else:
        prompt = (
            f"NO {category_phrase} product has {quantity}+ units in stock at the {store} store. "
            f"Write ONE natural customer question asking whether {quantity} {category_phrase}s "
            f"are available at {store}, and the correct reference answer: a brief, honest "
            f"statement that we don't have that many available there right now."
        )
    result = _call(client, prompt)
    return {"type": "store_stock", "relevant_skus": [p["sku"] for p in matches], **result}


def make_dimension(client: StructuredClient, category: str, field: str, target: float, matches: list[dict]) -> dict:
    category_phrase = category.replace("_", " ")
    label = DIMENSION_LABELS[field]
    context = products_to_context(matches)
    if matches:
        prompt = (
            f"Here is data for {category_phrase} products about {target:.0f}cm {label}:\n\n{context}\n\n"
            f"Write ONE natural customer question asking for a {category_phrase} around "
            f"{target:.0f}cm {label}, and a correct reference answer naming every matching "
            f"product above with its exact {label} measurement."
        )
    else:
        prompt = (
            f"NO {category_phrase} product is close to {target:.0f}cm {label}. Write ONE "
            f"natural customer question asking for this, and the correct reference answer: a "
            f"brief, honest statement that nothing that size is currently available."
        )
    result = _call(client, prompt)
    return {"type": "dimension", "relevant_skus": [p["sku"] for p in matches], **result}


def make_series(client: StructuredClient, series_products: list[dict]) -> dict:
    anchor = series_products[0]
    others = series_products[1:]
    context = products_to_context(series_products)
    prompt = (
        f"Here is data for a matching furniture series, '{anchor['series_name']}', including "
        f"the '{anchor['name']}' and {len(others)} other matching product(s):\n\n{context}\n\n"
        f"Write ONE natural customer question asking what matches or goes well with the "
        f"'{anchor['name']}', and a correct reference answer naming every other product in "
        f"the series above. Do not include the '{anchor['name']}' itself in the answer."
    )
    result = _call(client, prompt)
    return {"type": "series", "relevant_skus": [p["sku"] for p in series_products], **result}


def make_discovery(client: StructuredClient, room_label: str, vibe: str, products: list[dict]) -> dict:
    context = products_to_context(products)
    prompt = (
        f"Here is data for {len(products)} furniture products suited for a '{room_label}' "
        f"room:\n\n{context}\n\n"
        f"Write ONE open-ended, exploratory customer question from someone furnishing their "
        f"{room_label} who wants a {vibe} feel. They do NOT name a specific product or category "
        f"— they're browsing for ideas, not looking up something they already know exists. "
        f"Write a correct reference answer that recommends 2-4 of the products above that best "
        f"fit the request, briefly explaining why each fits, grounded ONLY in the facts given."
    )
    result = _call(client, prompt)
    return {"type": "discovery", "relevant_skus": [p["sku"] for p in products], **result}


def make_unanswerable(client: StructuredClient, products: list[dict]) -> dict:
    context = products_to_context(products)
    prompt = (
        f"Product data:\n\n{context}\n\n"
        f"Write ONE customer question that CANNOT be answered from this data (an attribute "
        f"not listed, like USB charging or waterproofing, or a totally different product "
        f"category not shown) and the correct reference answer: a brief, honest statement "
        f"that the information isn't available, with no invented facts."
    )
    result = _call(client, prompt)
    return {"type": "unanswerable", "relevant_skus": [p["sku"] for p in products], **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=NUM_TEST_QUESTIONS)
    parser.add_argument("--backend", choices=["claude", "local"], default="local")
    parser.add_argument("--local-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL for --backend local")
    parser.add_argument("--local-model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name for --backend local")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        raise SystemExit(f"No catalog found at {CATALOG_PATH}. Run catalog/generate_catalog.py first.")
    products = json.loads(CATALOG_PATH.read_text())
    by_category: dict[str, list[dict]] = {}
    by_room: dict[str, list[dict]] = {}
    for p in products:
        by_category.setdefault(p["category"], []).append(p)
        by_room.setdefault(p["room"], []).append(p)

    model = CLAUDE_MODEL if args.backend == "claude" else args.local_model
    client = build_client(args.backend, model, args.local_base_url)
    print(f"Using backend={args.backend!r} model={model!r}")
    retriever = ProductRetriever()
    categories = retriever.all_categories()
    color_words = retriever.all_color_words()
    series_ids = {p["series_id"] for p in products if p.get("series_id")}
    series_groups = [g for sid in series_ids if len(g := retriever.get_series(sid)) >= 2]

    n = args.num_questions
    counts = {
        "single": round(n * 0.30),
        "multi": round(n * 0.13),
        "enumeration": round(n * 0.13),
        "store_stock": round(n * 0.13),
        "dimension": round(n * 0.09),
        "series": min(round(n * 0.05), len(series_groups) * 3),
        "discovery": round(n * 0.12),
    }
    counts["unanswerable"] = n - sum(counts.values())

    jobs: list[tuple[str, tuple]] = []
    jobs += [("single", (random.choice(products),)) for _ in range(counts["single"])]
    jobs += [
        ("multi", (random.sample(random.choice(list(by_category.values())), k=2),))
        for _ in range(counts["multi"])
    ]
    for _ in range(counts["enumeration"]):
        mode = random.choice(["category", "color", "category_color"])
        category = random.choice(categories) if mode in ("category", "category_color") else None
        color_word = random.choice(color_words) if mode in ("color", "category_color") else None
        matches = retriever.filter_products(category=category, color_word=color_word)[:SHOWN_MAX_RESULTS]
        jobs.append(("enumeration", (category, color_word, matches)))
    for _ in range(counts["store_stock"]):
        category = random.choice(categories)
        store = random.choice(STORES)
        quantity = random.choice([1, 2, 2, 3, 4, 6, 8])
        matches = retriever.filter_products(category=category, store=store, min_quantity=quantity)[:SHOWN_MAX_RESULTS]
        jobs.append(("store_stock", (category, store, quantity, matches)))
    for _ in range(counts["dimension"]):
        category = random.choice(categories)
        field = random.choice(DIMENSION_FIELDS)
        target = round(random.uniform(30, 250))
        matches = retriever.filter_products(category=category, dimension=(field, target, 5))[:SHOWN_MAX_RESULTS]
        jobs.append(("dimension", (category, field, target, matches)))
    for _ in range(counts["series"]):
        jobs.append(("series", (random.choice(series_groups),)))
    rooms_with_products = [room for room, prods in by_room.items() if len(prods) >= 2]
    for _ in range(counts["discovery"]):
        room = random.choice(rooms_with_products)
        room_products = by_room[room]
        sample = random.sample(room_products, k=min(6, len(room_products)))
        vibe = random.choice(DISCOVERY_VIBES)
        jobs.append(("discovery", (ROOM_LABELS.get(room, room), vibe, sample)))
    jobs += [("unanswerable", (random.sample(products, k=random.choice([1, 2])),)) for _ in range(counts["unanswerable"])]
    random.shuffle(jobs)

    print(f"Generating {len(jobs)} test questions: {counts}")

    def run_job(kind: str, args_: tuple) -> dict:
        if kind == "single":
            return make_single(client, *args_)
        if kind == "multi":
            return make_multi(client, *args_)
        if kind == "enumeration":
            return make_enumeration(client, *args_)
        if kind == "store_stock":
            return make_store_stock(client, *args_)
        if kind == "dimension":
            return make_dimension(client, *args_)
        if kind == "series":
            return make_series(client, *args_)
        if kind == "discovery":
            return make_discovery(client, *args_)
        return make_unanswerable(client, *args_)

    questions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(with_retries, run_job, kind, a): kind for kind, a in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                questions.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! job failed: {exc}")
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} done")

    for i, q in enumerate(questions, start=1):
        q["id"] = i

    with TEST_QUESTIONS_PATH.open("w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nWrote {len(questions)} test questions to {TEST_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
