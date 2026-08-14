"""Generate the SFT dataset: grounded Q&A pairs conditioned on injected product context.

This is what actually teaches the model to answer from RAG-injected context
instead of its own (untrustworthy) parametric knowledge about furniture.
Kinds of examples produced, all using the exact same SYSTEM_PROMPT_TEMPLATE
and context-rendering the app/eval use at inference time:

  1. Single-product Q&A   — one product injected, question answerable from it.
  2. Multi-product Q&A    — 2-3 products injected (same category), comparison
                             or "which is cheaper/bigger/in stock" questions.
  3. Enumeration Q&A      — "what chairs come in yellow?" — list every match, or none.
  4. Store + quantity Q&A — "I need 6 dining tables available at Odense" —
                             checks a specific store's stock against a quantity.
  5. Dimension-fit Q&A    — "I need a 60cm deep kitchen unit" — numeric tolerance match.
  6. Series-matching Q&A  — "what goes with this dining table?" — names the other
                             products sharing the same series_id.
  7. Unanswerable Q&A     — products injected, but the question asks about an
                             attribute or product NOT present in the context,
                             so the target answer is a plain, honest refusal.
                             Without these, fine-tuning teaches the model to
                             always answer confidently, which defeats the RAG
                             grounding.

Usage:
    python -m training.generate_training_data

Writes training/data/sft_dataset.jsonl — one JSON object per line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
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
    ENUMERATION_MAX_RESULTS,
    EXAMPLES_PER_PRODUCT,
    GENERATION_MAX_WORKERS,
    NUM_DIMENSION_EXAMPLES,
    NUM_DISCOVERY_EXAMPLES,
    NUM_ENUMERATION_EXAMPLES,
    NUM_MULTI_PRODUCT_EXAMPLES,
    NUM_SERIES_EXAMPLES,
    NUM_SINGLE_PRODUCT_SAMPLE,
    NUM_STORE_STOCK_EXAMPLES,
    NUM_UNANSWERABLE_EXAMPLES,
    STORES,
    SYSTEM_PROMPT_TEMPLATE,
    TRAINING_DATA_PATH,
)
from rag.retriever import ProductRetriever  # noqa: E402

MAX_WORKERS = GENERATION_MAX_WORKERS
DIMENSION_FIELDS = ["depth_cm", "width_cm", "height_cm"]
DIMENSION_LABELS = {"depth_cm": "dyb", "width_cm": "bred", "height_cm": "høj"}

# Appended to every prompt: the catalog text is already Danish, but the
# model's own output language isn't otherwise constrained.
_DA_INSTRUCTION = " Skriv spørgsmålet og svaret på DANSK."

# Randomly assigned per call to force lexical/tonal variety across the
# dataset — without this, "vary phrasing" alone tends to converge on very
# similar sentence shapes across thousands of generation calls, and the
# model overfits to that one register instead of learning to handle real
# customer variety.
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
    "som en der allerede har læst om produktet online og vil bekræfte detaljer",
    "kort og direkte, uden høflighedsfraser",
    "entusiastisk og begejstret",
    "skrevet i telegramstil med få ord",
    "som en professionel indkøber, der handler til et kontor eller firma",
]


def _style_instruction(n: int) -> str:
    hints = random.sample(_STYLE_HINTS, k=min(n, len(_STYLE_HINTS)))
    styled = "; ".join(f"{i + 1}) {h}" for i, h in enumerate(hints))
    return (
        f" Varier tonen tydeligt mellem spørgsmålene — brug disse stilarter, "
        f"én pr. spørgsmål (gentag om nødvendigt hvis der er flere spørgsmål "
        f"end stilarter): {styled}."
    )

QA_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


def _call(client: StructuredClient, prompt: str, n: int) -> list[dict]:
    full_prompt = prompt + _style_instruction(n) + _DA_INSTRUCTION
    data = client.generate(full_prompt, QA_BATCH_SCHEMA, max_tokens=6000, temperature=0.7)
    return data["pairs"][:n]


def single_product_examples(client: StructuredClient, product: dict) -> list[dict]:
    context = products_to_context([product])
    prompt = (
        f"Here is a furniture product's full data:\n\n{context}\n\n"
        f"Generate {EXAMPLES_PER_PRODUCT} diverse customer questions about this exact "
        f"product, each with a correct, natural-sounding answer that uses ONLY the facts "
        f"given above. Cover a mix of: price, dimensions, available colors, material, "
        f"stock/availability, delivery lead time, warranty, and whether it suits a "
        f"stated use case (e.g. 'small apartment', 'kids room'). Vary phrasing and "
        f"question style (short factual, conversational, indirect)."
    )
    pairs = _call(client, prompt, EXAMPLES_PER_PRODUCT)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def multi_product_example(client: StructuredClient, products: list[dict]) -> list[dict]:
    context = products_to_context(products)
    prompt = (
        f"Here is data for {len(products)} furniture products (same category):\n\n{context}\n\n"
        f"Generate 2 customer questions that require comparing or reasoning across these "
        f"products (e.g. 'which is cheaper', 'which has more storage', 'which is in stock "
        f"and ships soonest'), each with a correct answer grounded ONLY in the facts above."
    )
    pairs = _call(client, prompt, 2)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def enumeration_example(
    client: StructuredClient, category: str | None, color_word: str | None, matches: list[dict]
) -> list[dict]:
    """'What chairs are available in yellow?' style: list every match, or say there are none.

    Semantic top-k retrieval truncates these silently, so the model needs
    explicit training on the "here are all N of them" / "we don't have any"
    shape of answer, not just single- or few-product lookups.
    """
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
            f"Write ONE natural, casually-phrased customer question asking to see or list the "
            f"available products matching {filter_desc} (not a question about one specific item), "
            f"and a correct answer that names every one of the {len(matches)} products above "
            f"(with price) — no more, no fewer. Do not invent any product not listed above."
        )
    else:
        prompt = (
            f"A customer asks to see products matching {filter_desc}, but the store's data has "
            f"NO products matching that filter. Write ONE natural customer question asking to see "
            f"or list products matching {filter_desc}, and the correct answer: a brief, honest "
            f"statement that none are currently available, without inventing any products."
        )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def store_stock_example(
    client: StructuredClient, category: str, store: str, quantity: int, matches: list[dict]
) -> list[dict]:
    """'I need 6 dining tables available at the Odense store' style."""
    category_phrase = category.replace("_", " ")
    context = products_to_context(matches)
    if matches:
        prompt = (
            f"Here is data for the {category_phrase} products that have at least {quantity} "
            f"units in stock at the {store} store:\n\n{context}\n\n"
            f"Write ONE natural customer question asking whether {quantity} {category_phrase}s "
            f"are available at the {store} store, and a correct answer that names every "
            f"qualifying product above with its {store} stock count and price. Do not invent "
            f"any product not listed, and do not mention other stores' stock."
        )
    else:
        prompt = (
            f"A customer asks whether {quantity} {category_phrase}s are available at the "
            f"{store} store, but NO {category_phrase} product currently has that much stock "
            f"at {store}. Write ONE natural customer question asking this, and the correct "
            f"answer: a brief, honest statement that we don't have that many available at "
            f"{store} right now, without inventing any product or stock numbers."
        )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def dimension_example(
    client: StructuredClient, category: str, field: str, target: float, matches: list[dict]
) -> list[dict]:
    """'I have a kitchen, and need a 60cm deep element' style — numeric-tolerance fit."""
    category_phrase = category.replace("_", " ")
    label = DIMENSION_LABELS[field]
    context = products_to_context(matches)
    if matches:
        prompt = (
            f"Here is data for {category_phrase} products that are about {target:.0f}cm {label} "
            f"(within a few cm):\n\n{context}\n\n"
            f"Write ONE natural, casually-phrased customer question asking for a {category_phrase} "
            f"that's around {target:.0f}cm {label} (e.g. mentioning a room it needs to fit), and a "
            f"correct answer naming every matching product above with its exact {label} "
            f"measurement. Do not invent any product not listed."
        )
    else:
        prompt = (
            f"A customer wants a {category_phrase} that's about {target:.0f}cm {label}, but NO "
            f"{category_phrase} product is close to that measurement. Write ONE natural customer "
            f"question asking for this, and the correct answer: a brief, honest statement that "
            f"nothing that size is currently available, without inventing any product."
        )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def discovery_example(client: StructuredClient, room_label: str, vibe: str, products: list[dict]) -> list[dict]:
    """'I'm doing up my living room in a cozy style, any ideas?' — the customer
    doesn't know what product they want yet, unlike every other example type
    here. Without this, the model only ever learns to look up or filter on a
    need the customer has already made concrete, and has no training signal
    for open-ended browsing/recommendation questions that lean on the
    semantic-fallback retrieval tier instead of an exact filter.
    """
    context = products_to_context(products)
    prompt = (
        f"Here is data for {len(products)} furniture products suited for a '{room_label}' "
        f"room:\n\n{context}\n\n"
        f"Write ONE open-ended, exploratory customer question from someone furnishing their "
        f"{room_label} who wants a {vibe} feel. They do NOT name a specific product or category "
        f"— they're browsing for ideas, not looking up something they already know exists "
        f"(e.g. 'I'm redoing my living room to feel cozier, what would you suggest?'). Write a "
        f"helpful, consultative reference answer that recommends 2-4 of the products above that "
        f"best fit the request, briefly explaining why each one fits, grounded ONLY in the facts "
        f"given above. Do not recommend or invent anything not listed."
    )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def series_example(client: StructuredClient, series_products: list[dict]) -> list[dict]:
    """'What goes with this dining table?' — names the other products in the same series."""
    anchor = series_products[0]
    others = series_products[1:]
    context = products_to_context(series_products)
    prompt = (
        f"Here is data for a matching furniture series, '{anchor['series_name']}', including "
        f"the '{anchor['name']}' ({anchor['category'].replace('_', ' ')}) and {len(others)} "
        f"other matching product(s):\n\n{context}\n\n"
        f"Write ONE natural customer question asking what matches or goes well with the "
        f"'{anchor['name']}', and a correct answer naming every other product in the series "
        f"above (with category and price) as matching pieces. Do not invent any product not "
        f"listed, and do not include the '{anchor['name']}' itself in the list of matches."
    )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def unanswerable_example(client: StructuredClient, products: list[dict]) -> list[dict]:
    context = products_to_context(products)
    prompt = (
        f"Here is data for these furniture products:\n\n{context}\n\n"
        f"Generate 1 customer question that CANNOT be answered from this data — either it "
        f"asks about a specific attribute not listed (e.g. 'does it have USB charging', "
        f"'is it waterproof', 'what's the return policy', 'do you have it in a different "
        f"category like a bed frame') or it asks about a different, unrelated product not "
        f"shown above. Provide the correct answer: a brief, polite, honest statement that "
        f"this information isn't available, without inventing anything. Do not apologize "
        f"excessively or repeat the disclaimer more than once."
    )
    pairs = _call(client, prompt, 1)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return [
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]}
        for p in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
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
    examples: list[dict] = []

    single_sample = random.sample(products, k=min(NUM_SINGLE_PRODUCT_SAMPLE, len(products)))

    jobs = []
    jobs += [("single", (p,)) for p in single_sample]
    jobs += [
        ("multi", (random.sample(cat_products, k=min(3, len(cat_products))),))
        for cat_products in by_category.values()
        for _ in range(max(1, NUM_MULTI_PRODUCT_EXAMPLES // max(1, len(by_category))))
    ]
    jobs += [
        ("unanswerable", (random.sample(products, k=min(2, len(products))),))
        for _ in range(NUM_UNANSWERABLE_EXAMPLES)
    ]

    # Enumeration jobs: category-only, color-only, and category+color combos,
    # deliberately including some combos with zero matches so the model also
    # learns the "we don't have any" case rather than only ever enumerating.
    for _ in range(NUM_ENUMERATION_EXAMPLES):
        mode = random.choice(["category", "color", "category_color"])
        category = random.choice(categories) if mode in ("category", "category_color") else None
        color_word = random.choice(color_words) if mode in ("color", "category_color") else None
        matches = retriever.filter_products(category=category, color_word=color_word)[:ENUMERATION_MAX_RESULTS]
        jobs.append(("enumeration", (category, color_word, matches)))

    # Store + quantity jobs — biased toward quantities > 1, mirroring "I need
    # 6 dining tables" style questions; deliberately includes combos with
    # insufficient stock so the model learns the honest "not enough" answer.
    for _ in range(NUM_STORE_STOCK_EXAMPLES):
        category = random.choice(categories)
        store = random.choice(STORES)
        quantity = random.choice([1, 2, 2, 3, 4, 6, 8])
        matches = retriever.filter_products(category=category, store=store, min_quantity=quantity)[:ENUMERATION_MAX_RESULTS]
        jobs.append(("store_stock", (category, store, quantity, matches)))

    # Dimension-fit jobs — target values sampled broadly enough to produce a
    # healthy mix of hits and "nothing that size" misses.
    for _ in range(NUM_DIMENSION_EXAMPLES):
        category = random.choice(categories)
        field = random.choice(DIMENSION_FIELDS)
        target = round(random.uniform(30, 250))
        matches = retriever.filter_products(category=category, dimension=(field, target, 5))[:ENUMERATION_MAX_RESULTS]
        jobs.append(("dimension", (category, field, target, matches)))

    # Series-matching jobs — only series with >=2 members are useful here.
    series_ids = {p["series_id"] for p in products if p.get("series_id")}
    series_groups = [g for sid in series_ids if len(g := retriever.get_series(sid)) >= 2]
    for _ in range(min(NUM_SERIES_EXAMPLES, len(series_groups) * 3)):
        jobs.append(("series", (random.choice(series_groups),)))

    # Discovery jobs — the customer doesn't know what product they want yet,
    # just a room + vibe. Grounds the answer in a handful of products
    # actually tagged for that room, mixed across categories (a real
    # "browsing" candidate set, not a single-category filter).
    rooms_with_products = [room for room, prods in by_room.items() if len(prods) >= 2]
    for _ in range(NUM_DISCOVERY_EXAMPLES):
        room = random.choice(rooms_with_products)
        room_products = by_room[room]
        sample = random.sample(room_products, k=min(6, len(room_products)))
        vibe = random.choice(DISCOVERY_VIBES)
        jobs.append(("discovery", (ROOM_LABELS.get(room, room), vibe, sample)))

    random.shuffle(jobs)

    print(f"Running {len(jobs)} generation jobs with {MAX_WORKERS} workers...")

    def run_job(kind: str, args: tuple) -> list[dict]:
        if kind == "single":
            return single_product_examples(client, *args)
        if kind == "multi":
            return multi_product_example(client, *args)
        if kind == "enumeration":
            return enumeration_example(client, *args)
        if kind == "store_stock":
            return store_stock_example(client, *args)
        if kind == "dimension":
            return dimension_example(client, *args)
        if kind == "series":
            return series_example(client, *args)
        if kind == "discovery":
            return discovery_example(client, *args)
        return unanswerable_example(client, *args)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(with_retries, run_job, kind, args): kind for kind, args in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                examples.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! job failed: {exc}")
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} jobs done, {len(examples)} examples so far")

    random.shuffle(examples)
    with TRAINING_DATA_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} training examples to {TRAINING_DATA_PATH}")


if __name__ == "__main__":
    main()
