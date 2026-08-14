"""RAG retrieval tests for entry-level queries — the first thing a customer
asks, with no prior conversation (no focus carryover). Covers the
rule-based attribute-filter tier (`ProductRetriever.retrieve`): a bare
category, a category plus one property (price / color / availability), and
a category plus two properties combined.

These are deterministic and don't touch the LLM/pod — just the local
embedding model (for index loading) and catalog data — so they run in
seconds and should catch retrieval regressions long before an eval run
would (see tests/run_eval.py for the slower, model-in-the-loop suite).

Usage:
    pytest tests/test_retriever.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENUMERATION_MAX_RESULTS, STORES  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402

# category key -> a real Danish phrase a customer would actually type.
CATEGORY_QUERIES = {
    "sofa": "sofaer",
    "sectional": "hjørnesofaer",
    "armchair": "lænestole",
    "dining_table": "spiseborde",
    "bed_frame": "sengerammer",
    "bookshelf": "reoler",
}

# A color common enough to have real matches across most categories.
COMMON_COLOR_WORD = "sorte"


@pytest.fixture(scope="module")
def retriever() -> ProductRetriever:
    return ProductRetriever()


def _effective_price(p: dict) -> float:
    return p.get("discount_price") or p["normal_price"]


# --- category alone ---------------------------------------------------

@pytest.mark.parametrize("category, phrase", CATEGORY_QUERIES.items())
def test_category_only_returns_matching_products(retriever, category, phrase):
    results = retriever.retrieve(f"Hvilke {phrase} har I?", top_k=8, focus=[])

    assert results, f"expected at least one {category} for a bare category query"
    assert len(results) <= ENUMERATION_MAX_RESULTS
    assert all(p["category"] == category for p in results), (
        f"non-{category} product leaked into results: "
        f"{[p['sku'] for p in results if p['category'] != category]}"
    )


def test_category_only_caps_at_enumeration_max(retriever):
    # "sofa" alone is a large category (60+ products) — confirms the cap is
    # actually enforced, not just coincidentally under it for small ones.
    results = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[])
    assert len(results) == ENUMERATION_MAX_RESULTS


# --- category + one property -------------------------------------------

@pytest.mark.parametrize("category, phrase", CATEGORY_QUERIES.items())
def test_category_plus_price_threshold(retriever, category, phrase):
    results = retriever.retrieve(f"Har I {phrase} under 5000 kr?", top_k=8, focus=[])

    assert all(p["category"] == category for p in results)
    assert all(_effective_price(p) <= 5000 for p in results), (
        f"product over budget leaked in: "
        f"{[(p['sku'], _effective_price(p)) for p in results if _effective_price(p) > 5000]}"
    )


@pytest.mark.parametrize("category, phrase", CATEGORY_QUERIES.items())
def test_category_plus_color(retriever, category, phrase):
    query = f"Har I {COMMON_COLOR_WORD} {phrase}?"
    results = retriever.retrieve(query, top_k=8, focus=[])

    # The catalog isn't consistent about color casing/inflection ("Sort" vs
    # "Sorte" vs "Sorte metal" are all distinct literal strings that the
    # word-based color detector legitimately matches for "sorte") — so the
    # correctness check has to use the same detection the retriever itself
    # runs, not one hardcoded literal, or it flags correct matches as bugs.
    expected_colors = retriever.detect_colors(query)
    assert all(p["category"] == category for p in results)
    assert all(set(p["colors"]) & expected_colors for p in results), (
        f"product matching none of {expected_colors} leaked in: "
        f"{[(p['sku'], p['colors']) for p in results if not set(p['colors']) & expected_colors]}"
    )


@pytest.mark.parametrize("store", STORES)
def test_category_plus_availability_at_specific_store(retriever, store):
    results = retriever.retrieve(f"Har I sofaer på lager i {store}?", top_k=8, focus=[])

    assert all(p["category"] == "sofa" for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results), (
        f"out-of-stock-at-{store} product leaked in: "
        f"{[p['sku'] for p in results if p['availability'][store]['stock_quantity'] < 1]}"
    )


def test_category_plus_generic_in_stock(retriever):
    # "på lager" with no store name = in stock *somewhere*, not a specific store.
    results = retriever.retrieve("Har I sofaer på lager?", top_k=8, focus=[])

    assert all(p["category"] == "sofa" for p in results)
    assert all(
        any(info["stock_quantity"] >= 1 for info in p["availability"].values())
        for p in results
    )


# --- category + two properties -----------------------------------------

def test_category_plus_price_and_color(retriever):
    query = f"Har I {COMMON_COLOR_WORD} sofaer under 8000 kr?"
    results = retriever.retrieve(query, top_k=8, focus=[])
    expected_colors = retriever.detect_colors(query)

    assert all(p["category"] == "sofa" for p in results)
    assert all(set(p["colors"]) & expected_colors for p in results)
    assert all(_effective_price(p) <= 8000 for p in results)


def test_category_plus_price_and_availability(retriever):
    store = STORES[0]
    results = retriever.retrieve(
        f"Har I sofaer under 8000 kr på lager i {store}?", top_k=8, focus=[]
    )

    assert all(p["category"] == "sofa" for p in results)
    assert all(_effective_price(p) <= 8000 for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results)


def test_category_plus_color_and_availability(retriever):
    store = STORES[0]
    query = f"Har I {COMMON_COLOR_WORD} sofaer på lager i {store}?"
    results = retriever.retrieve(query, top_k=8, focus=[])
    expected_colors = retriever.detect_colors(query)

    assert all(p["category"] == "sofa" for p in results)
    assert all(set(p["colors"]) & expected_colors for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results)


# --- sanity: an unfilterable broad query never silently returns nothing --

def test_broad_query_never_falls_back_to_empty_when_matches_exist(retriever):
    for category, phrase in CATEGORY_QUERIES.items():
        results = retriever.retrieve(f"Hvilke {phrase} har I?", top_k=8, focus=[])
        assert results, f"{category} unexpectedly returned nothing for a bare category query"
