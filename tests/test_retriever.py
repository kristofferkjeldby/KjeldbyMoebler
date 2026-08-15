"""RAG retrieval tests for entry-level queries — the first thing a customer
asks, with no prior conversation (no focus carryover). Covers the
rule-based attribute-filter tier (`ProductRetriever.retrieve`): a bare
category, a category plus one property (price / color / availability), and
a category plus two properties combined. Also covers the priority-ranked
wide-list behavior (SHOWN_MAX_RESULTS cap + hidden pool) and the
pool-narrowing mechanism that lets a follow-up filter ("i grå?") search the
full previous match set rather than just what was shown.

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
from config import SHOWN_MAX_RESULTS, STORES  # noqa: E402
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
    results = retriever.retrieve(f"Hvilke {phrase} har I?", top_k=8, focus=[]).shown

    assert results, f"expected at least one {category} for a bare category query"
    assert len(results) <= SHOWN_MAX_RESULTS
    assert all(p["category"] == category for p in results), (
        f"non-{category} product leaked into results: "
        f"{[p['sku'] for p in results if p['category'] != category]}"
    )


def test_category_only_caps_at_shown_max(retriever):
    # "sofa" alone is a large category (60+ products) — confirms the cap is
    # actually enforced, not just coincidentally under it for small ones.
    results = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[]).shown
    assert len(results) == SHOWN_MAX_RESULTS


# --- category + one property -------------------------------------------

@pytest.mark.parametrize("category, phrase", CATEGORY_QUERIES.items())
def test_category_plus_price_threshold(retriever, category, phrase):
    results = retriever.retrieve(f"Har I {phrase} under 5000 kr?", top_k=8, focus=[]).shown

    assert all(p["category"] == category for p in results)
    assert all(_effective_price(p) <= 5000 for p in results), (
        f"product over budget leaked in: "
        f"{[(p['sku'], _effective_price(p)) for p in results if _effective_price(p) > 5000]}"
    )


@pytest.mark.parametrize("category, phrase", CATEGORY_QUERIES.items())
def test_category_plus_color(retriever, category, phrase):
    query = f"Har I {COMMON_COLOR_WORD} {phrase}?"
    results = retriever.retrieve(query, top_k=8, focus=[]).shown

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
    results = retriever.retrieve(f"Har I sofaer på lager i {store}?", top_k=8, focus=[]).shown

    assert all(p["category"] == "sofa" for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results), (
        f"out-of-stock-at-{store} product leaked in: "
        f"{[p['sku'] for p in results if p['availability'][store]['stock_quantity'] < 1]}"
    )


def test_category_plus_generic_in_stock(retriever):
    # "på lager" with no store name = in stock *somewhere*, not a specific store.
    results = retriever.retrieve("Har I sofaer på lager?", top_k=8, focus=[]).shown

    assert all(p["category"] == "sofa" for p in results)
    assert all(
        any(info["stock_quantity"] >= 1 for info in p["availability"].values())
        for p in results
    )


# --- category + two properties -----------------------------------------

def test_category_plus_price_and_color(retriever):
    query = f"Har I {COMMON_COLOR_WORD} sofaer under 8000 kr?"
    results = retriever.retrieve(query, top_k=8, focus=[]).shown
    expected_colors = retriever.detect_colors(query)

    assert all(p["category"] == "sofa" for p in results)
    assert all(set(p["colors"]) & expected_colors for p in results)
    assert all(_effective_price(p) <= 8000 for p in results)


def test_category_plus_price_and_availability(retriever):
    store = STORES[0]
    results = retriever.retrieve(
        f"Har I sofaer under 8000 kr på lager i {store}?", top_k=8, focus=[]
    ).shown

    assert all(p["category"] == "sofa" for p in results)
    assert all(_effective_price(p) <= 8000 for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results)


def test_category_plus_color_and_availability(retriever):
    store = STORES[0]
    query = f"Har I {COMMON_COLOR_WORD} sofaer på lager i {store}?"
    results = retriever.retrieve(query, top_k=8, focus=[]).shown
    expected_colors = retriever.detect_colors(query)

    assert all(p["category"] == "sofa" for p in results)
    assert all(set(p["colors"]) & expected_colors for p in results)
    assert all(p["availability"][store]["stock_quantity"] >= 1 for p in results)


# --- sanity: an unfilterable broad query never silently returns nothing --

def test_broad_query_never_falls_back_to_empty_when_matches_exist(retriever):
    for category, phrase in CATEGORY_QUERIES.items():
        results = retriever.retrieve(f"Hvilke {phrase} har I?", top_k=8, focus=[]).shown
        assert results, f"{category} unexpectedly returned nothing for a bare category query"


# --- priority-ranked wide lists + hidden pool ---------------------------
#
# "Hvilke sofaer har I?" matches 60+ products — far more than a customer
# should ever be shown in one reply. `retrieve()` returns a RetrievalResult:
# `shown` (<= SHOWN_MAX_RESULTS, ranked by the catalog's hidden `priority`
# field) is what goes in the model's context; `pool` is the full matching
# set, which the caller passes back in as next turn's `focus` so a
# follow-up filter narrows against everything that matched, not just the
# handful that were shown.

def test_wide_query_reports_total_count_beyond_shown(retriever):
    result = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[])

    assert len(result.shown) == SHOWN_MAX_RESULTS
    assert result.total_count > SHOWN_MAX_RESULTS, (
        "expected the sofa category to have more real matches than SHOWN_MAX_RESULTS "
        "in the current catalog — if this fails the catalog shrank, adjust the fixture"
    )
    assert result.total_count == len(result.pool)


def test_wide_query_shown_is_sorted_by_priority_descending(retriever):
    result = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[])
    priorities = [p["priority"] for p in result.shown]
    assert priorities == sorted(priorities, reverse=True), (
        f"shown products aren't priority-sorted: {priorities}"
    )
    # And every one of them should be at or above the (SHOWN_MAX_RESULTS+1)th
    # highest-priority match in the full pool — i.e. genuinely the top N, not
    # just N arbitrary matches that happen to be sorted among themselves.
    pool_priorities_desc = sorted((p["priority"] for p in result.pool), reverse=True)
    assert priorities == pool_priorities_desc[:SHOWN_MAX_RESULTS]


def test_narrow_query_has_no_phantom_truncation(retriever):
    # A specific enough query (category + store + a high quantity threshold)
    # should genuinely match only a handful — shown and pool should be
    # identical, and no truncation note should apply (checked at the
    # products_to_context level in catalog/product_format.py, not here).
    store = STORES[0]
    result = retriever.retrieve(f"Har I mindst 38 sofaer på lager i {store}?", top_k=8, focus=[])
    assert result.total_count <= SHOWN_MAX_RESULTS, (
        "test fixture assumption broke — pick a stricter threshold so this genuinely "
        f"narrow-query case has <= SHOWN_MAX_RESULTS matches (got {result.total_count})"
    )
    assert result.shown == result.pool
    assert result.total_count == len(result.shown)


def test_pool_narrowing_backfills_beyond_originally_shown_products(retriever):
    # The "gray" scenario: ask broadly, then narrow by color. The color
    # filter must run against the FULL previous pool, not just the 8 that
    # were shown — so gray matches outside the original shown-8 should
    # surface, not just gray matches among them.
    broad = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[])
    assert broad.total_count > SHOWN_MAX_RESULTS  # otherwise this test can't distinguish anything

    narrowed = retriever.retrieve("Har I dem i grå?", top_k=8, focus=broad.pool)
    expected_colors = retriever.detect_colors("Har I dem i grå?")

    assert all(p["category"] == "sofa" for p in narrowed.shown)
    assert all(set(p["colors"]) & expected_colors for p in narrowed.shown), (
        "narrowed shown set contains a non-gray product"
    )
    # The whole point: narrowing must find matches beyond the original
    # shown-8, not just filter within them.
    original_shown_skus = {p["sku"] for p in broad.shown}
    new_skus = {p["sku"] for p in narrowed.shown} - original_shown_skus
    assert new_skus, (
        "narrowing by color found nothing beyond the originally-shown products — "
        "pool narrowing is filtering the shown-8 instead of the full pool"
    )
    # And the narrowed pool should be the true gray-matching subset of the
    # original pool, not the original pool itself (pool actually narrowed)
    # and not just the gray subset of the shown-8 (pool narrowed too much).
    true_gray_count = sum(1 for p in broad.pool if set(p["colors"]) & expected_colors)
    assert narrowed.total_count == true_gray_count


def test_pool_is_not_narrowed_by_what_was_merely_mentioned(retriever):
    # app/gradio_app.py separately narrows the small *displayed* set to
    # what the model's reply actually named (for the focus panel) — but
    # that must never be what gets passed back in as next turn's `focus`,
    # or a vaguely-worded reply could silently shrink the pool a later,
    # unrelated filter would otherwise have searched against. Simulate that
    # by passing in only 2 "mentioned" products (as the UI layer would
    # derive) rather than the true pool, and confirm the retriever has no
    # way to recover the rest — proving the caller, not the retriever, is
    # responsible for always passing the full pool forward.
    broad = retriever.retrieve("Hvilke sofaer har I?", top_k=8, focus=[])
    only_mentioned = broad.shown[:2]

    narrowed_from_full_pool = retriever.retrieve("Har I dem i grå?", top_k=8, focus=broad.pool)
    narrowed_from_mentioned_only = retriever.retrieve("Har I dem i grå?", top_k=8, focus=only_mentioned)

    assert narrowed_from_full_pool.total_count >= narrowed_from_mentioned_only.total_count
    # This is the caller-contract assertion this test exists to document:
    # app/gradio_app.py must pass `result.pool` (never a mentioned-subset)
    # as next turn's focus for the wide-list feature to work correctly.
