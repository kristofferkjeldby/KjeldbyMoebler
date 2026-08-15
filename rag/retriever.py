"""Retrieve relevant products from the FAISS index for a natural-language query.

Used by both the eval runner and the Gradio app, so retrieval behaves
identically in both places.

Pure semantic top-k is wrong for several common question shapes:

  - Enumeration: "what chairs are available in yellow?" — there might be 7
    matches; semantic search would silently truncate to top_k.
  - Store + quantity: "I need 6 dining tables available at the Odense
    store" — needs a specific store's stock checked against a quantity, not
    a semantic guess.
  - Superlatives: "what's the cheapest table?" — depends on comparing price
    across every matching product.
  - Dimension fit: "I need a 60cm deep kitchen unit" — a numeric tolerance
    filter on one dimension, not a semantic match on wording.

So retrieval is two-tier: detect explicit category / color / store /
quantity / stock-availability / price-superlative / dimension intent from
the query (matched against the catalog's own vocabulary), and if anything
was detected, filter (and sort) the full catalog accordingly — the model
only ever sees the top SHOWN_MAX_RESULTS by the catalog's hidden `priority`
field (see RetrievalResult below), including an empty list when nothing
matches (which correctly tells the model "we don't have that"). Otherwise,
fall back to semantic top-k for open-ended/specific questions.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.product_format import product_to_embedding_text  # noqa: E402
from config import (  # noqa: E402
    CATEGORY_DISAMBIGUATION_TERMS,
    CATEGORY_PHRASES,
    CATEGORY_WORDS,
    CHEAP_PHRASES,
    COLOR_WORD_STOPWORDS,
    CROSS_ENCODER_MODEL_NAME,
    DIM_KEYWORDS,
    DIMENSION_TOLERANCE_CM,
    EMBEDDING_MODEL_NAME,
    EXPENSIVE_PHRASES,
    EXTRA_CATEGORY_TRIGGER_WORDS,
    FAISS_INDEX_PATH,
    IN_STOCK_PHRASES,
    OUT_OF_STOCK_PHRASES,
    RAG_METADATA_PATH,
    RETRIEVAL_TOP_K,
    SEMANTIC_CANDIDATE_POOL_SIZE,
    SERIES_INTENT_PHRASES,
    SERIES_INTENT_WORD_PAIRS,
    SHOWN_MAX_RESULTS,
    STORES,
    WORD_CHARS,
)


@dataclass
class RetrievalResult:
    """`shown` is what actually goes in the model's context (<= SHOWN_MAX_RESULTS,
    ranked by the catalog's hidden `priority` field). `pool` is the full
    matching set `shown` was drawn from — pass it back in as next turn's
    `focus` so a follow-up filter (a color, a price cap) narrows the whole
    match set, not just the small subset that happened to be shown. `pool`
    is never narrowed by what the model's reply actually talks about — only
    an explicit new filter narrows it — see app/gradio_app.py.
    """

    shown: list[dict]
    pool: list[dict]
    total_count: int
    # Set only when the query's sole signal was a bare generic term that
    # spans several distinct catalog categories (config.py's
    # CATEGORY_DISAMBIGUATION_TERMS, e.g. "stol" -> office_chair /
    # dining_chair / bar_stool / armchair) with no other filter at all —
    # {category: count}. When set, `shown`/`pool` are empty: the caller
    # should present the category choice (with a per-category link)
    # instead of individual products. See app/gradio_app.py.
    category_breakdown: dict[str, int] | None = None


def _split_shown(matches: list[dict]) -> RetrievalResult:
    """Rank `matches` by the catalog's hidden `priority` field (highest
    first) and split into the model-facing `shown` subset vs. the full
    `pool` used for follow-up narrowing. A product with no `priority` set
    sorts last (0), rather than raising, so this stays safe against any
    catalog data that predates the field.
    """
    ranked = sorted(matches, key=lambda p: p.get("priority", 0), reverse=True)
    return RetrievalResult(shown=ranked[:SHOWN_MAX_RESULTS], pool=ranked, total_count=len(ranked))

# Customers say "sofa" generically to mean any couch-like seating, but the
# catalog's own category taxonomy splits that into three distinct values —
# so a follow-up like "hvilke af de to sofaer..." about a focus that's
# actually a sectional/loveseat shouldn't read as naming an unrelated
# category and blow away the existing focus.
_SOFA_LIKE_CATEGORIES = {"sofa", "sectional", "loveseat"}

_SKU_RE = re.compile(r"\bFRN-\d{4}\b", re.IGNORECASE)
_QUANTITY_RE = re.compile(r"\b(\d+)\b(?!\s?(?:cm|kg|%|kr))", re.IGNORECASE)
_DIM_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*cm", re.IGNORECASE)
_PRICE_THRESHOLD_RE = re.compile(
    r"(?P<dir>under|mindre end|billigere end|op til|højst|over|mere end|dyrere end)"
    r"\s+(?P<value>\d[\d.,]*)\s*(?:kr\.?|kroner)?",
    re.IGNORECASE,
)

_SPELLED_OUT_NUMBERS = {
    "en": 1, "et": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5,
    "seks": 6, "syv": 7, "otte": 8, "ni": 9, "ti": 10, "tolv": 12,
}

# Fuzzy name-match thresholds: require at least this many overlapping
# significant tokens and this fraction of the product name's own tokens
# covered, so a single generic word ("stol") can't spuriously "fuzzy match"
# every chair in the catalog.
_FUZZY_MIN_OVERLAP_TOKENS = 2
_FUZZY_MIN_COVERAGE = 0.6

# multilingual-e5 models expect these prefixes on embedded text for best
# retrieval quality — queries and passages are encoded asymmetrically.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "

_WORD_RE = re.compile(f"[{WORD_CHARS}]+")
_NON_WORD_RE = re.compile(f"[^{WORD_CHARS}\\s]")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _normalize(text: str) -> str:
    return _NON_WORD_RE.sub("", text.lower()).strip()


def _effective_price(product: dict) -> float:
    return product.get("discount_price") or product["normal_price"]


class ProductRetriever:
    def __init__(self) -> None:
        if not FAISS_INDEX_PATH.exists() or not RAG_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"RAG index not found at {FAISS_INDEX_PATH}. Run `python -m rag.build_index` first."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.products: list[dict] = json.loads(RAG_METADATA_PATH.read_text())
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._cross_encoder: CrossEncoder | None = None  # lazy-loaded: only needed for the semantic-fallback tier
        self._store_lookup = {s.lower(): s for s in STORES}
        self._build_vocab()
        self._build_name_index()

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return self._cross_encoder

    def _build_name_index(self) -> None:
        # (normalized name, significant tokens, product) — checked in a query
        # for an exact substring match first, then as a fallback for
        # token-overlap fuzzy matching. Sorted longest-name-first so a more
        # specific name is found before a shorter one that happens to be a
        # substring of it (rare, but cheap to guard against).
        self._name_entries: list[tuple[str, frozenset[str], dict]] = []
        for product in self.products:
            normalized = _normalize(product["name"])
            tokens = frozenset(w for w in normalized.split() if len(w) >= 3)
            self._name_entries.append((normalized, tokens, product))
        self._name_entries.sort(key=lambda entry: len(entry[0]), reverse=True)

    def _build_vocab(self) -> None:
        # category -> (Danish phrase, set of Danish trigger words), from
        # config.py — the phrase ("spisebord") disambiguates compound
        # categories that share a prefix word; the broader word set (plus
        # domain-slang extras) is the fallback for generic queries like
        # "hvilke stole har I".
        self._category_phrases: dict[str, str] = dict(CATEGORY_PHRASES)
        # Danish is heavily compound: "spisebord" (dining table) is a
        # substring of "spisebordsstol" (dining chair) and "sofa" is a
        # substring of "sofaborde" (coffee tables) — plain substring
        # containment would wrongly match both, so phrases are matched on
        # word boundaries instead.
        self._category_phrase_res: dict[str, re.Pattern] = {
            category: re.compile(rf"\b{re.escape(phrase)}\b")
            for category, phrase in self._category_phrases.items()
        }
        self._category_words: dict[str, set[str]] = {
            category: set(words) | EXTRA_CATEGORY_TRIGGER_WORDS.get(category, set())
            for category, words in CATEGORY_WORDS.items()
        }

        all_category_trigger_words = set().union(*self._category_words.values()) if self._category_words else set()

        # color word -> set of exact catalog color strings containing that word
        # (e.g. "yellow" -> {"Sunflower Yellow", "Pale Yellow"}). Words that
        # double as a category trigger (e.g. "frame", shared with the
        # bed_frame category, showing up in a two-tone color like "Black
        # Frame") are excluded — otherwise a query like "bed frame in stock"
        # spuriously also filters by that color and returns nothing.
        self._color_word_to_colors: dict[str, set[str]] = {}
        for product in self.products:
            for color in product["colors"]:
                for word in _words(color):
                    if len(word) < 3 or word in all_category_trigger_words or word in COLOR_WORD_STOPWORDS:
                        continue
                    self._color_word_to_colors.setdefault(word, set()).add(color)

    # --- detection ---

    def _detect_categories(self, query_lower: str, query_words: set[str]) -> set[str]:
        # Prefer precise full-phrase matches ("dining table" narrows to just
        # dining_table) over the broad last-word/plural fallback, which is
        # reserved for generic queries with no specific phrase ("what chairs
        # do you have" should match every chair-like category).
        phrase_matches = {cat for cat, rx in self._category_phrase_res.items() if rx.search(query_lower)}
        if phrase_matches:
            return phrase_matches
        exact_matches = {cat for cat, words in self._category_words.items() if words & query_words}
        if exact_matches:
            return exact_matches
        # Danish is heavily inflected/compounded ("loveseater" = plural of
        # "loveseat", "belysningsprodukter" = "belysning" + "produkter") —
        # exact word equality misses these, so fall back to prefix matching
        # (query token starts with a category word, i.e. word+suffix
        # inflection) as a last resort. Danish compounds also prepend a
        # modifier before the head noun ("trækommode" = "træ" + "kommode",
        # "skindsofa" = "skind" + "sofa") — the head noun (the category) is
        # the END of the compound there, so a suffix check catches that
        # direction the prefix check can't. Only applied when nothing more
        # specific matched, so it can't override a genuine phrase/exact hit
        # from a different category.
        return {
            cat
            for cat, words in self._category_words.items()
            for word in words
            if len(word) >= 4 and any(qw.startswith(word) or qw.endswith(word) for qw in query_words)
        }

    @staticmethod
    def _detect_disambiguation_categories(query_words: set[str]) -> set[str]:
        """A bare generic term ("stol") that's the shared head-noun of
        several specific catalog categories, none of which is itself named
        that word — see CATEGORY_DISAMBIGUATION_TERMS in config.py.
        """
        result: set[str] = set()
        for word in query_words:
            result |= CATEGORY_DISAMBIGUATION_TERMS.get(word, set())
        return result

    def _detect_colors(self, query_words: set[str]) -> set[str]:
        matched_words = set(self._color_word_to_colors) & query_words
        colors: set[str] = set()
        for word in matched_words:
            colors |= self._color_word_to_colors[word]
        return colors

    def _detect_store(self, query_words: set[str]) -> str | None:
        for word in query_words:
            if word in self._store_lookup:
                return self._store_lookup[word]
        return None

    @staticmethod
    def _detect_quantity(query_lower: str, query_words: set[str]) -> int | None:
        match = _QUANTITY_RE.search(query_lower)
        if match:
            return int(match.group(1))
        # Spelled-out small numbers ("fire dining chairs") are natural Danish
        # phrasing and just as common as digits in this range — a regex on
        # digits alone misses them entirely.
        for word, value in _SPELLED_OUT_NUMBERS.items():
            if word in query_words:
                return value
        return None

    @staticmethod
    def _detect_stock_filter(query_lower: str, query_words: set[str]) -> bool | None:
        if any(phrase in query_lower for phrase in OUT_OF_STOCK_PHRASES):
            return False
        if any(phrase in query_lower for phrase in IN_STOCK_PHRASES) or "lager" in query_words:
            return True
        return None

    @staticmethod
    def _detect_price_sort(query_lower: str) -> str | None:
        if any(phrase in query_lower for phrase in CHEAP_PHRASES):
            return "asc"
        if any(phrase in query_lower for phrase in EXPENSIVE_PHRASES):
            return "desc"
        return None

    @staticmethod
    def _detect_price_threshold(query_lower: str) -> tuple[float, float] | None:
        """"under 4000 kr" -> (0, 4000); "over 4000 kr" -> (4000, inf) — a
        numeric bound, distinct from the superlative "cheapest" detector
        above. Without this, a budget query silently falls back to an
        unbounded category filter capped at SHOWN_MAX_RESULTS, which can
        miss the actual cheap/expensive matches entirely.
        """
        match = _PRICE_THRESHOLD_RE.search(query_lower)
        if not match:
            return None
        value = float(match.group("value").replace(".", "").replace(",", "."))
        direction = match.group("dir").lower()
        if direction in ("under", "mindre end", "billigere end", "op til", "højst"):
            return (0.0, value)
        return (value, float("inf"))

    def _detect_name_matches(self, query_lower: str) -> list[dict]:
        """Products whose name is (near-)cited in the query, e.g. 'is the
        Ironforge Industrial Writing Desk in stock?'. Exact substring match
        wins outright; multiple distinct products can match a comparison
        query naming several items, or products that happen to share an
        identical name (some series-generated names repeat) — both are
        returned rather than picking one arbitrarily. Falls back to
        token-overlap fuzzy matching only when nothing matched exactly.
        """
        normalized_query = _normalize(query_lower)
        # Space-padded so a name match only counts on a real word boundary —
        # plain substring containment let short, generic single-word product
        # names (e.g. a product literally named "Rustik") get an "exact"
        # match against any query merely containing that stem as a prefix of
        # a longer word (e.g. the adjective "rustikt"), which then won
        # outright over the fuzzy-overlap tier and returned a spurious
        # single-product match for what was really an open-ended query.
        padded_query = f" {normalized_query} "
        exact: list[dict] = []
        exact_names: list[str] = []
        seen_names: set[str] = set()
        for name, _tokens, product in self._name_entries:
            if name and f" {name} " in padded_query and name not in seen_names:
                exact.append(product)
                exact_names.append(name)
                seen_names.add(name)
        if exact:
            # Drop matches that are themselves a substring of another
            # matched name — e.g. a generic "Elegance" product's name is a
            # spurious hit when the customer actually quoted a longer, more
            # specific name like "Skandinavisk Elegance Spisestuebord" that
            # happens to contain it. Keeping both breaks the "exactly one
            # anchor" check series/comparison queries rely on downstream.
            return [
                product
                for product, name in zip(exact, exact_names)
                if not any(name != other and name in other for other in exact_names)
            ]

        query_tokens = frozenset(w for w in normalized_query.split() if len(w) >= 3)
        if not query_tokens:
            return []
        best_score = 0.0
        best_matches: list[dict] = []
        for name, tokens, product in self._name_entries:
            if not tokens:
                continue
            overlap = tokens & query_tokens
            if len(overlap) < _FUZZY_MIN_OVERLAP_TOKENS:
                continue
            coverage = len(overlap) / len(tokens)
            if coverage < _FUZZY_MIN_COVERAGE:
                continue
            if coverage > best_score:
                best_score = coverage
                best_matches = [product]
            elif coverage == best_score:
                best_matches.append(product)
        return best_matches

    @staticmethod
    def _detect_dimension(query_lower: str) -> tuple[str, float, float] | None:
        for match in _DIM_NUMBER_RE.finditer(query_lower):
            value = float(match.group(1).replace(",", "."))
            window = query_lower[max(0, match.start() - 20): match.end() + 20]
            for keyword, field in DIM_KEYWORDS.items():
                if keyword in window:
                    return (field, value, DIMENSION_TOLERANCE_CM)
        return None

    # --- filtering ---

    def _filter(
        self,
        categories: set[str] | None = None,
        colors: set[str] | None = None,
        store: str | None = None,
        min_quantity: int | None = None,
        stock_filter: bool | None = None,
        price_sort: str | None = None,
        price_range: tuple[float, float] | None = None,
        dimension: tuple[str, float, float] | None = None,
        candidates: list[dict] | None = None,
    ) -> list[dict]:
        matches = candidates if candidates is not None else self.products
        if categories:
            matches = [p for p in matches if p["category"] in categories]
        if colors:
            matches = [p for p in matches if set(p["colors"]) & colors]

        if store is not None:
            required = min_quantity or 1
            matches = [p for p in matches if p["availability"][store]["stock_quantity"] >= required]
        elif stock_filter is True:
            required = min_quantity or 1
            matches = [p for p in matches if any(info["stock_quantity"] >= required for info in p["availability"].values())]
        elif stock_filter is False:
            matches = [p for p in matches if all(info["stock_quantity"] == 0 for info in p["availability"].values())]

        if dimension is not None:
            field, target, tolerance = dimension
            matches = [p for p in matches if abs(p["dimensions"][field] - target) <= tolerance]

        if price_range is not None:
            lo, hi = price_range
            matches = [p for p in matches if lo <= _effective_price(p) <= hi]

        if price_sort is not None:
            matches = sorted(matches, key=_effective_price, reverse=(price_sort == "desc"))

        return matches

    def filter_products(
        self,
        category: str | None = None,
        color_word: str | None = None,
        store: str | None = None,
        min_quantity: int | None = None,
        stock_filter: bool | None = None,
        price_sort: str | None = None,
        dimension: tuple[str, float, float] | None = None,
    ) -> list[dict]:
        """Convenience wrapper over `_filter` for training/eval data generation,
        so examples are built from the exact same matching logic used at
        inference time instead of a separately-maintained copy of it."""
        colors = self._color_word_to_colors.get(color_word.lower(), set()) if color_word else None
        return self._filter(
            categories={category} if category else None,
            colors=colors,
            store=store,
            min_quantity=min_quantity,
            stock_filter=stock_filter,
            price_sort=price_sort,
            dimension=dimension,
        )

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K, focus: list[dict] | None = None) -> RetrievalResult:
        """`focus` is the FULL matching set the conversation is currently
        "about" — typically the previous turn's `RetrievalResult.pool`, not
        just what was shown — so a follow-up filter ("fås den i grå?") is
        applied against everything that matched originally, not just the
        handful the model happened to show. Explicit signals (SKU, product
        name, or a new category) always override focus, since those mean
        the customer has moved on to something else.

        The return value's `shown` (<= SHOWN_MAX_RESULTS, ranked by the
        catalog's hidden `priority` field) is what should go in the model's
        context; `pool` is what the caller should pass back in as next
        turn's `focus` so narrowing continues to work against the whole
        matching set, not just what was shown.
        """
        focus = focus or []
        query_lower = query.lower()
        query_words = _words(query)

        # A customer citing an exact SKU always wins over every other signal.
        # No priority ranking here — these are the exact products the
        # customer named, not a broad match that needs narrowing down.
        sku_matches = [m.group(0).upper() for m in _SKU_RE.finditer(query)]
        if sku_matches:
            found = [p for sku in sku_matches if (p := self.get_by_sku(sku)) is not None]
            if found:
                return RetrievalResult(shown=found, pool=found, total_count=len(found))

        # Product(s) named or closely referenced in the query take priority
        # over attribute filters and semantic search — this is what fixes
        # "is the <specific product> in stock" style questions that semantic
        # top-k alone frequently missed among thousands of catalog items.
        name_matches = self._detect_name_matches(query_lower)
        if name_matches:
            is_series_query = any(phrase in query_lower for phrase in SERIES_INTENT_PHRASES) or any(
                a in query_words and b in query_words for a, b in SERIES_INTENT_WORD_PAIRS
            )
            if is_series_query and len(name_matches) == 1 and name_matches[0].get("series_id"):
                siblings = self.get_series(name_matches[0]["series_id"])
                if siblings:
                    return _split_shown(siblings)
            return _split_shown(name_matches)

        categories = self._detect_categories(query_lower, query_words)
        colors = self._detect_colors(query_words)
        store = self._detect_store(query_words)
        quantity = self._detect_quantity(query_lower, query_words)
        stock_filter = self._detect_stock_filter(query_lower, query_words)
        price_sort = self._detect_price_sort(query_lower)
        price_range = self._detect_price_threshold(query_lower)
        dimension = self._detect_dimension(query_lower)

        # A bare generic term ("Hvilke stole har I?", "Jeg skal bruge en
        # sofa") that spans several distinct catalog categories. For
        # "stol"/"bord" the bare word matches no category at all
        # (`categories` is empty). "sofa" is different: it's ALSO one of
        # the sofa-family's own literal category words (CATEGORY_WORDS),
        # so the exact-match tier alone already returns {"sofa"} — not
        # empty. The `issubset(...) and len(...) <= 1` check below still
        # recognizes that as "nothing MORE SPECIFIC than the generic word
        # itself was named": a phrase-tier match naming a sibling directly
        # ("hjørnesofa") changes `categories` to something that isn't a
        # same-or-smaller subset of just this word's own family, so that
        # case is correctly left to search directly instead of asking.
        disambiguation_categories = self._detect_disambiguation_categories(query_words)
        is_pure_disambiguation = bool(disambiguation_categories) and categories.issubset(
            disambiguation_categories
        ) and len(categories) <= 1
        other_signal = bool(
            colors or store or stock_filter is not None
            or price_sort is not None or price_range is not None or dimension is not None
        )

        if is_pure_disambiguation:
            if not other_signal:
                # Nothing else to narrow by — rather than guess a subtype
                # (or fall through to semantic search, which has no notion
                # of "stol" and returns unrelated noise), tell the caller
                # which categories actually matched and how many products
                # are in each, so the customer can be asked to pick one
                # instead of being shown an arbitrary blend.
                counts = {
                    cat: sum(1 for p in self.products if p["category"] == cat)
                    for cat in disambiguation_categories
                }
                return RetrievalResult(
                    shown=[], pool=[], total_count=sum(counts.values()), category_breakdown=counts
                )
            # A color/price/store alongside "stole" is enough specificity
            # to just search the union of categories directly instead of
            # asking first — but the union still has to scope the search,
            # or "sorte stole" would silently search "sort, any category"
            # across the whole catalog instead of just chairs.
            categories = disambiguation_categories

        any_filter = bool(
            categories or colors or store or stock_filter is not None
            or price_sort is not None or price_range is not None or dimension is not None
        )

        # No new category named: treat this as a follow-up about the focused
        # product(s) rather than a fresh catalog-wide query. A category
        # match means the customer switched topics ("what about a lamp?"),
        # which should bypass focus and search fresh — UNLESS every matched
        # category is already represented in the current focus (e.g. asking
        # "hvilke af de to sofaer..." about a focus that already contains a
        # sofa-like item), which is still just talking about the same set,
        # not a switch. Sofa/sectional/loveseat are checked as one family
        # since customers say "sofa" for any of them.
        def _cat_family(cat: str) -> str:
            return "sofa_like" if cat in _SOFA_LIKE_CATEGORIES else cat

        focus_families = {_cat_family(p["category"]) for p in focus}
        categories_are_new = bool(categories) and not any(_cat_family(c) in focus_families for c in categories)
        if focus and not categories_are_new:
            if any_filter:
                # `candidates=focus` filters against the FULL previous pool
                # (not just what was shown) — this is what lets "fås den i
                # grå?" surface gray options beyond the ones already shown,
                # not just gray options among them.
                filtered = self._filter(
                    colors=colors or None,
                    store=store,
                    min_quantity=quantity,
                    stock_filter=stock_filter,
                    price_sort=price_sort,
                    price_range=price_range,
                    dimension=dimension,
                    candidates=focus,
                )
                # Even an empty result is meaningful here (e.g. "fås den i
                # sort?" when it isn't) — don't fall back to a fresh search,
                # that would silently answer about a different product.
                return _split_shown(filtered)
            return _split_shown(focus)

        # store/stock/price alone (no category, color, or dimension signal)
        # isn't a trustworthy "what kind of product" filter over the WHOLE
        # catalog — e.g. a category word we failed to recognize plus "i
        # stock in Odense" would otherwise silently return whatever
        # products happen to be first in catalog order with Odense stock,
        # which looks like a real answer but is unrelated noise. Semantic
        # search on the raw query is far more likely to actually find the
        # right products in that case.
        has_product_type_signal = bool(categories or colors or dimension)
        if any_filter and has_product_type_signal:
            matches = self._filter(
                categories=categories or None,
                colors=colors or None,
                store=store,
                min_quantity=quantity,
                stock_filter=stock_filter,
                price_sort=price_sort,
                price_range=price_range,
                dimension=dimension,
            )
            # Explicit filter detected: return it as-is (even if empty — an
            # empty list correctly tells the model "we don't have that").
            return _split_shown(matches)

        # Semantic search doesn't get the wide-list treatment: it's already
        # relevance-ranked top-k for an open-ended query, not a structured
        # match where "everything else that matched" is a meaningful set to
        # offer a "see all" link into.
        semantic_matches = self._semantic_search(query, top_k)
        return RetrievalResult(shown=semantic_matches, pool=semantic_matches, total_count=len(semantic_matches))

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """Two-stage: the bi-encoder (fast, independently-embedded vectors)
        recalls a wide candidate pool via FAISS, then a cross-encoder (scores
        the query and each candidate jointly, more precise but too slow to
        run over the whole catalog) reranks that pool down to top_k. This is
        the tier for genuinely open-ended/descriptive queries that hit
        neither the name-match nor attribute-filter tiers.
        """
        query_vec = self.model.encode([_E5_QUERY_PREFIX + query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        pool_size = max(top_k, SEMANTIC_CANDIDATE_POOL_SIZE)
        _scores, indices = self.index.search(query_vec, pool_size)
        candidates = [self.products[i] for i in indices[0] if i != -1]
        if not candidates:
            return []

        cross_encoder = self._get_cross_encoder()
        pairs = [(query, product_to_embedding_text(p)) for p in candidates]
        rerank_scores = cross_encoder.predict(pairs)
        ranked = [p for _, p in sorted(zip(rerank_scores, candidates), key=lambda pair: pair[0], reverse=True)]
        return ranked[:top_k]

    def get_by_sku(self, sku: str) -> dict | None:
        return next((p for p in self.products if p["sku"] == sku), None)

    def get_series(self, series_id: str) -> list[dict]:
        return [p for p in self.products if p.get("series_id") == series_id]

    def all_categories(self) -> list[str]:
        return sorted(self._category_words)

    def all_color_words(self) -> list[str]:
        return sorted(self._color_word_to_colors)

    def detect_colors(self, query: str) -> set[str]:
        """Public wrapper around the same color-word detection `retrieve()`
        uses internally — lets a caller ask "did the customer name a color
        in this message, and which exact catalog color string is that?"
        independently of running a full retrieval."""
        return self._detect_colors(_words(query))
