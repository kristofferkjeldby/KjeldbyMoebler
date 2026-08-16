"""Pure conversation-turn logic shared between the live Gradio chat
(app/gradio_app.py) and the offline case-runner (tests/run_case_tests.py).

Kept in one place so replaying a captured case (see cases/) exercises
exactly the same retrieval/prompt-building/post-processing/focus-narrowing
rules the live chat used when the case was recorded, instead of two
implementations that can silently drift apart — the same reasoning behind
rag.retriever.ProductRetriever.filter_products() being one shared
implementation rather than separately maintained copies for inference vs.
data generation.

Nothing here talks to Gradio or streams — `build_prompt` and
`finalize_turn` are plain functions taking/returning plain data, callable
turn-by-turn from either the streaming UI loop or a non-streaming replay.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

from catalog.product_format import (
    CATEGORY_LABELS,
    CATEGORY_LABELS_PLURAL,
    category_breakdown_to_context,
    products_to_context,
)
from config import CATEGORY_URL_BASE, RETRIEVAL_TOP_K, SYSTEM_PROMPT_TEMPLATE
from rag.retriever import ProductRetriever, RetrievalResult

# --- link rendering ------------------------------------------------------

# Renders any SKU or retrieved-product name the model mentions as a link to
# a fake product URL. The chat runs in a separate-origin iframe embedded in
# the landing page, so a click can't reach that page's DOM directly — a
# head script (PRODUCT_LINK_SCRIPT in app/gradio_app.py) intercepts the
# click and posts a message to the parent window, which opens the product
# modal.
_SKU_RE = re.compile(r"\bFRN-\d{4}\b")
_PRODUCT_URL = "https://kjeldbymobler.dk/produkt/{sku}"

# The model occasionally generates a complete, well-formed product link on
# its own (having presumably picked up the "[Name](https://.../produkt/SKU)"
# shape from context) — without protecting those first, the SKU substitution
# below matches the SKU sitting inside that URL, and the name substitution
# matches the name sitting inside that link text, each wrapping it AGAIN
# and producing garbled nested links like "[[Name](url)](url/[SKU](url))".
_EXISTING_LINK_RE = re.compile(r"\[[^\]]+\]\(https://kjeldbymobler\.dk/produkt/[^\)]+\)")

# The model has occasionally picked up the "[Se alle N <kategori> ->](...)"
# shape from its own context (the truncation note in products_to_context())
# and written its own version alongside the one finalize_turn always
# appends deterministically afterward — producing two "see all" links for
# the same category with two different labels (the model's own phrasing
# vs. CATEGORY_LABELS' fixed one). Since the appended one is always
# correct and the model's is redundant at best, any model-written category
# link is stripped out before the real one is added — never the reverse.
_MODEL_CATEGORY_LINK_RE = re.compile(r"\n*\[[^\]]*\]\(https://kjeldbymobler\.dk/kategori/[^\)]+\)")


def _safe_stream_prefix(text: str) -> str:
    """Trim `text` back to before any not-yet-closed markdown link.

    The model occasionally writes its own markdown link (see the comment on
    `_MODEL_CATEGORY_LINK_RE` above) directly in its streamed reply. Shown
    token-by-token as-is, that link's raw syntax — a bare "[", then the link
    text, then "](https://kjeldby..." — is visible mid-stream before it's
    complete. Holding back everything from the last unresolved "[" onward
    means the customer only ever sees a link once it's fully formed, never
    its broken-looking raw markdown while it's still being typed out.
    """
    idx = text.rfind("[")
    if idx == -1:
        return text
    close_bracket = text.find("]", idx)
    if close_bracket == -1 or close_bracket + 1 >= len(text) or text[close_bracket + 1] != "(":
        return text[:idx]
    if text.find(")", close_bracket) == -1:
        return text[:idx]
    return text


def _linkify(text: str, products: list[dict]) -> str:
    placeholders: dict[str, str] = {}

    def protect(m: re.Match) -> str:
        key = f"\x00LINK{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    text = _EXISTING_LINK_RE.sub(protect, text)

    text = _SKU_RE.sub(lambda m: f"[{m.group(0)}]({_PRODUCT_URL.format(sku=m.group(0))})", text)

    names = sorted({p["name"] for p in products}, key=len, reverse=True)
    name_to_sku = {p["name"]: p["sku"] for p in products}
    for name in names:
        sku = name_to_sku[name]
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        text = pattern.sub(lambda m, sku=sku: f"[{m.group(0)}]({_PRODUCT_URL.format(sku=sku)})", text)

    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


# The fine-tuned model was trained heavily on comma-separated enumeration
# prose ("X, Y og Z") and doesn't reliably switch to a bullet-list format
# just from a system-prompt instruction — so runs of 3+ linked products
# get reformatted into a markdown list here instead of at generation time.
# The model also isn't consistent about *how* it phrases the enumeration
# from one sample to the next (quoted vs. unquoted links, an inline price
# after each one or not), so both an item and the separator between items
# tolerate that variation rather than matching one exact phrasing.
_ENUM_PRICE_RE = r"(?:\s*(?:til\s+)?\(?[\d.,]+\s*kr\.?\)?)?"
_ENUM_ITEM_RE = rf"['\"]?\[[^\]]+\]\(https://kjeldbymobler\.dk/produkt/[^\)]+\)['\"]?{_ENUM_PRICE_RE}"
# Order matters: the combined "X, Y, og Z" serial-comma form (comma
# immediately followed by "og") must be tried before the bare comma
# alternative, or the bare comma matches first, leaves "og [Next Item]"
# behind (neither remaining alternative can pick that up — "og" itself
# isn't the required "[" an item starts with, and the leading whitespace
# a bare " og " match needs was already consumed by the comma's own
# trailing \s*), and the whole run — and every item after that point —
# silently falls out of the match instead of getting bulletized.
_ENUM_SEP_RE = r"(?:,\s+og\s+|,\s*|\s+og\s+)"
_ENUM_RUN_RE = re.compile(rf"({_ENUM_ITEM_RE}(?:{_ENUM_SEP_RE}{_ENUM_ITEM_RE}){{2,}})\.?")


def _bulletize_enumerations(text: str) -> str:
    def repl(m: re.Match) -> str:
        items = re.findall(_ENUM_ITEM_RE, m.group(1))
        return "\n" + "\n".join(f"- {item.strip()}" for item in items) + "\n"

    return _ENUM_RUN_RE.sub(repl, text)


# Retrieval often returns more candidates than the model actually names in
# its reply (e.g. "sofaer under 9000 kr" may retrieve 10 but the reply only
# lists 4) — narrowing focus to what was actually mentioned keeps the "I
# fokus" panel in sync with what the customer can see in the chat, instead
# of showing extra products the reply never brought up.
_MENTIONED_SKU_RE = re.compile(r"/produkt/(FRN-\d{4})")


def _mentioned_products(rendered_text: str, retrieved: list[dict]) -> list[dict]:
    mentioned_skus = dict.fromkeys(_MENTIONED_SKU_RE.findall(rendered_text))
    by_sku = {p["sku"]: p for p in retrieved}
    result = [by_sku[sku] for sku in mentioned_skus if sku in by_sku]
    return result or retrieved


def _reorder_by_mention(rendered_text: str, retrieved: list[dict]) -> list[dict]:
    """Same idea as `_mentioned_products`, but for a pure follow-up that
    didn't trigger a fresh retrieval ("which of these has the best
    reviews?") — the reply may only re-mention one product by name, but
    that's it picking a winner within the existing set, not narrowing what
    the set *is*. Whatever got mentioned moves to the front; nothing drops.
    """
    mentioned_skus = list(dict.fromkeys(_MENTIONED_SKU_RE.findall(rendered_text)))
    by_sku = {p["sku"]: p for p in retrieved}
    mentioned = [by_sku[sku] for sku in mentioned_skus if sku in by_sku]
    rest = [p for p in retrieved if p["sku"] not in mentioned_skus]
    return mentioned + rest


def focus_payload(products: list[dict], detected_colors_lower: set[str]) -> str:
    return json.dumps([
        {
            "sku": p["sku"],
            "name": p["name"],
            "category": p["category"],
            "price": p.get("discount_price") or p["normal_price"],
            "normal_price": p["normal_price"],
            "discount_percent": p.get("discount_percent") or 0,
            "selected_color": next(
                (c for c in p.get("colors", []) if c.lower() in detected_colors_lower), None
            ),
        }
        for p in products[:8]
    ])


# --- turn-level logic ------------------------------------------------

@dataclass
class TurnPrompt:
    result: RetrievalResult
    system_prompt: str
    detected_colors: set[str]
    new_colors: set[str]


def build_prompt(retriever: ProductRetriever, message: str, previous_pool: list[dict]) -> TurnPrompt:
    result = retriever.retrieve(message, top_k=RETRIEVAL_TOP_K, focus=previous_pool)
    # Computed once up front (not just when narrowing focus at the end) so
    # the "see all" link in finalize_turn can carry the same active color
    # filter into the category modal — otherwise "Se alle 10 røde
    # kontorstole" would open the modal to all 75 unfiltered, a mismatch
    # between what the link promises and what clicking it shows.
    detected_colors = retriever.detect_colors(message)
    new_colors = {c.lower() for c in detected_colors} if detected_colors else set()
    context = (
        category_breakdown_to_context(result.category_breakdown)
        if result.category_breakdown
        else products_to_context(result.shown, result.total_count)
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return TurnPrompt(result=result, system_prompt=system_prompt, detected_colors=detected_colors, new_colors=new_colors)


@dataclass
class FinalizedTurn:
    rendered: str
    new_pool: list[dict]
    new_shown: list[dict]
    new_colors: set[str]
    is_disambiguation: bool


def finalize_turn(
    raw_text: str,
    prompt: TurnPrompt,
    previous_pool: list[dict],
    previous_shown: list[dict],
    previous_colors: set[str],
) -> FinalizedTurn:
    result = prompt.result
    # `new_colors` falls back to an empty set (see build_prompt) when
    # nothing was freshly detected this turn — the caller's own previous
    # colors are the correct fallback, not an empty set, so that's applied
    # here rather than inside build_prompt (which doesn't have access to
    # `previous_colors`).
    new_colors = prompt.new_colors or previous_colors

    rendered = _bulletize_enumerations(_linkify(raw_text, result.shown))
    rendered = _MODEL_CATEGORY_LINK_RE.sub("", rendered).rstrip()

    # More matched than were shown: append a deterministic link to the
    # category-table modal rather than trusting the (un-fine-tuned) model
    # to construct the URL itself — same reliability reasoning as why
    # product links are code-appended via _linkify rather than generated
    # by the model.
    if result.category_breakdown:
        # One link per category, largest first, so the customer can jump
        # straight to whichever type they actually meant.
        for category, count in sorted(result.category_breakdown.items(), key=lambda item: -item[1]):
            label = CATEGORY_LABELS_PLURAL.get(category, category)
            url = CATEGORY_URL_BASE.format(category=category)
            rendered += f"\n\n[Se alle {count} {label} →]({url})"
    elif result.total_count > len(result.shown) and result.shown:
        category = result.shown[0]["category"]
        label = CATEGORY_LABELS.get(category, category)
        see_all_url = CATEGORY_URL_BASE.format(category=category)
        # Only attach a color filter freshly named THIS turn, not one
        # merely carried over from a prior turn — a stale color from an
        # earlier, unrelated topic could otherwise get attached to a
        # category it was never actually filtered by.
        if prompt.detected_colors:
            see_all_url += "?farver=" + quote(json.dumps(sorted(prompt.detected_colors)))
        rendered += f"\n\n[Se alle {result.total_count} {label} →]({see_all_url})"

    if result.category_breakdown:
        # A bare generic disambiguation term ("stole") is always a fresh
        # topic, not a continuation — any prior pool/shown no longer
        # applies once we've asked the customer to pick a type.
        return FinalizedTurn(
            rendered=rendered, new_pool=[], new_shown=[], new_colors=previous_colors, is_disambiguation=True,
        )

    new_pool = result.pool if result.pool else previous_pool

    if result.shown:
        # A pure carryover follow-up ("which of these has the best
        # reviews?") returns the exact same shown set it was given —
        # nothing new was searched for, so a reply naming just one product
        # is picking a winner, not narrowing the set itself.
        is_pure_carryover = {p["sku"] for p in result.shown} == {p["sku"] for p in previous_shown}
        if is_pure_carryover:
            mentioned = _reorder_by_mention(rendered, result.shown)
        else:
            mentioned = _mentioned_products(rendered, result.shown)
        return FinalizedTurn(
            rendered=rendered, new_pool=new_pool, new_shown=mentioned, new_colors=new_colors, is_disambiguation=False,
        )

    return FinalizedTurn(
        rendered=rendered, new_pool=new_pool, new_shown=previous_shown, new_colors=previous_colors, is_disambiguation=False,
    )
