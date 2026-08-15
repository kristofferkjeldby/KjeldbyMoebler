"""Gradio chat UI: retrieves relevant products via RAG, injects them into the
system prompt, and generates a streamed reply from the fine-tuned model
served on the RunPod GPU via vLLM.

Runs on your Mac, but the model itself runs on the pod — reach it through an
SSH tunnel first:
    ssh -f -N -L 8000:localhost:8000 -p <port> -i <key> root@<pod-ip>

Usage:
    python -m app.gradio_app [--model mistralai/Mistral-Small-3.1-24B-Instruct-2503] [--base-url http://localhost:8000/v1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import gradio as gr

# Renders any SKU or retrieved-product name the model mentions as a link
# to a fake product URL. The chat runs in a separate-origin iframe embedded
# in the landing page, so a click can't reach that page's DOM directly — a
# head script (see PRODUCT_LINK_SCRIPT below) intercepts the click and
# posts a message to the parent window, which opens the product modal.
# (A bare "#product-<SKU>" fragment href doesn't render reliably through
# Gradio's markdown pipeline — a normal-looking absolute URL does.)
_SKU_RE = re.compile(r"\bFRN-\d{4}\b")
_PRODUCT_URL = "https://kjeldbymobler.dk/produkt/{sku}"

# The "focus-data" textbox (see build_app) carries the current focus set as
# JSON, refreshed on every reply. It's a normal *visible* component (hidden
# only via our own CSS below) so it always mounts through Gradio's regular
# update path — an actually-invisible component is a less-exercised code
# path and wasn't reliably readable from here. Gradio sets the value as a JS
# property, not an HTML attribute, so a MutationObserver won't see it change
# — polling is the reliable way to notice updates and relay them to the
# parent page (a different origin, so it can't just read this iframe's DOM
# directly).
PRODUCT_LINK_SCRIPT = """
<style>
  #focus-data, #chat-log-data { position: absolute !important; width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; }
  #copy-log-btn { align-self: flex-end !important; margin: 0 0 4px !important; flex: 0 0 auto !important; }

  /* Gradio's fill_height relies on flex-grow all the way down, which only
     resolves if html/body actually have a real height to grow into — by
     default they don't inside an iframe, so without this the chat content
     just shrinks to fit and leaves the rest of the iframe blank. */
  /* overflow:hidden here is what actually matters: without it, the page
     itself (not the inner message list) absorbs the extra height and
     scrolls as a whole — which drags the input box out of view too. */
  html, body { height: 100% !important; margin: 0 !important; overflow: hidden !important; }
  /* .main.fillable had overflow:hidden but no actual fixed height, so it
     just grew past 900px along with everything else and silently clipped
     content with no scrollbar anywhere — verified via direct DOM inspection
     (Playwright), not guessed. It needs a real height to clip *against*. */
  .gradio-container .main.fillable {
    padding: 10px 15px !important;
    overflow: hidden !important;
    height: 900px !important;
    box-sizing: border-box !important;
  }
  /* flex-shrink alone doesn't shrink a flex item below its content size —
     flex-basis defaults to "auto" (content-sized) unless overridden, so
     every level here sized to its content first and never actually
     compressed to fit. Forcing flex-basis:0 (via the shorthand) makes each
     level start from zero and grow only into the space actually available,
     which is what finally lets it get bounded instead of just growing
     forever. Confirmed empirically, not guessed — same for min-height:0. */
  .gradio-container .main.fillable, .gradio-container .main.fillable * {
    min-height: 0 !important;
  }
  /* .main.fillable's direct children include the real chat wrapper plus
     a couple of empty Gradio-internal placeholder divs (toast/error slots)
     — applying flex:1 to *every* direct child let those empty siblings
     claim an equal share of the 900px, starving the actual chat down to
     about a third of the frame. Only the real chat wrapper should grow. */
  .gradio-container .main.fillable > .wrap.svelte-zxu34v {
    flex: 1 1 auto !important;
  }
  .gradio-container .main.fillable > *:not(.wrap.svelte-zxu34v) {
    flex: 0 0 0 !important;
  }
  .gradio-container .main.fillable .wrap.svelte-zxu34v,
  .gradio-container .main.fillable .contain.svelte-zxu34v,
  .gradio-container .main.fillable .column,
  .gradio-container .main.fillable .block.flex,
  .gradio-container .main.fillable .wrapper {
    flex: 1 1 0 !important;
  }
  .bubble-wrap, .message-wrap { overflow-y: auto !important; flex: 1 1 0 !important; }
  /* Retry/undo (and edit) are per-message hover icons ChatInterface wires
     up unconditionally in this Gradio version — no Python-level flag
     disables them, so they're hidden via their icon-row wrapper instead. */
  .message-buttons-left, .message-buttons-right { display: none !important; }
  .example {
    background: #e4dbcd !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
  }
  .gradio-container, .message, .bubble {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif !important;
  }
</style>
<script>
document.addEventListener('click', function (e) {
  var a = e.target.closest('a[href*="kjeldbymobler.dk/produkt/"]');
  if (a) {
    e.preventDefault();
    var sku = a.getAttribute('href').split('/produkt/')[1];
    window.parent.postMessage({type: 'kjeldby-open-product', sku: sku}, '*');
    return;
  }
  // "Se alle N <kategori> ->" link appended after a reply that only showed
  // the top SHOWN_MAX_RESULTS of a larger match — opens the same
  // category-table modal the landing page's category cards use. When the
  // shown set was narrowed by color (e.g. "10 red office chairs"), the
  // link carries those exact colors as a ?farver= param so the modal opens
  // pre-filtered to the same count the link's label promised, instead of
  // the full unfiltered category.
  var catA = e.target.closest('a[href*="kjeldbymobler.dk/kategori/"]');
  if (catA) {
    e.preventDefault();
    var rest = catA.getAttribute('href').split('/kategori/')[1];
    var parts = rest.split('?');
    var category = parts[0];
    var colors = null;
    if (parts[1]) {
      var farverParam = new URLSearchParams(parts[1]).get('farver');
      if (farverParam) {
        try { colors = JSON.parse(farverParam); } catch (err) {}
      }
    }
    window.parent.postMessage({type: 'kjeldby-open-category', category: category, colors: colors}, '*');
  }
});

(function () {
  var lastValue = null;
  setInterval(function () {
    var el = document.querySelector('#focus-data textarea, #focus-data input');
    if (!el || el.value === lastValue) return;
    lastValue = el.value;
    try {
      var products = JSON.parse(el.value || '[]');
      window.parent.postMessage({type: 'kjeldby-focus-update', products: products}, '*');
    } catch (err) {}
  }, 600);
})();

// "Kopiér chat + fokus-log": reads the hidden #chat-log-data textbox (kept
// current by respond()'s third yielded value, one block per turn) and
// writes it straight to the clipboard — for pasting the whole exchange,
// including what was actually shown/pooled at each step, somewhere else
// for manual review. Requires the parent page's <iframe> to grant
// clipboard-write (see app/landing_page.html) since this runs in a
// cross-origin child frame.
document.addEventListener('click', function (e) {
  var btn = e.target.closest('#copy-log-btn, #copy-log-btn button');
  if (!btn) return;
  var el = document.querySelector('#chat-log-data textarea, #chat-log-data input');
  var text = el ? el.value : '';
  var label = btn.tagName === 'BUTTON' ? btn : btn.querySelector('button') || btn;
  var original = label.textContent;
  navigator.clipboard.writeText(text || '(ingen beskeder endnu)').then(function () {
    label.textContent = '✓ Kopieret!';
    setTimeout(function () { label.textContent = original; }, 1500);
  }).catch(function () {
    label.textContent = '⚠ Kunne ikke kopiere';
    setTimeout(function () { label.textContent = original; }, 1500);
  });
});

// Gradio's own autoscroll doesn't reliably keep up while streaming inside
// this iframe layout, so the message list is force-scrolled to the bottom
// directly instead of relying on it — but only while the customer is
// already near the bottom (i.e. actively following the reply). Without
// that check this fights any attempt to scroll up and read earlier
// messages, snapping back down every tick regardless of intent.
(function () {
  var NEAR_BOTTOM_PX = 80;
  setInterval(function () {
    document.querySelectorAll('.bubble-wrap, .message-wrap').forEach(function (el) {
      var distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (distanceFromBottom < NEAR_BOTTOM_PX) {
        el.scrollTop = el.scrollHeight;
      }
    });
  }, 200);
})();
</script>
"""


# The model occasionally generates a complete, well-formed product link on
# its own (having presumably picked up the "[Name](https://.../produkt/SKU)"
# shape from context) — without protecting those first, the SKU substitution
# below matches the SKU sitting inside that URL, and the name substitution
# matches the name sitting inside that link text, each wrapping it AGAIN
# and producing garbled nested links like "[[Name](url)](url/[SKU](url))".
_EXISTING_LINK_RE = re.compile(r"\[[^\]]+\]\(https://kjeldbymobler\.dk/produkt/[^\)]+\)")


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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.llm_backend import PodChatModel  # noqa: E402
from catalog.product_format import (  # noqa: E402
    CATEGORY_LABELS,
    CATEGORY_LABELS_PLURAL,
    category_breakdown_to_context,
    products_to_context,
)
from config import BASE_MODEL_ID, CATEGORY_URL_BASE, RETRIEVAL_TOP_K, SYSTEM_PROMPT_TEMPLATE  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402


def build_app(model_name: str, base_url: str) -> gr.Blocks:
    retriever = ProductRetriever()
    model = PodChatModel(model=model_name, base_url=base_url)

    # Single-user local POC, so plain closure variables are enough to track
    # conversation state across turns — a multi-session deployment would
    # need these in per-session gr.State instead.
    #
    # Two separate pieces of state, deliberately not one:
    #   - pool_state: the FULL matching set for the current topic (can be
    #     dozens of products — "all 67 sectional sofas"), passed back into
    #     retriever.retrieve() as `focus` so a follow-up filter ("i grå?")
    #     narrows against everything that matched, not just what was shown.
    #     It only narrows on an explicit new filter — never on which subset
    #     of `shown_state` the model's reply happened to mention, so a
    #     vaguely-worded reply can't silently shrink the pool a later,
    #     unrelated filter would otherwise have searched against.
    #   - shown_state: the <= SHOWN_MAX_RESULTS products actually shown to
    #     the model / displayed in the focus panel this turn — purely a
    #     display concern, reordered (not filtered) by what the reply
    #     actually mentioned.
    pool_state: dict[str, list[dict]] = {"products": []}
    shown_state: dict[str, list[dict]] = {"products": []}
    # The customer may name a color before, after, or never relative to
    # naming a product ("har I dem i mørkegrå" after already seeing some
    # sofas). Rather than track "is this color valid right now", each
    # product is just checked against its own real `colors` list whenever
    # the panel is built — a stale remembered color simply won't match a
    # product that doesn't come in it, so it silently stops showing without
    # needing separate logic to detect the topic having moved on. Matching
    # is case-insensitive because the catalog itself isn't consistent about
    # casing for the same color across products ("Mørkegrå" vs "mørkegrå"),
    # and the product's own exact string is what gets displayed either way.
    color_state: dict[str, set[str]] = {"colors": set()}

    def _focus_payload(products: list[dict], detected_colors_lower: set[str]) -> str:
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

    # Plain-text conversation + focus-state log, one block per turn, for the
    # "Kopiér chat + fokus-log" button below — lets manual testing paste the
    # whole exchange (including what was actually shown/pooled, not just
    # what's visible in the chat bubbles) somewhere else for review.
    chat_log_state: dict[str, list[str]] = {"turns": []}

    def _log_turn(turn_num: int, message: str, rendered: str, result, new_colors: set[str]) -> None:
        colors_line = ", ".join(sorted(new_colors)) or "(ingen)"
        if result.category_breakdown:
            breakdown_lines = "\n".join(
                f"    - {cat}: {count}"
                for cat, count in sorted(result.category_breakdown.items(), key=lambda item: -item[1])
            )
            focus_block = (
                f"  Kategori-opdeling ({result.total_count} i alt på tværs af "
                f"{len(result.category_breakdown)} kategorier):\n{breakdown_lines}\n"
            )
        else:
            shown_lines = "\n".join(f"    - {p['sku']}  {p['name']}" for p in result.shown) or "    (ingen)"
            focus_block = (
                f"  Vist ({len(result.shown)} af {result.total_count} matches):\n"
                f"{shown_lines}\n"
                f"  Pool-størrelse (til næste turs indsnævring): {len(result.pool)}\n"
            )
        chat_log_state["turns"].append(
            f"=== Tur {turn_num} ===\n"
            f"Kunde: {message}\n"
            f"Assistent: {rendered}\n"
            f"\n"
            f"Fokus efter denne tur:\n"
            f"{focus_block}"
            f"  Aktive farver: {colors_line}\n"
        )

    def _chat_log_text() -> str:
        return "\n".join(chat_log_state["turns"])

    def respond(message: str, history: list[dict]):
        # Keep showing the *previous* turn's shown set while this reply
        # streams — updating it immediately (before the model has said
        # anything) made the panel flash a larger list that then shrank
        # once narrowed to what was actually mentioned. It only changes
        # once, at the end, straight to its final value.
        previous_pool = pool_state["products"]
        previous_shown = shown_state["products"]
        previous_colors = color_state["colors"]
        previous_log_text = _chat_log_text()
        result = retriever.retrieve(message, top_k=RETRIEVAL_TOP_K, focus=previous_pool)
        # Computed once up front (not just when narrowing focus at the end)
        # so the "see all" link below can carry the same active color
        # filter into the category modal — otherwise "Se alle 10 røde
        # kontorstole" would open the modal to all 75 unfiltered, a mismatch
        # between what the link promises and what clicking it shows.
        detected_colors = retriever.detect_colors(message)
        new_colors = {c.lower() for c in detected_colors} if detected_colors else previous_colors
        context = (
            category_breakdown_to_context(result.category_breakdown)
            if result.category_breakdown
            else products_to_context(result.shown, result.total_count)
        )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        previous_shown_json = _focus_payload(previous_shown, previous_colors)

        # Linkifying/bulletizing during streaming caused a visible flip once
        # a product name or SKU completed mid-token — it showed as plain
        # text while partial, then suddenly re-rendered as a link. Streaming
        # the raw text avoids that; the fully-formatted version is swapped
        # in as a single atomic update once the reply is done.
        partial = ""
        for token in model.chat_stream(system_prompt, message, history=history):
            partial += token
            yield partial, previous_shown_json, previous_log_text

        rendered = _bulletize_enumerations(_linkify(partial, result.shown))

        # More matched than were shown: append a deterministic link to the
        # category-table modal rather than trusting the (un-fine-tuned)
        # model to construct the URL itself — same reliability reasoning as
        # why product links are code-appended via _linkify rather than
        # generated by the model.
        if result.category_breakdown:
            # One link per category, largest first, so the customer can
            # jump straight to whichever type they actually meant.
            for category, count in sorted(result.category_breakdown.items(), key=lambda item: -item[1]):
                label = CATEGORY_LABELS_PLURAL.get(category, category)
                url = CATEGORY_URL_BASE.format(category=category)
                rendered += f"\n\n[Se alle {count} {label} →]({url})"
        elif result.total_count > len(result.shown) and result.shown:
            category = result.shown[0]["category"]
            label = CATEGORY_LABELS.get(category, category)
            see_all_url = CATEGORY_URL_BASE.format(category=category)
            # Only attach a color filter freshly named THIS turn, not one
            # merely carried over from a prior turn (new_colors falls back
            # to previous_colors when nothing new was detected) — a stale
            # color from an earlier, unrelated topic could otherwise get
            # attached to a category it was never actually filtered by.
            if detected_colors:
                see_all_url += "?farver=" + quote(json.dumps(sorted(detected_colors)))
            rendered += f"\n\n[Se alle {result.total_count} {label} →]({see_all_url})"

        yield rendered, previous_shown_json, previous_log_text

        if result.category_breakdown:
            # A bare generic disambiguation term ("stole") is always a
            # fresh topic, not a continuation — any prior pool/shown no
            # longer applies once we've asked the customer to pick a type.
            pool_state["products"] = []
            shown_state["products"] = []
            _log_turn(len(chat_log_state["turns"]) + 1, message, rendered, result, previous_colors)
            yield rendered, _focus_payload([], previous_colors), _chat_log_text()
            return

        if result.pool:
            pool_state["products"] = result.pool

        if result.shown:
            # A pure carryover follow-up ("which of these has the best
            # reviews?") returns the exact same shown set it was given —
            # nothing new was searched for, so a reply naming just one
            # product is picking a winner, not narrowing the set itself.
            # This only reorders the small displayed set for the focus
            # panel — it never affects pool_state (see the comment above).
            is_pure_carryover = {p["sku"] for p in result.shown} == {p["sku"] for p in previous_shown}
            if is_pure_carryover:
                mentioned = _reorder_by_mention(rendered, result.shown)
            else:
                mentioned = _mentioned_products(rendered, result.shown)
            shown_state["products"] = mentioned
            color_state["colors"] = new_colors
            _log_turn(len(chat_log_state["turns"]) + 1, message, rendered, result, new_colors)
            yield rendered, _focus_payload(mentioned, new_colors), _chat_log_text()
        else:
            _log_turn(len(chat_log_state["turns"]) + 1, message, rendered, result, previous_colors)
            yield rendered, previous_shown_json, _chat_log_text()

    def reset_focus() -> tuple[str, str]:
        pool_state["products"] = []
        shown_state["products"] = []
        color_state["colors"] = set()
        chat_log_state["turns"] = []
        return _focus_payload([], set()), _chat_log_text()

    with gr.Blocks(title="Kjeldby Møbler", fill_width=True, fill_height=True) as demo:
        focus_data = gr.Textbox(elem_id="focus-data", show_label=False, container=False)
        # Hidden carrier for the plain-text chat+focus log — read and copied
        # to the clipboard by the "Kopiér chat + fokus-log" button below, via
        # the same CSS-hide + JS-poll relay PRODUCT_LINK_SCRIPT already uses
        # for focus-data (see its comment for why polling, not visible=False).
        chat_log_data = gr.Textbox(elem_id="chat-log-data", show_label=False, container=False)
        # Not wired to any server-side handler — PRODUCT_LINK_SCRIPT below
        # handles its click entirely client-side (read #chat-log-data,
        # write to clipboard).
        gr.Button("📋 Kopiér chat + fokus-log", elem_id="copy-log-btn", size="sm")
        chat = gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height="100%", buttons=[], feedback_options=None),
            additional_outputs=[focus_data, chat_log_data],
            examples=[
                "Hvilke farver fås sofabordet Skandinavisk Harmoni Coffee Table i?",
                "Er kommoden FRN-0021 på lager, og hvor lang er garantien?",
                "Følger der en madras med Skandinavisk Elegance Sengestativ?",
            ],
        )
        # The trash/clear icon on the chatbot only clears the visible
        # conversation by default — pool_state/shown_state/color_state are
        # separate server-side variables that need their own reset, and the frontend
        # focus panel only follows the "focus-data" textbox, so that has to
        # be pushed back to empty too for the panel to actually clear.
        chat.chatbot.clear(fn=reset_focus, outputs=[focus_data, chat_log_data])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL_ID, help="Model name as served by vLLM on the pod")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL (reach via SSH tunnel)")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app(args.model, args.base_url)
    demo.launch(share=args.share, footer_links=[], head=PRODUCT_LINK_SCRIPT)


if __name__ == "__main__":
    main()
