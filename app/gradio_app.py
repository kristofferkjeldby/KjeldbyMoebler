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
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
  #case-input-panel { display: flex !important; gap: 6px !important; align-items: center !important;
    align-self: flex-end !important; margin: 0 0 4px !important; }
  #case-input-panel input { flex: 1 !important; min-width: 180px !important; padding: 4px 8px !important;
    font-size: 0.8rem !important; border: 1px solid #ccc !important; border-radius: 6px !important; }
  #case-input-panel button { font-size: 0.8rem !important; padding: 4px 10px !important; cursor: pointer !important; }

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
  // category-table modal the landing page's category cards use. The link
  // carries the exact pool (every SKU behind that count, however it was
  // filtered — color, price, dimension, store...) as a ?skus= param so the
  // modal opens showing precisely what the link's count promised, instead
  // of the full unfiltered category.
  var catA = e.target.closest('a[href*="kjeldbymobler.dk/kategori/"]');
  if (catA) {
    e.preventDefault();
    var rest = catA.getAttribute('href').split('/kategori/')[1];
    var parts = rest.split('?');
    var category = parts[0];
    var skus = null;
    if (parts[1]) {
      var skusParam = new URLSearchParams(parts[1]).get('skus');
      if (skusParam) {
        try { skus = JSON.parse(skusParam); } catch (err) {}
      }
    }
    window.parent.postMessage({type: 'kjeldby-open-category', category: category, skus: skus}, '*');
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

// "Gem som test-case": click reveals an inline "Forventet resultat" input
// next to the button; submitting POSTs {expected_result} to /api/cases
// (see build_app in app/gradio_app.py), which snapshots the current
// conversation + retrieval state (conversation_state) and writes it to
// cases/unresolved/ — turning a bug found during manual testing straight
// into a durable regression case, replayed later by tests/run_case_tests.py
// once fixed and moved to cases/resolved/. Built via plain DOM manipulation
// rather than a new Gradio component, same as the button's previous
// clipboard-only behavior was.
document.addEventListener('click', function (e) {
  var btn = e.target.closest('#copy-log-btn, #copy-log-btn button');
  if (btn) {
    if (document.getElementById('case-input-panel')) return;
    var label = btn.tagName === 'BUTTON' ? btn : btn.querySelector('button') || btn;
    var original = label.textContent;
    label.style.display = 'none';

    var panel = document.createElement('div');
    panel.id = 'case-input-panel';
    panel.innerHTML =
      '<input type="text" id="case-input" placeholder="Forventet resultat...">' +
      '<button type="button" id="case-save-btn">Gem</button>' +
      '<button type="button" id="case-cancel-btn">✕</button>';
    label.parentElement.appendChild(panel);
    var input = panel.querySelector('#case-input');
    input.focus();

    function cleanup() {
      panel.remove();
      label.style.display = '';
    }

    function flash(text) {
      label.textContent = text;
      setTimeout(function () { label.textContent = original; }, 1500);
    }

    function submitCase() {
      var expected = input.value.trim();
      if (!expected) { input.focus(); return; }
      fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_result: expected }),
      }).then(function (r) {
        if (!r.ok) throw new Error('save failed');
        cleanup();
        flash('✓ Gemt som sag');
      }).catch(function () {
        cleanup();
        flash('⚠ Kunne ikke gemme');
      });
    }

    panel.querySelector('#case-save-btn').addEventListener('click', submitCase);
    panel.querySelector('#case-cancel-btn').addEventListener('click', cleanup);
    input.addEventListener('keydown', function (e2) {
      if (e2.key === 'Enter') submitCase();
      if (e2.key === 'Escape') cleanup();
    });
    return;
  }
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


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.conversation import (  # noqa: E402
    build_prompt,
    finalize_turn,
    focus_payload,
    _safe_stream_prefix,
)
from app.llm_backend import PodChatModel  # noqa: E402
from config import BASE_MODEL_ID, CASES_UNRESOLVED_DIR, RETRIEVAL_TOP_K  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402


def build_app(app: FastAPI, model_name: str, base_url: str) -> gr.Blocks:
    retriever = ProductRetriever()
    model = PodChatModel(model=model_name, base_url=base_url)

    # search.html (served statically from the landing site on :8080) calls
    # back into these routes on :7860 — a genuine cross-origin request from
    # the browser's point of view, so it needs CORS, unlike the chat iframe
    # (which talks to its parent via postMessage, not fetch).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/autocomplete")
    def api_autocomplete(q: str = "") -> list[dict]:
        q_norm = q.strip().lower()
        if not q_norm:
            return []
        matches = []
        for p in retriever.products:
            name_match = q_norm in p["name"].lower() or q_norm in p["sku"].lower()
            # A bare color term ("grå") on its own rarely appears in a
            # product's name, so matching only name/SKU left autocomplete
            # blind to color queries even though /api/search's underlying
            # retriever already handles them fine — check the product's own
            # colors too so a color-only query surfaces real suggestions.
            matched_color = next((c for c in p.get("colors", []) if q_norm in c.lower()), None)
            if name_match or matched_color:
                matches.append({
                    "sku": p["sku"], "name": p["name"], "category": p["category"],
                    "matched_color": matched_color,
                })
                if len(matches) >= 10:
                    break
        return matches

    @app.get("/api/search")
    def api_search(q: str = "") -> dict:
        # Stateless single-shot query, no `focus` pool from a prior turn —
        # this endpoint exists to inspect what the retriever does with one
        # query in isolation, the same thing every chat turn starts from.
        result = retriever.retrieve(q, top_k=RETRIEVAL_TOP_K)
        # Surfaced so the results page can show/open products in the color
        # the query actually asked for ("hjørnesofaer i rød") rather than
        # each product's arbitrary default photo — same detection the chat
        # already uses to build its own color-filtered "see all" links.
        detected_colors = sorted(retriever.detect_colors(q))
        return {
            "shown": [p["sku"] for p in result.shown],
            "pool": [p["sku"] for p in result.pool],
            "total_count": result.total_count,
            "category_breakdown": result.category_breakdown,
            "detected_colors": detected_colors,
        }

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

    # Plain-text conversation + focus-state log, one block per turn, for the
    # "Gem som test-case" button below — lets manual testing read the whole
    # exchange (including what was actually shown/pooled, not just what's
    # visible in the chat bubbles) at a glance.
    chat_log_state: dict[str, list[str]] = {"turns": []}

    # Structured, replayable counterpart to chat_log_state: `messages` is
    # exactly the {"role", "content"} shape model.chat()/chat_stream()
    # expect as `history`, so a saved case (see /api/cases below) can be
    # fed straight back through build_prompt/model.chat/finalize_turn by
    # tests/run_case_tests.py without needing to reparse the plain-text
    # log. `turns` carries the retrieval outcome per turn (not derivable
    # from the reply text alone) for the same replay/judging use.
    conversation_state: dict[str, list] = {"messages": [], "turns": []}

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
        conversation_state["messages"].append({"role": "user", "content": message})
        conversation_state["messages"].append({"role": "assistant", "content": rendered})
        conversation_state["turns"].append({
            "shown_skus": [p["sku"] for p in result.shown],
            "pool_size": len(result.pool),
            "total_count": result.total_count,
            "category_breakdown": result.category_breakdown,
            "colors": sorted(new_colors),
        })

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

        prompt = build_prompt(retriever, message, previous_pool)
        previous_shown_json = focus_payload(previous_shown, previous_colors)

        # Linkifying/bulletizing during streaming caused a visible flip once
        # a product name or SKU completed mid-token — it showed as plain
        # text while partial, then suddenly re-rendered as a link. Streaming
        # the raw text avoids that; the fully-formatted version is swapped
        # in as a single atomic update once the reply is done (see
        # finalize_turn). Separately, _safe_stream_prefix holds back any
        # markdown link the model is mid-way through writing itself, so its
        # raw "[...](https://..." syntax is never shown broken — only once
        # complete, or once the full reply lands below.
        partial = ""
        for token in model.chat_stream(prompt.system_prompt, message, history=history):
            partial += token
            yield _safe_stream_prefix(partial), previous_shown_json, previous_log_text

        turn = finalize_turn(partial, prompt, previous_pool, previous_shown, previous_colors)
        yield turn.rendered, previous_shown_json, previous_log_text

        pool_state["products"] = turn.new_pool
        shown_state["products"] = turn.new_shown
        color_state["colors"] = turn.new_colors

        if turn.is_disambiguation:
            # A bare generic disambiguation term ("stole") is always a
            # fresh topic, not a continuation — any prior pool/shown no
            # longer applies once we've asked the customer to pick a type.
            _log_turn(len(chat_log_state["turns"]) + 1, message, turn.rendered, prompt.result, previous_colors)
            yield turn.rendered, focus_payload([], previous_colors), _chat_log_text()
            return

        _log_turn(len(chat_log_state["turns"]) + 1, message, turn.rendered, prompt.result, turn.new_colors)
        yield turn.rendered, focus_payload(turn.new_shown, turn.new_colors), _chat_log_text()

    def reset_focus() -> tuple[str, str]:
        pool_state["products"] = []
        shown_state["products"] = []
        color_state["colors"] = set()
        chat_log_state["turns"] = []
        conversation_state["messages"] = []
        conversation_state["turns"] = []
        return focus_payload([], set()), _chat_log_text()

    # "Gem som test-case" (client-side, see PRODUCT_LINK_SCRIPT) POSTs here
    # once the tester has typed what they expected — snapshots the current
    # conversation + retrieval state into cases/unresolved/ for later
    # regression testing (tests/run_case_tests.py) once the bug it captured
    # gets fixed. Same-origin call (this route and the iframe that calls it
    # both live on :7860), unlike /api/search — no CORS needed here.
    @app.post("/api/cases")
    def save_case(payload: dict) -> dict:
        expected_result = (payload.get("expected_result") or "").strip()
        case_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        case = {
            "id": case_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "expected_result": expected_result,
            "chat_log_text": _chat_log_text(),
            "messages": conversation_state["messages"],
            "turns": conversation_state["turns"],
        }
        path = CASES_UNRESOLVED_DIR / f"{case_id}.json"
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False))
        return {"ok": True, "path": str(path)}

    with gr.Blocks(title="Kjeldby Møbler", fill_width=True, fill_height=True) as demo:
        focus_data = gr.Textbox(elem_id="focus-data", show_label=False, container=False)
        # Hidden carrier for the plain-text chat+focus log — shown to the
        # tester (read client-side by PRODUCT_LINK_SCRIPT) alongside the
        # "Gem som test-case" button below, via the same CSS-hide + JS-poll
        # relay PRODUCT_LINK_SCRIPT already uses for focus-data (see its
        # comment for why polling, not visible=False).
        chat_log_data = gr.Textbox(elem_id="chat-log-data", show_label=False, container=False)
        # Not wired to any server-side handler — PRODUCT_LINK_SCRIPT below
        # handles its click entirely client-side (reveal an inline
        # "Forventet resultat" input, then POST to /api/cases on submit).
        gr.Button("🐞 Gem som test-case", elem_id="copy-log-btn", size="sm")
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
    args = parser.parse_args()

    # Built as a plain FastAPI app with the Gradio UI mounted into it
    # (rather than demo.launch()) so /api/autocomplete and /api/search
    # (see build_app) can be served from the same process — reusing the
    # already-loaded retriever (embedding model + cross-encoder + FAISS
    # index) instead of a second process paying to load it again.
    app = FastAPI()
    demo = build_app(app, args.model, args.base_url)
    app = gr.mount_gradio_app(app, demo, path="/", head=PRODUCT_LINK_SCRIPT, footer_links=[])
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
