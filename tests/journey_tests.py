"""Manual multi-turn journey smoke tests for the RAG focus + conversation
history feature (rag/retriever.py's `focus` param + app/llm_backend.py's
`history` param). Not part of the automated single-turn eval suite — this
runs a handful of realistic multi-turn conversations end-to-end against the
live retriever and pod model, and prints the full transcript for manual
review, covering the three journeys the assistant needs to support:

  1. Discovery            — customer doesn't know what product they want yet
  2. Product information  — customer has a product in mind, asks follow-ups
  3. Availability & price — customer is ready to buy, checks stock/cost

Usage:
    python -m tests.journey_tests [--model ...] [--base-url http://localhost:8000/v1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.llm_backend import PodChatModel  # noqa: E402
from catalog.product_format import category_breakdown_to_context, products_to_context  # noqa: E402
from config import BASE_MODEL_ID, CATALOG_PATH, RETRIEVAL_TOP_K, SYSTEM_PROMPT_TEMPLATE  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402


def _pick(products: list[dict], category: str) -> dict:
    candidates = [p for p in products if p["category"] == category]
    return candidates[0] if candidates else products[0]


def build_journeys(products: list[dict]) -> dict[str, list[str]]:
    table = _pick(products, "dining_table")
    dresser = _pick(products, "dresser")
    return {
        "discovery": [
            "Jeg skal indrette mit hjemmekontor og vil gerne have noget moderne og stilrent. Hvad foreslår du?",
            "Fortæl lidt mere om den første mulighed du nævnte.",
            "Hvilke farver fås den i?",
            "Er den svær at samle selv?",
        ],
        "product_info": [
            f"Hvad er målene på '{table['name']}'?",
            "Hvilket materiale er det lavet af?",
            "Hvor mange personer kan sidde ved det?",
            "Kræver det samling, og hvor lang er garantien?",
        ],
        "availability_price": [
            f"Har I '{dresser['name']}' på lager?",
            "Hvad koster den?",
            "Er den på lager i Odense lige nu?",
            "Er der rabat på den for tiden?",
        ],
    }


def run_journey(name: str, questions: list[str], retriever: ProductRetriever, model: PodChatModel) -> None:
    print(f"\n{'=' * 70}\nJOURNEY: {name}\n{'=' * 70}")
    history: list[dict] = []
    focus: list[dict] = []
    for i, question in enumerate(questions, start=1):
        result = retriever.retrieve(question, top_k=RETRIEVAL_TOP_K, focus=focus)
        if result.pool:
            focus = result.pool  # full matching set, not just what's shown — see RetrievalResult
        context = (
            category_breakdown_to_context(result.category_breakdown)
            if result.category_breakdown
            else products_to_context(result.shown, result.total_count)
        )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        answer = model.chat(system_prompt, question, history=history)

        print(f"\n[{i}] Kunde: {question}")
        print(f"    Vist ({len(result.shown)}/{result.total_count}): {[p['name'] for p in result.shown][:4]}")
        print(f"    Svar: {answer}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL_ID, help="Model name as served by vLLM on the pod")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL (reach via SSH tunnel)")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        raise SystemExit(f"No catalog found at {CATALOG_PATH}. Run catalog/generate_catalog.py first.")
    products = json.loads(CATALOG_PATH.read_text())

    retriever = ProductRetriever()
    model = PodChatModel(model=args.model, base_url=args.base_url)

    for name, questions in build_journeys(products).items():
        run_journey(name, questions, retriever, model)


if __name__ == "__main__":
    main()
