"""Run the fine-tuned model (served on the pod via vLLM) over the test suite.

For each test question, retrieves products via the same RAG path the Gradio
app uses, generates an answer from the pod-hosted model, and records it
alongside the question's reference answer for judging.

Usage:
    python -m tests.run_eval [--model mistralai/Mistral-Small-3.1-24B-Instruct-2503] [--limit 50]

Writes tests/results/raw_answers.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.llm_backend import PodChatModel  # noqa: E402
from catalog.product_format import products_to_context  # noqa: E402
from catalog.structured_client import with_retries  # noqa: E402
from config import BASE_MODEL_ID, RAW_ANSWERS_PATH, RETRIEVAL_TOP_K, SYSTEM_PROMPT_TEMPLATE, TEST_QUESTIONS_PATH  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL_ID, help="Model name as served by vLLM on the pod")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL (reach via SSH tunnel)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick smoke tests)")
    parser.add_argument("--questions-file", type=Path, default=TEST_QUESTIONS_PATH, help="Override input questions file (e.g. a subset for re-testing prior failures)")
    parser.add_argument("--output", type=Path, default=RAW_ANSWERS_PATH, help="Override output path")
    args = parser.parse_args()

    if not args.questions_file.exists():
        raise SystemExit(f"No test questions found at {args.questions_file}. Run tests/generate_test_questions.py first.")

    questions = [json.loads(line) for line in args.questions_file.open()]
    if args.limit:
        questions = questions[: args.limit]

    retriever = ProductRetriever()
    # (connect, read) tuple, not a single float — a single float resets on every
    # socket read, so a connection that's ESTABLISHED but stalled mid-response
    # (observed: SSH-tunnel hang with zero bytes trickling) never times out.
    model = PodChatModel(model=args.model, base_url=args.base_url, timeout=(10, 60))

    print(f"Running {len(questions)} questions through the pod-hosted model...")
    results = []
    skipped = []
    for i, q in enumerate(questions, start=1):
        retrieved = retriever.retrieve(q["question"], top_k=RETRIEVAL_TOP_K)
        context = products_to_context(retrieved)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        try:
            answer = with_retries(model.chat, system_prompt, q["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Q{q['id']} failed after retries, skipping: {exc}")
            skipped.append(q["id"])
            continue

        results.append({
            **q,
            "retrieved_skus": [p["sku"] for p in retrieved],
            "model_answer": answer,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(questions)} done")

    with args.output.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(results)} answers to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} questions after retries failed: {skipped}")
    print("Next: python -m tests.judge")


if __name__ == "__main__":
    main()
