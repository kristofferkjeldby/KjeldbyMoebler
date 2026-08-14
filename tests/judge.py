"""Judge the fine-tuned model's answers against ground truth, using Claude Sonnet.

For each (question, retrieved context, reference answer, model answer) tuple,
asks Claude Sonnet to score the model's answer on a 1-5 scale plus a
pass/fail verdict, checking specifically for:
  - factual accuracy against the reference answer / retrieved product data
  - hallucination (facts not present in the retrieved context)
  - appropriate refusal when the question is genuinely unanswerable
  - completeness / directness

Usage:
    python -m tests.judge

Writes tests/results/judged_results.jsonl and tests/results/report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.product_format import products_to_context  # noqa: E402
from catalog.structured_client import with_retries  # noqa: E402
from config import JUDGE_MODEL, JUDGED_RESULTS_PATH, RAW_ANSWERS_PATH, REPORT_PATH  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402

MAX_WORKERS = 8

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1 (very bad) to 5 (excellent)"},
        "passed": {"type": "boolean", "description": "true if the answer is factually correct, grounded, and appropriately handles unanswerable questions"},
        "hallucinated": {"type": "boolean", "description": "true if the answer states a fact not present in the retrieved context"},
        "reasoning": {"type": "string", "description": "One or two sentences explaining the score"},
    },
    "required": ["score", "passed", "hallucinated", "reasoning"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are grading a furniture-store chatbot's answer for accuracy and groundedness.

Question type: {qtype}
Customer question: {question}

Retrieved product context the chatbot was given:
{context}

Reference (known-correct) answer: {reference_answer}

Chatbot's actual answer: {model_answer}

Grade the chatbot's answer:
- Does it match the facts in the reference answer and retrieved context (price, dimensions, colors, stock, etc.)?
- Does it invent any fact not present in the retrieved context (hallucination)? This is a serious failure even if the invented fact sounds plausible.
- If the question type is "unanswerable", does the chatbot correctly decline rather than guessing or making something up?
- If the question type is "enumeration", does it list EVERY matching product from the context with none omitted and none invented (a partial list is a failure, not partial credit)?
- If the question type is "store_stock", does it correctly identify which products have enough stock at the specific store named (not just "in stock somewhere"), and correctly say so if none qualify?
- If the question type is "dimension", does it correctly identify products matching the requested measurement (or correctly say none do)?
- If the question type is "series", does it name the other matching products in the series without omissions and without including the anchor product itself?
- Is it a direct, complete, helpful answer to what was actually asked?

Score 1-5 and give a pass/fail verdict. A hallucination should generally fail regardless of tone."""


def judge_one(client: anthropic.Anthropic, retriever: ProductRetriever, record: dict) -> dict:
    products = [retriever.get_by_sku(sku) for sku in record["retrieved_skus"]]
    products = [p for p in products if p is not None]
    context = products_to_context(products)

    prompt = JUDGE_PROMPT.format(
        qtype=record["type"],
        question=record["question"],
        context=context,
        reference_answer=record["reference_answer"],
        model_answer=record["model_answer"],
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=800,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    verdict = json.loads(text)
    return {**record, "judge": verdict}


def build_report(judged: list[dict]) -> str:
    lines = ["# Evaluation Report\n"]
    n = len(judged)
    n_passed = sum(1 for r in judged if r["judge"]["passed"])
    n_hallucinated = sum(1 for r in judged if r["judge"]["hallucinated"])
    avg_score = sum(r["judge"]["score"] for r in judged) / n if n else 0

    lines.append(f"**Total questions:** {n}")
    lines.append(f"**Pass rate:** {n_passed}/{n} ({100 * n_passed / n:.1f}%)")
    lines.append(f"**Hallucination rate:** {n_hallucinated}/{n} ({100 * n_hallucinated / n:.1f}%)")
    lines.append(f"**Average score (1-5):** {avg_score:.2f}\n")

    lines.append("## By question type\n")
    lines.append("| Type | Count | Pass rate | Hallucination rate | Avg score |")
    lines.append("|---|---|---|---|---|")
    for qtype in sorted({r["type"] for r in judged}):
        subset = [r for r in judged if r["type"] == qtype]
        m = len(subset)
        p = sum(1 for r in subset if r["judge"]["passed"])
        h = sum(1 for r in subset if r["judge"]["hallucinated"])
        s = sum(r["judge"]["score"] for r in subset) / m
        lines.append(f"| {qtype} | {m} | {100*p/m:.1f}% | {100*h/m:.1f}% | {s:.2f} |")

    failures = [r for r in judged if not r["judge"]["passed"]][:20]
    if failures:
        lines.append("\n## Sample failures (first 20)\n")
        for r in failures:
            lines.append(f"### Q{r['id']} ({r['type']})")
            lines.append(f"- **Question:** {r['question']}")
            lines.append(f"- **Reference:** {r['reference_answer']}")
            lines.append(f"- **Model answer:** {r['model_answer']}")
            lines.append(f"- **Judge:** {r['judge']['reasoning']}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=Path, default=RAW_ANSWERS_PATH, help="Override input raw answers file")
    parser.add_argument("--judged-output", type=Path, default=JUDGED_RESULTS_PATH, help="Override judged results output path")
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH, help="Override report output path")
    args = parser.parse_args()

    if not args.answers_file.exists():
        raise SystemExit(f"No raw answers found at {args.answers_file}. Run tests/run_eval.py first.")

    records = [json.loads(line) for line in args.answers_file.open()]
    client = anthropic.Anthropic()
    retriever = ProductRetriever()

    print(f"Judging {len(records)} answers with {JUDGE_MODEL}...")
    judged = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(with_retries, judge_one, client, retriever, r): r["id"] for r in records}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                judged.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! judging failed: {exc}")
            if i % 50 == 0:
                print(f"  {i}/{len(records)} judged")

    if not judged:
        raise SystemExit(
            f"All {len(records)} judging calls failed — nothing to report. "
            f"Not overwriting {args.report_output} or {args.judged_output}. Check the errors above (commonly a missing/expired ANTHROPIC_API_KEY)."
        )

    judged.sort(key=lambda r: r["id"])
    with args.judged_output.open("w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")

    report = build_report(judged)
    args.report_output.write_text(report)

    print(f"\nWrote {len(judged)} judged results to {args.judged_output}")
    print(f"Wrote report to {args.report_output}")
    print("\n" + report.split("## By question type")[0])


if __name__ == "__main__":
    main()
