"""Replay resolved manual-testing cases (cases/resolved/) through the same
retrieval + prompt-building + post-processing pipeline the live chat uses
(app/conversation.py), and judge whether the final turn's outcome now
satisfies the case's freeform `expected_result`, using Claude as judge —
the same LLM-judge pattern as tests/judge.py, but for real bugs a human
found during manual testing (captured via the "Gem som test-case" button
in app/gradio_app.py) instead of synthetic eval questions.

A case starts in cases/unresolved/ when saved; once the bug it describes is
fixed, move the file to cases/resolved/ (a plain `mv`) so this script picks
it up as a standing regression test on every future run.

Usage:
    python -m tests.run_case_tests [--cases-dir cases/resolved]

Needs the fine-tuned model reachable (same PodChatModel/SSH-tunnel setup as
app/gradio_app.py) and ANTHROPIC_API_KEY for judging.

Writes cases/results/results.jsonl and cases/results/report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.conversation import build_prompt, finalize_turn  # noqa: E402
from app.llm_backend import PodChatModel  # noqa: E402
from catalog.structured_client import with_retries  # noqa: E402
from config import BASE_MODEL_ID, CASES_RESOLVED_DIR, CASES_RESULTS_DIR, JUDGE_MODEL  # noqa: E402
from rag.retriever import ProductRetriever  # noqa: E402

# Each case replays a full multi-turn conversation through the model
# sequentially (one non-streaming call per turn) — kept modest so a
# multi-turn case doesn't fan out into a burst of concurrent model calls.
MAX_WORKERS = 4

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean", "description": "true if the actual outcome satisfies what the tester expected"},
        "reasoning": {"type": "string", "description": "One or two sentences explaining the verdict"},
    },
    "required": ["passed", "reasoning"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are checking whether a furniture-store chatbot's behavior now matches what a human tester expected, after replaying a real conversation that previously exposed a bug.

Full conversation replayed just now (customer / assistant, in order):
{transcript}

What the tester expected the final turn's outcome to be:
{expected_result}

What actually happened on the final turn:
- Assistant's reply: {final_reply}
- Products shown/mentioned this turn (SKUs): {final_shown_skus}
- Total matching products this turn: {final_total_count}
- Category breakdown, if the assistant asked the customer to pick a type: {final_category_breakdown}
- Active color filters after this turn: {final_colors}

Does the actual outcome satisfy what the tester expected? Judge the SPECIFIC
issue the tester described, not general answer quality unless that's what
they described. Give a pass/fail verdict."""


def replay_case(retriever: ProductRetriever, model: PodChatModel, case: dict) -> dict:
    """Feeds the case's recorded customer messages back through
    build_prompt/model.chat/finalize_turn turn by turn, rebuilding
    pool/shown/colors state exactly like a live session — the same logic
    app/gradio_app.py's respond() drives, just non-streaming here."""
    messages = case["messages"]
    pool: list[dict] = []
    shown: list[dict] = []
    colors: set[str] = set()
    history: list[dict] = []
    prompt = None
    turn = None

    for i in range(0, len(messages), 2):
        user_msg = messages[i]["content"]
        prompt = build_prompt(retriever, user_msg, pool)
        raw_reply = model.chat(prompt.system_prompt, user_msg, history=history)
        turn = finalize_turn(raw_reply, prompt, pool, shown, colors)
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": turn.rendered},
        ]
        pool, shown, colors = turn.new_pool, turn.new_shown, turn.new_colors

    result = prompt.result
    return {
        "transcript": history,
        "final_reply": turn.rendered,
        "final_shown_skus": [p["sku"] for p in turn.new_shown],
        "final_total_count": result.total_count,
        "final_category_breakdown": result.category_breakdown,
        "final_colors": sorted(turn.new_colors),
    }


def judge_one(client: anthropic.Anthropic, retriever: ProductRetriever, model: PodChatModel, case: dict) -> dict:
    replay = replay_case(retriever, model, case)
    transcript_text = "\n".join(f"{m['role']}: {m['content']}" for m in replay["transcript"])

    prompt = JUDGE_PROMPT.format(
        transcript=transcript_text,
        expected_result=case["expected_result"],
        final_reply=replay["final_reply"],
        final_shown_skus=", ".join(replay["final_shown_skus"]) or "(ingen)",
        final_total_count=replay["final_total_count"],
        final_category_breakdown=replay["final_category_breakdown"] or "(ingen)",
        final_colors=", ".join(replay["final_colors"]) or "(ingen)",
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    verdict = json.loads(text)
    return {**case, "replay": replay, "judge": verdict}


def build_report(judged: list[dict]) -> str:
    lines = ["# Case Regression Report\n"]
    n = len(judged)
    n_passed = sum(1 for r in judged if r["judge"]["passed"])
    lines.append(f"**Total cases:** {n}")
    lines.append(f"**Pass rate:** {n_passed}/{n} ({100 * n_passed / n:.1f}%)\n")

    failures = [r for r in judged if not r["judge"]["passed"]]
    if failures:
        lines.append("## Failing cases\n")
        for r in failures:
            lines.append(f"### {r['id']}")
            lines.append(f"- **Expected:** {r['expected_result']}")
            lines.append(f"- **Actual reply:** {r['replay']['final_reply']}")
            lines.append(f"- **Judge:** {r['judge']['reasoning']}\n")

    passing = [r for r in judged if r["judge"]["passed"]]
    if passing:
        lines.append("## Passing cases\n")
        for r in passing:
            lines.append(f"- {r['id']}: {r['expected_result']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=CASES_RESOLVED_DIR, help="Directory of resolved case JSON files")
    parser.add_argument("--model", default=BASE_MODEL_ID, help="Model name as served by vLLM on the pod")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL (reach via SSH tunnel)")
    parser.add_argument("--results-output", type=Path, default=CASES_RESULTS_DIR / "results.jsonl")
    parser.add_argument("--report-output", type=Path, default=CASES_RESULTS_DIR / "report.md")
    args = parser.parse_args()

    case_files = sorted(args.cases_dir.glob("*.json"))
    if not case_files:
        raise SystemExit(f"No resolved cases found in {args.cases_dir}.")

    cases = [json.loads(f.read_text()) for f in case_files]
    client = anthropic.Anthropic()
    retriever = ProductRetriever()
    model = PodChatModel(model=args.model, base_url=args.base_url)

    print(f"Replaying and judging {len(cases)} case(s)...")
    judged = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(with_retries, judge_one, client, retriever, model, c): c["id"] for c in cases}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                judged.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! case {case_id} failed: {exc}")

    if not judged:
        raise SystemExit(f"All {len(cases)} cases failed — nothing to report. Check the errors above.")

    judged.sort(key=lambda r: r["id"])
    with args.results_output.open("w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = build_report(judged)
    args.report_output.write_text(report)

    print(f"\nWrote {len(judged)} judged case(s) to {args.results_output}")
    print(f"Wrote report to {args.report_output}")
    print("\n" + report.split("## ")[0])


if __name__ == "__main__":
    main()
