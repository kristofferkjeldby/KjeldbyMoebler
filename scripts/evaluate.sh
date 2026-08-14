#!/usr/bin/env bash
# Run the fine-tuned model (served on the pod via vLLM, reached through an SSH
# tunnel) over the test suite, then judge every answer with Claude Sonnet and
# produce tests/results/report.md.
#
# Usage: bash scripts/evaluate.sh [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/2: Running pod-hosted model over test questions =="
python -m tests.run_eval "$@"

echo "== 2/2: Judging answers with Claude Sonnet =="
python -m tests.judge

echo ""
echo "Report written to tests/results/report.md"
