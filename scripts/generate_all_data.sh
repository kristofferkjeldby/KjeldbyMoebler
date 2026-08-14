#!/usr/bin/env bash
# Local, Claude-Sonnet-powered data generation: catalog -> RAG index ->
# SFT training data -> 500-question test suite. Run this before uploading
# training/data/sft_dataset.jsonl to RunPod for fine-tuning.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4: Generating fake furniture catalog =="
python -m catalog.generate_catalog

echo "== 2/4: Building RAG index (FAISS + sentence-transformers) =="
python -m rag.build_index

echo "== 3/4: Generating SFT training data (grounded Q&A) =="
python -m training.generate_training_data

echo "== 4/4: Generating 500-question test suite =="
python -m eval.generate_test_questions

echo ""
echo "Done. Next: upload training/data/sft_dataset.jsonl to your RunPod pod and"
echo "run train_qlora.py -> merge_lora.py -> convert_to_gguf.sh there (see README)."
