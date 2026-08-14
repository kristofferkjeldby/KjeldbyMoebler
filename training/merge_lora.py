"""Merge the trained LoRA adapter into the base model, producing a standalone
fp16 model ready to be served directly via vLLM on the pod.

Runs on the RunPod pod right after train_qlora.py (needs the merged model in
fp16, not 4-bit, so this reloads the base model in fp16 first).

Usage:
    python -m training.merge_lora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL_ID, LORA_OUTPUT_DIR, MERGED_MODEL_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=LORA_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=MERGED_MODEL_DIR)
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"No adapter found at {args.adapter_dir}. Run training/train_qlora.py first.")

    print(f"Loading base model '{args.base_model}' in fp16...")
    # Explicit single-GPU placement, same reasoning as train_qlora.py:
    # device_map="auto" speculatively probes free VRAM and can refuse rather
    # than just placing everything on the (only) GPU.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, fix_mistral_regex=True)

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nSaved merged fp16 model to {args.output_dir}")
    print("Next: serve the merged model with vLLM on the pod (see catalog/structured_client.py docstring for the launch command).")


if __name__ == "__main__":
    main()
