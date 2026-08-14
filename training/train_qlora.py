"""QLoRA fine-tune of Mistral-Small-3.1-24B-Instruct on the generated Danish
product Q&A dataset.

Runs on the RunPod GPU pod (96GB VRAM). Not meant to run on a Mac — this is
the training half; see training/merge_lora.py for merging the adapter, then
serve the merged model directly via vLLM on the pod (no local Mac inference).

Usage (on the RunPod pod, after `pip install -r requirements-train.txt`):

    python -m training.train_qlora \
        --dataset training/data/sft_dataset.jsonl \
        --output-dir training/output/lora-adapter \
        --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL_ID, LORA_OUTPUT_DIR, TRAINING_DATA_PATH  # noqa: E402

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def formatting_func(example: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=TRAINING_DATA_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=LORA_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3,
        help="Stop after this many consecutive evals with no eval_loss improvement (beyond --early-stopping-threshold). Set to 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-threshold", type=float, default=0.0,
        help="Minimum eval_loss decrease to count as an improvement — filters out noise, not just any uptick.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found at {args.dataset}. Run training/generate_training_data.py first.")

    print(f"Loading base model '{args.base_model}' in 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # Explicit single-GPU placement rather than device_map="auto": "auto"
    # speculatively probes free VRAM and refuses (rather than just placing
    # everything on GPU) if it estimates too little headroom — fragile on a
    # single-GPU pod where something else (e.g. vLLM) may transiently still
    # hold memory. This pod has exactly one GPU, so pin to it directly.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # fix_mistral_regex: this specific repo's tokenizer_config.json ships a
    # known-broken split regex (see the load-time warning) that silently
    # mis-tokenizes without this flag — cheap to set, expensive to discover
    # after training on subtly wrong tokens.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_seq_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # save_strategy must match eval_strategy for load_best_model_at_end —
        # that's what lets early stopping actually discard an overfit
        # checkpoint at the end rather than keeping whatever the last step
        # happened to be.
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        packing=False,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    # With load_best_model_at_end=True, the trainer has already swapped in
    # the lowest-eval_loss checkpoint (not necessarily the final step) before
    # this save — early stopping only helps if we actually persist that one.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nSaved best LoRA adapter (eval_loss={trainer.state.best_metric:.4f}) to {args.output_dir}")


if __name__ == "__main__":
    main()
