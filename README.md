# Furniture Store Chatbot — RAG + Fine-Tuned Llama POC

A proof-of-concept product chatbot: a fake furniture catalog, a RAG layer
that injects the relevant products into context, a Llama 3.1 8B model
QLoRA-fine-tuned (on RunPod) to answer grounded in that injected context, and
a 500-question eval suite judged by Claude Sonnet. The final model runs
locally on a Mac as a quantized GGUF file via `llama-cpp-python`, served
through a Gradio chat UI.

## Architecture

```
                    ┌─────────────────────┐
                    │  Claude Sonnet       │  generates: catalog, SFT
                    │  (data generation &  │  training data, 500 test
                    │   judging)           │  questions, and judges results
                    └──────────┬───────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
  catalog.json          sft_dataset.jsonl      test_questions.jsonl
       │                       │                       │
       ▼                       ▼                       │
  FAISS index            RunPod: QLoRA fine-tune        │
  (sentence-transformers) Llama 3.1 8B Instruct         │
       │                       │                        │
       │                merge LoRA -> GGUF (Q4_K_M)      │
       │                       │                        │
       │                 download .gguf to Mac           │
       │                       │                        │
       └──────────┬────────────┘                        │
                   ▼                                     │
        Gradio app (Mac, local inference)                │
        RAG-retrieved products -> system prompt          │
        -> llama-cpp-python -> streamed answer            │
                   │                                     │
                   └───────────── tests/run_eval.py ◄──────┘
                                  + tests/judge.py (Claude Sonnet)
                                  -> tests/results/report.md
```

## Where things run

| Stage | Runs on | Needs |
|---|---|---|
| Catalog generation, RAG indexing, SFT/test data generation | **Mac (local)** | `ANTHROPIC_API_KEY` |
| QLoRA fine-tuning, LoRA merge, GGUF conversion | **RunPod GPU pod** | HF token with Llama-3.1 access, GPU (24GB+ VRAM) |
| Gradio chat app, eval runner | **Mac (local)** | the downloaded `.gguf` file |
| Judging | **Mac (local)**, calls Claude | `ANTHROPIC_API_KEY` |

## Setup

```bash
# Local (Mac)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or copy .env.example to .env
```

## 1. Generate the catalog, RAG index, and training/eval data

```bash
bash scripts/generate_all_data.sh
```

This runs, in order:
- `catalog/generate_catalog.py` — 200 fake furniture products (price, colors,
  dimensions, stock, material, rating, lead time, warranty, etc.) via Claude Sonnet.
- `rag/build_index.py` — embeds the catalog with `all-MiniLM-L6-v2` and builds a FAISS index.
- `training/generate_training_data.py` — ~1,500+ grounded Q&A examples (single-product,
  multi-product comparison, and "can't answer" refusals), each with the exact
  system-prompt + injected-context shape the model will see at inference time.
- `tests/generate_test_questions.py` — 500 test questions with ground-truth
  reference answers, in the same three categories.

## 2. Fine-tune on RunPod

Connect to the pod:

```bash
ssh root@213.192.2.81 -p 40064 -i ~/.ssh/id_ed25519
```

Then, on the pod:

```bash
git clone <this repo> && cd KjeldProduct-Llama
pip install -r requirements-train.txt
huggingface-cli login   # needs a token with Llama-3.1 access

python -m training.train_qlora            # -> training/output/lora-adapter
python -m training.merge_lora             # -> training/output/merged-model (fp16)
bash training/convert_to_gguf.sh          # -> models/furniture-assistant-...-Q4_K_M.gguf
```

Copy `training/data/sft_dataset.jsonl` up to the pod before training (e.g.
`scp -P 40064 -i ~/.ssh/id_ed25519 training/data/sft_dataset.jsonl root@213.192.2.81:~/KjeldProduct-Llama/training/data/`),
and copy just the final `.gguf` file back down to your Mac's `models/`
directory afterwards (e.g. `scp -P 40064 -i ~/.ssh/id_ed25519 root@213.192.2.81:~/KjeldProduct-Llama/models/*.gguf models/`)
— it's a few GB, far smaller than the merged fp16 model, so no need to pull
the intermediate artifacts down.

## 3. Chat with it locally

```bash
python -m app.gradio_app
```

Opens a Gradio chat UI. Every message is embedded, matched against the FAISS
index, and the top-3 products are injected into the system prompt before the
local model generates a streamed reply — so it can only ever "know about"
products that were actually retrieved.

## 4. Evaluate

```bash
bash scripts/evaluate.sh                          # full 500-question run
bash scripts/evaluate.sh "" --limit 20             # quick smoke test
bash scripts/evaluate.sh models/my-other-run.gguf  # test a specific checkpoint
```

Runs every test question through the same RAG + local-model path as the
Gradio app, then has Claude Sonnet judge each answer for factual accuracy,
hallucination, and appropriate refusal on unanswerable questions. Produces
`tests/results/report.md` with an overall pass rate, hallucination rate, and a
breakdown by question type (single-product / multi-product / unanswerable).

## Project layout

```
config.py                        # shared paths, model IDs, system prompt template
catalog/
  generate_catalog.py            # Claude Sonnet -> catalog.json
  product_format.py              # product -> context text / embedding text (shared everywhere)
rag/
  build_index.py                 # catalog -> FAISS index
  retriever.py                   # query -> top-k products
training/
  generate_training_data.py      # Claude Sonnet -> sft_dataset.jsonl
  train_qlora.py                 # RunPod: QLoRA fine-tune
  merge_lora.py                  # RunPod: merge adapter -> fp16 model
  convert_to_gguf.sh             # RunPod: fp16 -> quantized GGUF
tests/
  generate_test_questions.py     # Claude Sonnet -> 500 test questions + reference answers
  run_eval.py                    # local model + RAG -> raw answers
  judge.py                       # Claude Sonnet -> scored report
app/
  llm_backend.py                 # llama-cpp-python wrapper (shared by app + eval)
  gradio_app.py                  # chat UI
scripts/
  generate_all_data.sh
  evaluate.sh
```

## Notes / things to tune for a real project

- **Catalog size / eval size**: `NUM_PRODUCTS`, `NUM_TEST_QUESTIONS`, etc. are
  all in `config.py`.
- **Cost**: data generation and judging make one Claude Sonnet call per
  example/question (with concurrency). For 200 products + ~1,700 training
  examples + 500 test questions + 500 judged answers, expect on the order of
  a few thousand Sonnet calls total — cheap in absolute terms, but if you
  scale the catalog up significantly, consider switching the generation
  scripts to the Message Batches API (50% cheaper, async).
- **Base model**: `BASE_MODEL_ID` in `config.py` defaults to Llama 3.1 8B
  Instruct — a good fit for a single RunPod GPU and for running comfortably
  quantized on a Mac. Swap to a smaller model (e.g. Llama 3.2 3B) for faster
  iteration, or a larger one if you have more RunPod budget (note: this
  changes VRAM requirements for both training and inference).
- **Quantization**: `Q4_K_M` is a good default quality/size tradeoff for Mac
  inference. Use `Q5_K_M` or `Q8_0` if you have RAM to spare and want higher
  fidelity, at the cost of a larger file and slightly slower inference.
