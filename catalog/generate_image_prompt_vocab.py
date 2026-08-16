"""Translate the catalog's Danish `colors`/`material` vocabulary into short,
English, prompt-ready phrases for the SDXL image-generation pipeline
(catalog/generate_product_images.py).

Colors (273 distinct values) and materials (750 distinct, often long
free-text Danish sentences) can't feed an English SDXL prompt as-is, and are
too numerous/irregular to hand-translate — so each distinct value gets one
short Claude-generated English phrase, cached to disk. Category names don't
need this: there are only 22 of them, hand-mapped directly in
generate_product_images.py.

Usage:
    python -m catalog.generate_image_prompt_vocab

Writes catalog/data/image_prompt_vocab.json: {"colors": {...}, "materials": {...}}
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.structured_client import ClaudeStructuredClient, with_retries  # noqa: E402
from config import CATALOG_PATH, CLAUDE_MODEL, GENERATION_MAX_WORKERS  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "image_prompt_vocab.json"

BATCH_SIZE = 40

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "matches the input list's index"},
                    "phrase": {"type": "string", "description": "short English, prompt-ready phrase"},
                },
                "required": ["index", "phrase"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}

COLOR_PROMPT = """Translate each Danish furniture color/finish name below into a short English
phrase suitable for an image-generation prompt (e.g. "Sort med grå detaljer" ->
"black with gray accent details", "Mørkegrå" -> "dark gray", "Naturligt egetræ" ->
"natural oak wood tone"). Keep each phrase under 6 words. Return one entry per
input, in the same order, with "index" matching the input's position (0-based).

Inputs:
{items}"""

MATERIAL_PROMPT = """Translate each Danish furniture material description below into a short
English phrase suitable for an image-generation prompt — compress it to the
key visual materials only, dropping marketing language (e.g. "Fremstillet af
højkvalitets aluminium og polstret i blødt, luksuriøst stof." -> "aluminum
frame with soft fabric upholstery", "Massivt egetræ med oliebehandling" ->
"solid oak wood with an oiled finish"). Keep each phrase under 8 words. Return
one entry per input, in the same order, with "index" matching the input's
position (0-based).

Inputs:
{items}"""


def translate_batch(client: ClaudeStructuredClient, prompt_template: str, batch: list[str]) -> dict[int, str]:
    items = "\n".join(f"{i}. {value}" for i, value in enumerate(batch))
    result = client.generate(prompt_template.format(items=items), SCHEMA, max_tokens=4000)
    return {t["index"]: t["phrase"] for t in result["translations"]}


def translate_all(client: ClaudeStructuredClient, prompt_template: str, values: list[str]) -> dict[str, str]:
    batches = [values[i:i + BATCH_SIZE] for i in range(0, len(values), BATCH_SIZE)]
    translations: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=GENERATION_MAX_WORKERS) as pool:
        futures = {
            pool.submit(with_retries, translate_batch, client, prompt_template, batch): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            by_index = future.result()
            for i, value in enumerate(batch):
                if i in by_index:
                    translations[value] = by_index[i]
                else:
                    print(f"  ! missing translation for {value!r}, skipping")

    return translations


def main() -> None:
    products = json.loads(CATALOG_PATH.read_text())
    colors = sorted({c for p in products for c in p["colors"]})
    materials = sorted({p["material"] for p in products})

    client = ClaudeStructuredClient(CLAUDE_MODEL)

    print(f"Translating {len(colors)} distinct colors...")
    color_vocab = translate_all(client, COLOR_PROMPT, colors)

    print(f"Translating {len(materials)} distinct materials...")
    material_vocab = translate_all(client, MATERIAL_PROMPT, materials)

    OUTPUT_PATH.write_text(
        json.dumps({"colors": color_vocab, "materials": material_vocab}, ensure_ascii=False, indent=2)
    )
    print(f"\nWrote {len(color_vocab)} colors + {len(material_vocab)} materials to {OUTPUT_PATH}")

    missing_colors = set(colors) - set(color_vocab)
    missing_materials = set(materials) - set(material_vocab)
    if missing_colors or missing_materials:
        print(f"  ! {len(missing_colors)} colors and {len(missing_materials)} materials failed translation")


if __name__ == "__main__":
    main()
