"""Generate real per-product/per-color catalog photos via SDXL + ControlNet,
replacing the symlink scaffolding under app/images/products/ (see
catalog/generate_image_catalog.py) with actual generated images.

Approach, validated via manual pilot on the target pod:
  1. One SDXL base image per product (its first listed color), prompted as
     an isolated e-commerce product photo.
  2. rembg background removal + white-canvas composite — SDXL alone only
     produces a clean isolated shot on a fraction of seeds (it has a strong
     prior toward full "living room" scenes for furniture prompts); this
     step makes the white background deterministic regardless of what SDXL
     renders behind the product.
  3. Crop to the product's bounding box and re-centre with a consistent
     margin, so framing doesn't vary product to product.
  4. Canny-edge that cleaned base image, then run ControlNet-conditioned
     generation for every other color the product comes in — the edge map
     locks pose/silhouette/proportions, only the color-describing part of
     the prompt changes.

Runs on a GPU pod (needs ~16GB+ VRAM for SDXL + ControlNet). Copy this
script alongside config.py, catalog/image_service.py, catalog/data/catalog.json,
and catalog/data/image_prompt_vocab.json onto the pod first, in the same
relative layout as the repo, then:

    python -m catalog.generate_product_images [--category X] [--limit N] [--force]

Writes JPEGs into app/images/products/<category-folder>/<sku>/{default,<color-slug>}.jpg
(replacing any existing symlink scaffolding). Afterwards, rsync
app/images/products/ back to the Mac and run catalog/generate_image_manifest.py
there to refresh app/data/image_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image
from rembg import new_session, remove

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.image_service import CATEGORY_FOLDERS, PRODUCTS_ROOT, slugify  # noqa: E402
from config import CATALOG_PATH  # noqa: E402

VOCAB_PATH = Path(__file__).resolve().parent / "data" / "image_prompt_vocab.json"
FAILURES_LOG = Path(__file__).resolve().parent / "data" / "image_gen_failures.log"

STEPS = 30
CONTROLNET_SCALE = 0.6
JPEG_QUALITY = 90
# Fraction of the canvas left as empty margin around the recentred product —
# SDXL's own framing/zoom varies product to product; this normalizes it so
# every image has the item occupying roughly the same fraction of frame.
MARGIN_FRAC = 0.08

# Only 22 categories — hand-mapped directly, no translation needed (unlike
# colors/materials, which are free-text Danish; see generate_image_prompt_vocab.py).
CATEGORY_IMAGE_PHRASES: dict[str, str] = {
    "sofa": "sofa",
    "armchair": "armchair",
    "loveseat": "loveseat sofa",
    "sectional": "sectional sofa",
    "dining_table": "dining table",
    "coffee_table": "coffee table",
    "side_table": "side table",
    "console_table": "console table",
    "bed_frame": "bed frame",
    "nightstand": "nightstand",
    "dresser": "dresser",
    "wardrobe": "wardrobe",
    "bookshelf": "bookshelf",
    "desk": "desk",
    "office_chair": "office chair",
    "dining_chair": "dining chair",
    "bar_stool": "bar stool",
    "tv_stand": "TV stand",
    "outdoor_set": "outdoor furniture set",
    "rug": "area rug",
    "lighting": "lamp",
    "kitchen_unit": "kitchen cabinet unit",
}

# Terms describing unwanted room props/context. Deliberately a list, not a
# single string: several of these ("lamp", "coffee table", "rug") are also
# literal category names — when generating a lamp itself, having "lamp" in
# the negative prompt directly contradicts the positive prompt asking for
# one, which produced genuinely confused output (observed: two different
# lamp designs rendered side by side instead of one). build_neg_prompt()
# excludes any term that overlaps the current category's own phrase.
NEG_PROMPT_BASE_TERMS = [
    "interior", "living room", "house", "room", "window", "curtains", "artwork", "painting",
    "plant", "rug", "carpet", "lamp", "coffee table", "pillow", "cushion", "decor",
    "other furniture", "wall art", "wood floor", "close-up", "macro", "cropped", "zoomed in",
    "text", "watermark", "logo", "blurry", "low quality", "people", "human", "border",
    "frame", "vignette", "drop shadow", "colored background", "gradient background",
    "multiple views", "multiple angles", "photo collage", "collage", "grid layout",
    "contact sheet", "multiple photos", "split image", "comparison", "montage",
    "stairs", "staircase", "steps", "stacked blocks", "stacked boxes",
    "architectural background structure", "buildings", "city", "skyline",
    "two of the same item", "pair", "set of two", "multiple identical items", "duplicate item",
    "decorative vases", "decorative objects", "styling props", "ornaments",
]


def build_neg_prompt(category: str) -> str:
    self_phrase = CATEGORY_IMAGE_PHRASES.get(category, category).lower()
    terms = [
        t for t in NEG_PROMPT_BASE_TERMS
        if t.lower() not in self_phrase and self_phrase not in t.lower()
    ]
    return ", ".join(terms)

# SDXL sometimes renders a "spec sheet" style collage — the same product
# from 2-3 different angles laid out on one canvas — instead of a single
# photo, for a fraction of seeds (observed on a small manual test batch:
# roughly 2 of 3 products). rembg still cleanly cuts out each fragment, so
# the giveaway isn't a messy mask, it's several separate, similarly-sized
# blobs instead of one dominant one. Retried with a different seed rather
# than accepted, since it silently corrupts both the "layout should be
# consistent" requirement and the ControlNet-conditioned color variants
# that inherit whatever the base's structure was.
MAX_GEN_ATTEMPTS = 4
SINGLE_COMPONENT_AREA_FRACTION = 0.85


def build_prompt(category: str, material_phrase: str, color_phrase: str) -> str:
    # Product/color lead the prompt (CLIP truncates at 77 tokens — the most
    # important, variant-specific words must come first, generic style
    # boilerplate last, so if anything gets cut it's the boilerplate).
    product = f"a complete {CATEGORY_IMAGE_PHRASES.get(category, category)} with {material_phrase}"
    return (
        f"{color_phrase} {product}, isolated product photo, cutout on solid "
        f"pure white background, studio product photography, e-commerce "
        f"catalog listing photo, no background, no room, no props, entire "
        f"item visible, wide shot, centered, square image"
    )


def remove_background(raw: Image.Image, session) -> Image.Image:
    return remove(raw, session=session)


def is_single_component(cutout: Image.Image) -> bool:
    """True if one connected blob accounts for most of the foreground —
    false for a multi-panel collage (several similarly-sized, separate
    blobs) or an empty/near-empty mask. Used to decide whether to retry
    with a different seed, not to accept/reject what ends up on disk —
    see keep_largest_component for that."""
    alpha = np.array(cutout.split()[3])
    # A much stricter cutoff than the compositing mask (>10) — two visually
    # separate items are sometimes bridged by a faint low-alpha shadow or
    # reflection, which >10 treats as one connected blob. Requiring
    # near-opaque pixels for *connectivity* (not for the final soft-edged
    # composite, which still uses the full alpha) breaks that false bridge.
    binary = (alpha > 160).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = areas.sum()
    if total == 0:
        return False
    return (areas.max() / total) >= SINGLE_COMPONENT_AREA_FRACTION


def keep_largest_component(cutout: Image.Image) -> Image.Image:
    """Zeroes out every blob except the largest one. A smaller stray blob
    (a background remnant rembg didn't fully clear, a rendering artifact,
    or a second item entirely) can pass the is_single_component ratio
    check — since it's a minority of the foreground area — yet still show
    up as visible clutter in the final crop. This makes the "only the
    product itself" guarantee unconditional rather than probabilistic.

    Uses the same strict threshold as is_single_component to decide
    connectivity (two visually separate items are sometimes bridged by a
    faint low-alpha shadow the soft compositing mask would treat as one
    blob), then dilates the kept blob back out a few pixels so the item's
    own soft/antialiased edge isn't clipped off in the process."""
    alpha = np.array(cutout.split()[3])
    strict = (alpha > 160).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strict, connectivity=8)
    if num_labels <= 2:  # background + at most one foreground blob already
        return cutout
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    largest_mask = (labels == largest_label).astype(np.uint8)
    dilated = cv2.dilate(largest_mask, np.ones((15, 15), np.uint8), iterations=1)
    new_alpha = Image.fromarray(np.where(dilated.astype(bool), alpha, 0).astype(np.uint8))
    r, g, b, _ = cutout.split()
    return Image.merge("RGBA", (r, g, b, new_alpha))


def recentre(cutout: Image.Image, size: int) -> Image.Image:
    alpha = np.array(cutout.split()[3])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(cutout, mask=cutout.split()[3])
        return canvas

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = cutout.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))

    margin = int(size * MARGIN_FRAC)
    target_w, target_h = size - 2 * margin, size - 2 * margin
    scale = min(target_w / cropped.width, target_h / cropped.height, 1.0)
    new_w, new_h = max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), mask=resized.split()[3])
    return canvas


def generate_clean(pipe_call, base_seed: int, rembg_session, size: int) -> tuple[Image.Image, bool]:
    """Calls `pipe_call(generator) -> raw PIL image` up to MAX_GEN_ATTEMPTS
    times with different seeds (derived from base_seed) until rembg's mask
    shows a single dominant blob, then crops/recentres it. Returns
    (image, ok) — ok is False if every attempt still looked like a collage,
    in which case the caller should treat this as a failure rather than
    write a known-bad image."""
    cutout = None
    for attempt in range(MAX_GEN_ATTEMPTS):
        # Large, decorrelated offsets rather than base_seed+1/+2/... so a
        # retry isn't just a near-identical noise sample of the same bad
        # composition.
        seed = base_seed + attempt * 7919
        generator = torch.Generator("cuda").manual_seed(seed)
        raw = pipe_call(generator)
        cutout = remove_background(raw, rembg_session)
        if is_single_component(cutout):
            return recentre(keep_largest_component(cutout), size), True
    return recentre(keep_largest_component(cutout), size), False


def canny_edges(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    edges = cv2.Canny(arr, 80, 160)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def is_real_file(path: Path) -> bool:
    # Plain .exists() follows symlinks — the pre-existing scaffolding
    # (catalog/generate_image_catalog.py) symlinks every color to a real
    # category photo elsewhere, so .exists() would be True for a color that
    # hasn't actually been generated yet. Only a non-symlink counts as done.
    return path.exists() and not path.is_symlink()


def write_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    image.save(path, "JPEG", quality=JPEG_QUALITY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="Only process this category")
    parser.add_argument("--sku", action="append", help="Only process this SKU (repeatable) — for spot-fixing individual products after a full run")
    parser.add_argument("--limit", type=int, help="Only process the first N matching products")
    parser.add_argument("--force", action="store_true", help="Regenerate even if files already exist")
    parser.add_argument("--start-after", help="Skip products up to and including this SKU (resume aid)")
    args = parser.parse_args()

    products = json.loads(CATALOG_PATH.read_text())
    vocab = json.loads(VOCAB_PATH.read_text())

    if args.category:
        products = [p for p in products if p["category"] == args.category]
    if args.sku:
        wanted = set(args.sku)
        products = [p for p in products if p["sku"] in wanted]
    if args.start_after:
        skus = [p["sku"] for p in products]
        if args.start_after in skus:
            products = products[skus.index(args.start_after) + 1:]
    if args.limit:
        products = products[:args.limit]

    print(f"Processing {len(products)} product(s)...")

    base_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    rembg_session = new_session("isnet-general-use")

    done = 0
    for product in products:
        sku = product["sku"]
        category = product["category"]
        cat_folder = CATEGORY_FOLDERS.get(category, category)
        product_dir = PRODUCTS_ROOT / cat_folder / sku
        material_phrase = vocab["materials"].get(product["material"], product["material"])
        colors = product["colors"]

        # Deterministic per-product seed (stable across reruns) — the same
        # numeric seed is reused fresh for every image of this product
        # (base + each color), not carried forward through one Generator
        # object, matching the pilot: identical starting noise per call is
        # part of what keeps structure consistent across color variants.
        seed = int(sku.split("-")[-1])

        default_path = product_dir / "default.jpg"
        first_color_path = product_dir / f"{slugify(colors[0])}.jpg"
        neg_prompt = build_neg_prompt(category)

        try:
            if args.force or not is_real_file(default_path):
                color_phrase = vocab["colors"].get(colors[0], colors[0])
                prompt = build_prompt(category, material_phrase, color_phrase)
                base_clean, ok = generate_clean(
                    lambda gen: base_pipe(
                        prompt, negative_prompt=neg_prompt, num_inference_steps=STEPS, generator=gen,
                    ).images[0],
                    seed, rembg_session, size=1024,
                )
                if not ok:
                    raise RuntimeError(f"base image still looked like a multi-view collage after {MAX_GEN_ATTEMPTS} attempts")
                write_image(base_clean, default_path)
                write_image(base_clean, first_color_path)
            else:
                base_clean = Image.open(default_path).convert("RGB")

            if len(colors) > 1:
                canny = canny_edges(base_clean)
                for color in colors[1:]:
                    color_path = product_dir / f"{slugify(color)}.jpg"
                    if not args.force and is_real_file(color_path):
                        continue
                    color_phrase = vocab["colors"].get(color, color)
                    prompt = build_prompt(category, material_phrase, color_phrase)
                    variant_clean, ok = generate_clean(
                        lambda gen: cn_pipe(
                            prompt, negative_prompt=neg_prompt, image=canny,
                            num_inference_steps=STEPS, controlnet_conditioning_scale=CONTROLNET_SCALE,
                            generator=gen,
                        ).images[0],
                        seed, rembg_session, size=1024,
                    )
                    if not ok:
                        raise RuntimeError(f"{color!r} variant still looked like a multi-view collage after {MAX_GEN_ATTEMPTS} attempts")
                    write_image(variant_clean, color_path)

            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(products)} products done ({sku})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {sku} failed: {exc}")
            FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FAILURES_LOG.open("a") as f:
                f.write(f"{sku}\t{exc}\n")

    print(f"\nDone: {done}/{len(products)} products generated.")


if __name__ == "__main__":
    main()
