"""Shared configuration and paths for the whole pipeline.

Every stage (catalog generation, RAG indexing, training-data generation,
fine-tuning, evaluation, and the Gradio app) imports from here so paths and
the system-prompt template stay identical between training and inference.
"""
from pathlib import Path

ROOT = Path(__file__).parent

# --- Claude (data generation + judging) ---
CLAUDE_MODEL = "claude-sonnet-5"
# Judging (scoring a model answer against a reference for factual accuracy /
# hallucination) is a structured, comparison-style task rather than an
# open-ended generation one — the smallest current model is reasonable for
# it and meaningfully cheaper across a 500-question run.
JUDGE_MODEL = "claude-haiku-4-5"

# --- Catalog ---
CATALOG_DIR = ROOT / "catalog" / "data"
CATALOG_PATH = CATALOG_DIR / "catalog.json"
NUM_PRODUCTS = 3000

# Shared concurrency knob for all the Claude-generation scripts (catalog,
# training data, test questions) — bump this if you have room on your rate
# limit tier and want faster (if noisier / more retry-prone) generation.
GENERATION_MAX_WORKERS = 20

CATEGORIES = [
    "sofa", "armchair", "loveseat", "sectional",
    "dining_table", "coffee_table", "side_table", "console_table",
    "bed_frame", "nightstand", "dresser", "wardrobe",
    "bookshelf", "desk", "office_chair", "dining_chair", "bar_stool",
    "tv_stand", "outdoor_set", "rug",
    "lighting", "kitchen_unit",
]

# Which of the flexible `attributes` fields (see catalog/generate_catalog.py)
# are meaningful for each category — used to instruct Claude which fields to
# fill in and which to leave null, since all products share one flat
# attributes schema (structured-output schemas can't vary shape per row).
CATEGORY_ATTRIBUTE_FIELDS = {
    "dining_chair": ["seat_height_cm", "weight_capacity_kg"],
    "office_chair": ["seat_height_cm", "weight_capacity_kg"],
    "bar_stool": ["seat_height_cm", "weight_capacity_kg"],
    "dining_table": ["seats_count", "extendable"],
    "coffee_table": [],
    "side_table": [],
    "console_table": [],
    "sofa": ["seat_depth_cm", "weight_capacity_kg"],
    "armchair": ["seat_depth_cm", "weight_capacity_kg"],
    "loveseat": ["seat_depth_cm", "weight_capacity_kg"],
    "sectional": ["seat_depth_cm", "weight_capacity_kg"],
    "bed_frame": ["bed_size"],
    "nightstand": ["num_drawers"],
    "dresser": ["num_drawers"],
    "wardrobe": ["num_shelves", "num_drawers"],
    "bookshelf": ["num_shelves"],
    "desk": [],
    "tv_stand": ["num_shelves"],
    "outdoor_set": ["seats_count", "weather_resistant"],
    "rug": [],
    "lighting": ["bulb_type", "lumens", "wattage", "dimmable", "mount_type"],
    "kitchen_unit": ["unit_type", "door_count"],
}
ALL_ATTRIBUTE_FIELDS = sorted({f for fields in CATEGORY_ATTRIBUTE_FIELDS.values() for f in fields})

# --- Stores ---
STORES = ["København", "Århus", "Odense"]
OUT_OF_STOCK_PROBABILITY = 0.15   # per store, independently
RESTOCK_DAYS_RANGE = (7, 60)      # when out of stock, restock date is this many days out
STOCK_QUANTITY_RANGE = (0, 40)    # sampled when a store isn't forced out-of-stock

# --- Discounts ---
DISCOUNT_PROBABILITY = 0.2
DISCOUNT_PERCENT_RANGE = (10, 40)

# --- Series (matching sets, e.g. a dining table + its matching chairs) ---
NUM_SERIES = 300
SERIES_ARCHETYPES = [
    {"name": "dining_set", "categories": ["dining_table", "dining_chair"]},
    {"name": "bedroom_set", "categories": ["bed_frame", "nightstand", "dresser"]},
    {"name": "living_set", "categories": ["sofa", "armchair", "coffee_table"]},
]

# --- RAG ---
RAG_DIR = ROOT / "rag" / "data"
FAISS_INDEX_PATH = RAG_DIR / "catalog.index"
RAG_METADATA_PATH = RAG_DIR / "catalog_meta.json"
# Danish isn't in all-MiniLM-L6-v2's training data — multilingual-e5 covers it
# and is a drop-in sentence-transformers replacement. Note: e5 models expect
# "query: "/"passage: " prefixes on input text for best retrieval quality.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# Reranks the bi-encoder's candidate pool for the semantic-fallback tier
# (open-ended queries that hit neither name nor attribute matching) — a
# cross-encoder scores (query, candidate) pairs jointly, which is more
# precise than cosine similarity over independently-embedded vectors.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SEMANTIC_CANDIDATE_POOL_SIZE = 30  # bi-encoder recall pool, reranked down to RETRIEVAL_TOP_K
RETRIEVAL_TOP_K = 8          # semantic fallback for open-ended questions (name/attribute matching are tried first)
# Cap on how many products the model is actually shown/allowed to talk about
# in one reply, even when a query matches far more (e.g. "which sectional
# sofas do you have" can match 60+) — the top SHOWN_MAX_RESULTS by `priority`
# (catalog/product_format.py) are shown; the rest are only reachable via the
# "see all" link (app/gradio_app.py) or the category-table modal
# (app/landing_page.html). Was ENUMERATION_MAX_RESULTS (12) before priority
# ranking existed to pick *which* subset to show.
SHOWN_MAX_RESULTS = 8
DIMENSION_TOLERANCE_CM = 5    # "60cm deep" matches products within +/- this many cm
CATEGORY_URL_BASE = "https://kjeldbymobler.dk/kategori/{category}"  # "see all" link — intercepted client-side, see app/gradio_app.py

# --- Training data ---
TRAINING_DIR = ROOT / "training" / "data"
TRAINING_DATA_PATH = TRAINING_DIR / "sft_dataset.jsonl"
EXAMPLES_PER_PRODUCT = 6
# Generation now runs against the pod model (no per-token API cost), so full
# catalog coverage for single-product examples is affordable — every product
# gets seen at least once, instead of the ~17% sample the Claude-token budget
# used to force. The other categories are scaled up 3x to match.
NUM_SINGLE_PRODUCT_SAMPLE = 2920
NUM_MULTI_PRODUCT_EXAMPLES = 900
NUM_UNANSWERABLE_EXAMPLES = 450
NUM_ENUMERATION_EXAMPLES = 600   # "what chairs come in yellow?" style — list ALL matches, or none
NUM_STORE_STOCK_EXAMPLES = 600   # "which dining tables are available at the Odense store?"
NUM_DIMENSION_EXAMPLES = 300     # "I need a 60cm deep kitchen unit"
NUM_SERIES_EXAMPLES = 300        # "what matches this dining table?"
NUM_DISCOVERY_EXAMPLES = 500     # "I'm doing up my living room in a cozy style, any ideas?" — open-ended, no known product

# Room + vibe combinations used to construct open-ended "help me find
# something" discovery scenarios — the customer doesn't know what product
# they want yet, unlike every other example type above.
DISCOVERY_VIBES = [
    "hyggeligt og varmt", "minimalistisk og enkelt", "moderne og stilrent",
    "rustikt og naturligt", "skandinavisk", "industrielt", "elegant og klassisk",
    "børnevenligt og praktisk", "budgetvenligt", "luksuriøst",
]

# --- Fine-tuning (runs on RunPod) ---
# Mistral-Small-3.1-24B-Instruct-2503 (used for catalog/training-data
# generation via vLLM, per EuroEval Danish NLG 2025/11/04) turned out to be a
# vision-language model (Mistral3ForConditionalGeneration / Pixtral-based) —
# fine via vLLM's own loading path, but NOT registered under transformers'
# AutoModelForCausalLM, so QLoRA training via standard HF/PEFT/TRL tooling
# can't load it directly. For the actual fine-tuned chat model, we use its
# text-only predecessor instead: same Mistral-Small lineage and Apache-2.0
# license, plain MistralForCausalLM architecture, ungated. Served on the pod
# via vLLM after training (no GGUF conversion, no local Mac inference).
BASE_MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
LORA_OUTPUT_DIR = ROOT / "training" / "output" / "lora-adapter"
MERGED_MODEL_DIR = ROOT / "training" / "output" / "merged-model"

# --- Evaluation ---
EVAL_DIR = ROOT / "tests" / "data"
TEST_QUESTIONS_PATH = EVAL_DIR / "test_questions.jsonl"
NUM_TEST_QUESTIONS = 500
RESULTS_DIR = ROOT / "tests" / "results"
RAW_ANSWERS_PATH = RESULTS_DIR / "raw_answers.jsonl"
JUDGED_RESULTS_PATH = RESULTS_DIR / "judged_results.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"

# --- Cases: manual-testing bug reports captured from the chat's "Gem som
# test-case" button (app/gradio_app.py), turned into durable regression
# tests. A case starts in CASES_UNRESOLVED_DIR; once the underlying bug is
# fixed, moving the file to CASES_RESOLVED_DIR (a plain `mv`, no tooling)
# makes tests/run_case_tests.py replay and judge it on every future run. ---
CASES_DIR = ROOT / "cases"
CASES_UNRESOLVED_DIR = CASES_DIR / "unresolved"
CASES_RESOLVED_DIR = CASES_DIR / "resolved"
CASES_RESULTS_DIR = CASES_DIR / "results"

# --- Chat / system prompt (used identically in training data and inference) ---
SYSTEM_PROMPT_TEMPLATE = """Du er en venlig og kyndig assistent i en møbelforretning. \
Butikken har tre afdelinger: København, Århus og Odense.

Besvar kundens spørgsmål udelukkende ud fra de produktoplysninger, der er angivet nedenfor. \
Opfind aldrig priser, mål, farver, materialer, lagerstatus eller lokal tilgængelighed, \
som ikke udtrykkeligt er angivet. Hvis svaret ikke findes i de angivne produkter, så sig \
det ærligt og tilbyd at hjælpe med noget andet — gæt aldrig.

Hvis kunden spørger om en egenskab, der slet ikke er nævnt i produktoplysningerne (f.eks. \
USB-stik, vejrbestandighed, om et materiale er genbrugt, hylders bæreevne), betyder det \
IKKE, at egenskaben ikke findes — det betyder, at du ikke har information om det. Sig det \
ærligt ("vi har desværre ikke oplysninger om det"), og udled eller gæt aldrig et svar ud fra \
relaterede oplysninger (f.eks. at gætte vejrbestandighed ud fra materialet).

Når du nævner to eller flere produkter, så brug en Markdown-punktliste (en linje pr. \
produkt, startende med "- "), i stedet for at remse dem op i en sætning.

Tilgængelige produkter:
{context}"""

# --- Danish-language retrieval vocab ---
# Categories themselves stay as the same English internal keys used
# everywhere else (SKUs, training data, attribute schema) — this only maps
# how a Danish-speaking customer would refer to them in a query, used by
# rag/retriever.py's rule-based category-detection tier.
CATEGORY_PHRASES = {
    "sofa": "sofa",
    "armchair": "lænestol",
    "loveseat": "to personers sofa",
    "sectional": "hjørnesofa",
    "dining_table": "spisebord",
    "coffee_table": "sofabord",
    "side_table": "sidebord",
    "console_table": "konsolbord",
    "bed_frame": "sengeramme",
    "nightstand": "natbord",
    "dresser": "kommode",
    "wardrobe": "garderobeskab",
    "bookshelf": "bogreol",
    "desk": "skrivebord",
    "office_chair": "kontorstol",
    "dining_chair": "spisebordsstol",
    "bar_stool": "barstol",
    "tv_stand": "tv-bord",
    "outdoor_set": "havemøbelsæt",
    "rug": "tæppe",
    "lighting": "belysning",
    "kitchen_unit": "køkkenelement",
}
CATEGORY_WORDS = {
    "sofa": {"sofa", "sofaer"},
    # "lænestol" (lounge/recliner chair) is the formal term; "armstol" (arm
    # chair, literal) is at least as common colloquially and was previously
    # missing entirely — a real customer saying "armstole" got zero category
    # match and silently fell back to weak semantic search.
    "armchair": {"lænestol", "lænestole", "armstol", "armstole"},
    "loveseat": {"loveseat", "loveseater", "to-personers", "sofa"},
    "sectional": {"hjørnesofa", "hjørnesofaer"},
    # "spisestuebord" (dining ROOM table, more formal/common phrasing) was
    # missing alongside the shorter "spisebord".
    "dining_table": {"spisebord", "spiseborde", "spisestuebord", "spisestueborde"},
    # "kaffebord" (coffee table, literal) is arguably the more common Danish
    # term than "sofabord" (sofa table) and was previously missing entirely.
    "coffee_table": {"sofabord", "sofaborde", "kaffebord", "kaffeborde"},
    "side_table": {"sidebord", "sideborde"},
    "console_table": {"konsolbord", "konsolborde"},
    "bed_frame": {"sengeramme", "sengerammer", "seng", "senge"},
    "nightstand": {"natbord", "natborde"},
    "dresser": {"kommode", "kommoder"},
    # Bare "skab"/"skabe" (generic "cabinet") is ambiguous between wardrobe
    # and kitchen_unit — it was previously an EXTRA_CATEGORY_TRIGGER_WORDS
    # entry for kitchen_unit only, so a customer asking about a generic
    # "skab" got silently routed to kitchen units exclusively, wardrobe
    # never considered. Listing it here too means both categories match
    # (a proper union, not a wrong exclusive pick) and downstream ranking
    # sorts it out from there.
    "wardrobe": {"garderobeskab", "garderobeskabe", "klædeskab", "klædeskabe", "skab", "skabe"},
    # "boghylde" (book shelf, literal) was missing alongside "bogreol"/"reol".
    "bookshelf": {"bogreol", "bogreoler", "reol", "reoler", "boghylde", "boghylder"},
    "desk": {"skrivebord", "skriveborde"},
    "office_chair": {"kontorstol", "kontorstole"},
    # "spisestol" (dining chair, shorter/more natural form) was missing
    # alongside the more formal "spisebordsstol".
    "dining_chair": {"spisebordsstol", "spisebordsstole", "spisestol", "spisestole"},
    # "barstool" (English loanword spelling, common code-switch) is caught
    # via the prefix-fallback tier against "barstooler" etc.
    "bar_stool": {"barstol", "barstole", "barstool"},
    # "tv-bord" is hyphenated and tokenizes to "tv" + "bord" — a query using
    # the English loanword "tv-stand[s]" instead never matched either half.
    # "stand"/"stands" catches it via the prefix-fallback tier.
    "tv_stand": {"tv-bord", "tv-borde", "stand", "stands"},
    # "terrassesæt" (patio/terrace set) was missing alongside "havemøbelsæt".
    "outdoor_set": {"havemøbelsæt", "havemøbler", "terrassesæt"},
    "rug": {"tæppe", "tæpper"},
    # "lysarmatur"/"lysprodukt" (light fixture/product) were missing — NOT
    # adding bare "lys" (light/pale), which collides with color names like
    # "Lys Grå"/"Lys Egetræ" and would false-positive on color-only queries.
    "lighting": {"belysning", "lampe", "lamper", "lysarmatur", "lysarmaturer", "lysprodukt", "lysprodukter"},
    # "køkkenenhed" (kitchen unit, the literal translation of the category's
    # own English name) was missing alongside "køkkenelement"/"køkkenskab".
    "kitchen_unit": {"køkkenelement", "køkkenelementer", "køkkenskab", "køkkenskabe", "køkkenenhed", "køkkenenheder"},
}
EXTRA_CATEGORY_TRIGGER_WORDS = {
    "kitchen_unit": {"skab", "skabe", "element", "elementer"},
}
# Danish grammatical connectors that show up constantly inside compound
# catalog color descriptions ("Sort MED grå detaljer", "Hvid MED guld") but
# aren't themselves color/material terms — indexing them as color-trigger
# words meant an unrelated customer phrase merely containing one of them
# (e.g. "Hvad MED rød?", the idiom "what about red?") spuriously matched
# every catalog color containing that connector as a substring word,
# including products with no real relation to what was asked. The plain
# length>=3 guard alone doesn't exclude "med" (with) — it's exactly 3
# chars — so this stopword list is checked in addition to it.
COLOR_WORD_STOPWORDS = {"med", "og", "til", "uden", "af", "som", "den", "det", "for"}

# A bare generic term that spans several distinct catalog categories, none
# of which is itself named that generic word — e.g. "stol" (chair) isn't a
# catalog category on its own; it's the shared head-noun of four specific
# compound categories (kontorstol/office_chair, spisebordsstol/dining_chair,
# barstol/bar_stool, lænestol/armchair). Without this, "Hvilke stole har
# I?" matched no category at all and fell through to semantic search,
# which returned a noisy, largely-irrelevant mix (a desk, an outdoor set,
# ...) since nothing in the retriever actually knew "stol" meant "any of
# these four". When a query's ONLY signal is a bare disambiguation term —
# no other filter at all — rag/retriever.py returns a category breakdown
# (counts per category) instead of guessing which subtype to show
# products from; see RetrievalResult.category_breakdown.
CATEGORY_DISAMBIGUATION_TERMS: dict[str, set[str]] = {
    "stol": {"office_chair", "dining_chair", "bar_stool", "armchair"},
    "stole": {"office_chair", "dining_chair", "bar_stool", "armchair"},
    # "bord" (table) is the shared head-noun of four specific compound
    # categories (spisebord/dining_table, sofabord+kaffebord/coffee_table,
    # sidebord/side_table, konsolbord/console_table) — same situation as
    # "stol" above. Deliberately not including tv_stand ("tv-bord") or
    # outdoor_set here: those are a stretch of "table" a customer asking
    # generically for "et bord" isn't likely picturing.
    "bord": {"dining_table", "coffee_table", "side_table", "console_table"},
    "borde": {"dining_table", "coffee_table", "side_table", "console_table"},
    # "sofa" is different from "stol"/"bord": it's ALSO one of the
    # sofa-family's own literal category words (CATEGORY_WORDS["sofa"]),
    # so a bare "sofa" query already matches the plain sofa category
    # directly rather than matching nothing. It's included here anyway —
    # rag/retriever.py's disambiguation check specifically recognizes this
    # "matched only its own bare self, nothing more specific" case — so a
    # customer asking generically for "en sofa" is offered sectional and
    # loveseat too, not just the literal "sofa" category, matching
    # _SOFA_LIKE_CATEGORIES' existing "customers say sofa for any of
    # these" reasoning used elsewhere in the retriever.
    "sofa": {"sofa", "sectional", "loveseat"},
    "sofaer": {"sofa", "sectional", "loveseat"},
}

OUT_OF_STOCK_PHRASES = ("udsolgt", "ikke på lager", "ikke tilgængelig")
IN_STOCK_PHRASES = ("på lager", "tilgængelig", "tilgængelige")
CHEAP_PHRASES = ("billigst", "billigste", "laveste pris")
EXPENSIVE_PHRASES = ("dyrest", "dyreste", "højeste pris")
# Single-substring signals only — reliable regardless of what's in between.
# Word-pair signals ("passer ... til/sammen", "går ... med") that tolerate
# intervening words ("passer GODT sammen") are handled separately in
# retriever.py, since a fixed phrase list can't cover every insertion.
SERIES_INTENT_PHRASES = (
    "matcher", "matchende", "sæt", "serie", "kollektion",
)
SERIES_INTENT_WORD_PAIRS = (
    ("passer", "til"), ("passer", "sammen"), ("går", "med"),
)
DIM_KEYWORDS = {
    "dyb": "depth_cm", "dybde": "depth_cm",
    "bred": "width_cm", "bredde": "width_cm",
    "høj": "height_cm", "højde": "height_cm",
}
# Danish alphabet includes æ/ø/å — plain [a-z] in a tokenizer regex silently
# mangles them, splitting one Danish word into fragments.
WORD_CHARS = "a-zæøå0-9"

for _dir in (
    CATALOG_DIR, RAG_DIR, TRAINING_DIR, EVAL_DIR, RESULTS_DIR,
    CASES_UNRESOLVED_DIR, CASES_RESOLVED_DIR, CASES_RESULTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
