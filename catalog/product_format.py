"""Shared helpers for turning a product dict into text for embedding or prompts.

Used by rag/build_index.py (what gets embedded), rag/retriever.py and the
Gradio app (what gets injected into the system prompt), and the training-data
generator (so training-time context matches inference-time context exactly).

Product shape (see catalog/generate_catalog.py for how it's produced):
    {
      "sku", "name", "short_description", "category",
      "series_id", "series_name",              # nullable — matching-set grouping
      "normal_price", "discount_percent", "discount_price",  # discount_* nullable
      "colors", "material", "dimensions", "weight_kg",
      "rating", "review_count", "warranty_years", "assembly_required", "room",
      "priority",                              # 0-100, business-set — see below
      "availability": {"København": {"stock_quantity", "restock_date"}, "Århus": {...}, "Odense": {...}},
      "attributes": {category-specific fields; irrelevant ones are null},
    }

`priority` (0-100, higher = shown first) is a hidden business ranking used
by rag/retriever.py to pick which products surface when a query matches
more than SHOWN_MAX_RESULTS (config.py) — e.g. "which sectional sofas do
you have" matching 67 products. It must NEVER be rendered into the model's
context (product_to_context_block below deliberately omits it) or exposed
to the customer — it's an internal ranking signal only.

Text fields (name, short_description, colors, material) are in Danish;
category/room are English internal keys translated for display via
CATEGORY_LABELS/ROOM_LABELS below.
"""
from __future__ import annotations

CATEGORY_LABELS = {
    "sofa": "sofa",
    "armchair": "lænestol",
    "loveseat": "to-personers sofa",
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

ROOM_LABELS = {
    "living_room": "stue",
    "bedroom": "soveværelse",
    "dining_room": "spisestue",
    "home_office": "hjemmekontor",
    "outdoor": "udendørs",
    "entryway": "entré",
    "kids_room": "børneværelse",
    "kitchen": "køkken",
}

ATTRIBUTE_LABELS = {
    "seat_height_cm": "Sædehøjde (cm)",
    "seat_depth_cm": "Sædedybde (cm)",
    "weight_capacity_kg": "Vægtkapacitet (kg)",
    "seats_count": "Antal siddepladser",
    "extendable": "Udtrækkelig",
    "bed_size": "Sengestørrelse",
    "num_drawers": "Skuffer",
    "num_shelves": "Hylder",
    "weather_resistant": "Vejrbestandig",
    "bulb_type": "Pæretype",
    "lumens": "Lumen",
    "wattage": "Watt",
    "dimmable": "Dæmpbar",
    "mount_type": "Monteringstype",
    "unit_type": "Elementtype",
    "door_count": "Låger",
}


def _format_kr(amount: float) -> str:
    # Danish number formatting: dot for thousands, comma for decimals.
    return f"{amount:,.2f} kr".replace(",", "X").replace(".", ",").replace("X", ".")


def _price_lines(product: dict) -> str:
    if product.get("discount_price") is not None:
        return (
            f"  Pris: {_format_kr(product['discount_price'])} "
            f"(før {_format_kr(product['normal_price'])}, {product['discount_percent']}% rabat)"
        )
    return f"  Pris: {_format_kr(product['normal_price'])}"


def _availability_lines(product: dict) -> str:
    lines = []
    for store, info in product["availability"].items():
        if info["stock_quantity"] > 0:
            lines.append(f"    {store}: {info['stock_quantity']} på lager")
        else:
            lines.append(f"    {store}: udsolgt (forventes på lager {info['restock_date']})")
    return "\n".join(lines)


def _attribute_lines(product: dict) -> str:
    attrs = product.get("attributes") or {}
    lines = [
        f"  {ATTRIBUTE_LABELS.get(key, key)}: {value}"
        for key, value in attrs.items()
        if value is not None
    ]
    return "\n".join(lines)


def product_to_context_block(product: dict) -> str:
    """Render one product as the compact block injected into the system prompt."""
    dims = product["dimensions"]
    colors = ", ".join(product["colors"])
    series_line = (
        f"  Serie: {product['series_name']} (matcher andre produkter i denne serie)\n"
        if product.get("series_name")
        else ""
    )
    attrs = _attribute_lines(product)
    attrs_block = f"{attrs}\n" if attrs else ""
    category_label = CATEGORY_LABELS.get(product["category"], product["category"])
    room_label = ROOM_LABELS.get(product["room"], product["room"])

    return (
        f"- SKU: {product['sku']}\n"
        f"  Navn: {product['name']}\n"
        f"  Kategori: {category_label}\n"
        f"  Beskrivelse: {product['short_description']}\n"
        f"{series_line}"
        f"{_price_lines(product)}\n"
        f"  Tilgængelige farver: {colors}\n"
        f"  Materiale: {product['material']}\n"
        f"  Mål (B x D x H, cm): {dims['width_cm']} x {dims['depth_cm']} x {dims['height_cm']}\n"
        f"  Vægt: {product['weight_kg']} kg\n"
        f"{attrs_block}"
        f"  Lager pr. butik:\n{_availability_lines(product)}\n"
        f"  Bedømmelse: {product['rating']} ({product['review_count']} anmeldelser)\n"
        f"  Garanti: {product['warranty_years']} år\n"
        f"  Kræver samling: {'ja' if product['assembly_required'] else 'nej'}\n"
        f"  Rum: {room_label}"
    )


def products_to_context(products: list[dict], total_count: int | None = None) -> str:
    """Render a list of products as the full context block for the system prompt.

    `total_count` is the true size of the full matching set before it was
    narrowed to `products` (see rag/retriever.py's RetrievalResult) — when
    it's larger than `len(products)`, a trailing note tells the model more
    matches exist so its prose can acknowledge that. The note is advisory
    only: the actual "see all" link is appended deterministically by
    app/gradio_app.py after generation, not left to the model to construct.
    """
    if not products:
        return "(ingen matchende produkter fundet)"
    blocks = "\n\n".join(product_to_context_block(p) for p in products)
    if total_count and total_count > len(products):
        blocks += (
            f"\n\n(Bemærk: der er i alt {total_count} produkter, der matcher — kun de "
            f"{len(products)} mest relevante er vist her. Fortæl kunden at der findes "
            f"flere, og at de kan se hele listen.)"
        )
    return blocks


def product_to_embedding_text(product: dict) -> str:
    """Render a product as the text that gets embedded for retrieval.

    Kept separate from the context block: this is optimized for semantic
    search (natural language, no field labels) rather than for the model to
    read facts off of.
    """
    dims = product["dimensions"]
    category_label = CATEGORY_LABELS.get(product["category"], product["category"])
    room_label = ROOM_LABELS.get(product["room"], product["room"])
    series_phrase = f" Del af serien {product['series_name']}." if product.get("series_name") else ""
    price = product.get("discount_price") or product["normal_price"]
    return (
        f"{product['name']}. {category_label} til {room_label}.{series_phrase} "
        f"{product['short_description']} "
        f"Fås i {', '.join(product['colors'])}. Lavet af {product['material']}. "
        f"Mål {dims['width_cm']}x{dims['depth_cm']}x{dims['height_cm']} cm. "
        f"Pris {_format_kr(price)}."
    )
