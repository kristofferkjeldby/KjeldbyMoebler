const CATEGORY_IMAGES = {
  sofa: "images/products/sofa/default.png",
  armchair: "images/products/laenestol/default.png",
  loveseat: "images/products/to-personers_sofa/default.png",
  sectional: "images/products/hjoernesofa/default.png",
  dining_table: "images/products/spisebord/default.png",
  coffee_table: "images/products/sofabord/default.png",
  side_table: "images/products/sidebord/default.png",
  console_table: "images/products/konsolbord/default.png",
  bed_frame: "images/products/sengeramme/default.png",
  nightstand: "images/products/natbord/default.png",
  dresser: "images/products/komode/default.png",
  wardrobe: "images/products/garderobeskab/default.png",
  bookshelf: "images/products/bogreol/default.png",
  desk: "images/products/skrivebord/default.png",
  office_chair: "images/products/kontorstol/default.png",
  dining_chair: "images/products/spisebordstol/default.png",
  bar_stool: "images/products/barstol/default.png",
  tv_stand: "images/products/tv-bord/default.png",
  outdoor_set: "images/products/havemoeblesaet/default.png",
  rug: "images/products/taeppe/default.png",
  lighting: "images/products/belysning/default.png",
  kitchen_unit: "images/products/koekkenelement/default.png",
};
const CATEGORY_LABELS = {
  sofa: "Sofa", armchair: "Lænestol", loveseat: "To-personers sofa", sectional: "Hjørnesofa",
  dining_table: "Spisebord", coffee_table: "Sofabord", side_table: "Sidebord", console_table: "Konsolbord",
  bed_frame: "Sengeramme", nightstand: "Natbord", dresser: "Kommode", wardrobe: "Garderobeskab",
  bookshelf: "Bogreol", desk: "Skrivebord", office_chair: "Kontorstol", dining_chair: "Spisebordsstol",
  bar_stool: "Barstol", tv_stand: "TV-bord", outdoor_set: "Havemøbelsæt", rug: "Tæppe",
  lighting: "Belysning", kitchen_unit: "Køkkenelement",
};

let PRODUCTS = {};
const PRODUCTS_READY = fetch("data/products_by_sku.json")
  .then(r => r.json())
  .then(data => { PRODUCTS = data; });

let IMAGE_MANIFEST = {};
fetch("data/image_manifest.json")
  .then(r => r.json())
  .then(data => { IMAGE_MANIFEST = data; });

// Mirrors catalog/image_service.py's slugify() exactly — must stay in
// sync, since this is how the frontend looks up the same per-color
// filenames the Python-side manifest generator produced.
function slugifyColor(text) {
  return text.toLowerCase()
    .replace(/æ/g, "ae").replace(/ø/g, "oe").replace(/å/g, "aa")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

// Product+color -> product default -> category default, same fallback
// order as the Python ImageService.
function productImage(sku, category, color) {
  const entry = IMAGE_MANIFEST[sku];
  if (entry) {
    if (color) {
      const colorPath = entry.colors[slugifyColor(color)];
      if (colorPath) return "images/" + colorPath;
    }
    if (entry.default) return "images/" + entry.default;
  }
  return CATEGORY_IMAGES[category] || "";
}

const fmtKr = n => Math.round(n).toLocaleString("da-DK") + " kr.";

function effectivePrice(p) {
  return p.discount_price || p.normal_price;
}

// --- shared sortable product table -----------------------------------
//
// Renders `skus` into `tbodyEl` as sortable product rows, keeping every
// caller (the category modal, and app/static/search.js's results page)
// on one implementation instead of each maintaining its own copy of the
// same row-building/sorting logic. `sortState` is mutated by the caller
// between renders (e.g. on a header click), not by this function.
function renderProductTable(opts) {
  const { key, dir } = opts.sortState;
  const sorted = [...opts.skus].sort((a, b) => {
    const pa = PRODUCTS[a], pb = PRODUCTS[b];
    let va, vb;
    if (key === "price") { va = effectivePrice(pa); vb = effectivePrice(pb); }
    else if (key === "rating") { va = pa.rating || 0; vb = pb.rating || 0; }
    else { va = (pa[key] || "").toString().toLowerCase(); vb = (pb[key] || "").toString().toLowerCase(); }
    if (va < vb) return -1 * dir;
    if (va > vb) return 1 * dir;
    return 0;
  });

  if (opts.countEl) opts.countEl.textContent = sorted.length + " produkter";

  if (opts.headEls) {
    opts.headEls.forEach(th => {
      th.classList.toggle("sort-active", th.dataset.sort === key);
      th.querySelector(".sort-arrow").textContent = th.dataset.sort === key ? (dir === 1 ? "▲" : "▼") : "▲";
    });
  }

  opts.tbodyEl.innerHTML = sorted.map(sku => {
    const p = PRODUCTS[sku];
    const img = productImage(sku, p.category, null);
    const priceHtml = p.discount_percent
      ? fmtKr(p.discount_price) + ' <span class="was">' + fmtKr(p.normal_price) + "</span>"
      : fmtKr(p.normal_price);
    return (
      '<tr data-sku="' + sku + '">' +
        '<td><img class="row-thumb" src="' + img + '" alt=""></td>' +
        '<td class="col-name">' + p.name + '</td>' +
        '<td class="col-price">' + priceHtml + '</td>' +
        '<td>' + (p.colors || []).join(", ") + '</td>' +
        '<td>' + (p.material || "") + '</td>' +
        '<td>' + (p.rating ? p.rating + " ★" : "") + '</td>' +
      '</tr>'
    );
  }).join("");

  opts.tbodyEl.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => (opts.onRowClick || openProductModal)(tr.dataset.sku));
  });

  return sorted;
}

// --- category table modal -------------------------------------------

let categoryModalSkus = [];
let categorySort = { key: "name", dir: 1 };

// colorFilter (optional): array of catalog color strings from the chat's
// "[Se alle N <kategori> ->]" link — when the chat narrowed by color
// ("10 red office chairs"), the link carries those exact colors so the
// modal opens pre-filtered to the same 10, instead of the full
// unfiltered category the label would otherwise misleadingly promise.
// Matched case-insensitively since the retriever's own color detection
// is also case-insensitive.
function openCategoryModal(category, colorFilter) {
  const filterLower = colorFilter && colorFilter.length ? colorFilter.map(c => c.toLowerCase()) : null;
  categoryModalSkus = Object.keys(PRODUCTS).filter(sku => {
    const p = PRODUCTS[sku];
    if (p.category !== category) return false;
    if (!filterLower) return true;
    return (p.colors || []).some(c => filterLower.includes(c.toLowerCase()));
  });
  categorySort = { key: "name", dir: 1 };
  const label = CATEGORY_LABELS[category] || category;
  const titleEl = document.getElementById("category-modal-title");
  const clearBtnHtml = filterLower
    ? ' <button class="filter-clear-btn" id="category-modal-clear-filter" type="button">✕ Nulstil farvefilter</button>'
    : "";
  titleEl.innerHTML = filterLower
    ? label.charAt(0).toUpperCase() + label.slice(1) + " — " + colorFilter.join(", ") + clearBtnHtml
    : label.charAt(0).toUpperCase() + label.slice(1);
  if (filterLower) {
    document.getElementById("category-modal-clear-filter").addEventListener("click", () => openCategoryModal(category));
  }
  renderCategoryTable();
  document.getElementById("category-modal").classList.add("open");
}

function renderCategoryTable() {
  renderProductTable({
    skus: categoryModalSkus,
    sortState: categorySort,
    tbodyEl: document.getElementById("category-modal-tbody"),
    countEl: document.getElementById("category-modal-count"),
    headEls: document.querySelectorAll("#category-modal th.sortable"),
  });
}

document.querySelectorAll("#category-modal th.sortable").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    categorySort = categorySort.key === key ? { key, dir: -categorySort.dir } : { key, dir: 1 };
    renderCategoryTable();
  });
});

function closeCategoryModal() {
  document.getElementById("category-modal").classList.remove("open");
}

document.getElementById("category-modal-close").addEventListener("click", closeCategoryModal);
document.getElementById("category-modal").addEventListener("click", e => {
  if (e.target.id === "category-modal") closeCategoryModal();
});
document.querySelectorAll(".cat-card[data-category]").forEach(el => {
  el.addEventListener("click", () => openCategoryModal(el.dataset.category));
});

function openProductModal(sku, color) {
  const p = PRODUCTS[sku];
  if (!p) return;

  document.getElementById("modal-img").src = productImage(sku, p.category, color || null);
  document.getElementById("modal-cat").textContent = CATEGORY_LABELS[p.category] || p.category;
  document.getElementById("modal-name").textContent = p.name;
  document.getElementById("modal-desc").textContent = p.short_description || "";

  const priceEl = document.getElementById("modal-price");
  if (p.discount_percent) {
    priceEl.innerHTML = fmtKr(p.discount_price) + ' <span class="was">' + fmtKr(p.normal_price) + "</span>";
  } else {
    priceEl.textContent = fmtKr(p.normal_price);
  }

  const dims = p.dimensions || {};
  const facts = [
    ["SKU", sku],
    ["Mål (B×D×H)", [dims.width_cm, dims.depth_cm, dims.height_cm].filter(Boolean).join(" × ") + " cm"],
    ["Materiale", p.material],
    ["Farver", (p.colors || []).join(", ")],
    ["Vægt", p.weight_kg ? p.weight_kg + " kg" : ""],
    ["Garanti", p.warranty_years ? p.warranty_years + " år" : ""],
    ["Bedømmelse", p.rating ? p.rating + " ★ (" + p.review_count + " anmeldelser)" : ""],
    ["Samling påkrævet", p.assembly_required === true ? "Ja" : p.assembly_required === false ? "Nej" : ""],
  ];
  document.getElementById("modal-facts").innerHTML = facts
    .filter(([, v]) => v)
    .map(([k, v]) => '<div><div class="fact-label">' + k + '</div><div class="fact-value">' + v + "</div></div>")
    .join("");

  const stockList = document.getElementById("modal-stock-list");
  stockList.innerHTML = Object.entries(p.availability || {})
    .map(([store, info]) => {
      const inStock = info.stock_quantity > 0;
      const status = inStock ? info.stock_quantity + " på lager" : "Udsolgt" + (info.restock_date ? " (forventes " + info.restock_date + ")" : "");
      return '<li><span>' + store + '</span><span class="' + (inStock ? "in-stock" : "out-stock") + '">' + status + "</span></li>";
    })
    .join("");

  const cartBtn = document.getElementById("modal-cart-btn");
  cartBtn.textContent = "Tilføj til kurv";
  cartBtn.classList.remove("added");

  document.getElementById("product-modal").classList.add("open");
}

document.getElementById("modal-cart-btn").addEventListener("click", function () {
  this.textContent = "Tilføjet til kurv ✓";
  this.classList.add("added");
});

function closeProductModal() {
  document.getElementById("product-modal").classList.remove("open");
}

document.getElementById("modal-close").addEventListener("click", closeProductModal);
document.getElementById("product-modal").addEventListener("click", e => {
  if (e.target.id === "product-modal") closeProductModal();
});
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  closeProductModal();
  closeCategoryModal();
});

document.querySelectorAll(".series-card li .pname[data-sku]").forEach(el => {
  el.addEventListener("click", () => openProductModal(el.dataset.sku));
});

// The chat lives in a separate-origin iframe (Gradio on :7860), so it can't
// reach into this page's DOM directly — it posts a message instead when a
// product link is clicked, and we open the modal here.
// Products the customer has "added to basket" stay pinned in the focus
// panel even once the chat moves on to a different topic and the model's
// own retrieval focus changes — this is purely a display concern, it
// never gets sent back to the chat/RAG side.
let basketProducts = {};
let liveFocusProducts = [];

window.addEventListener("message", e => {
  if (e.data && e.data.type === "kjeldby-open-product" && e.data.sku) {
    openProductModal(e.data.sku);
  }
  if (e.data && e.data.type === "kjeldby-open-category" && e.data.category) {
    openCategoryModal(e.data.category, e.data.colors || null);
  }
  if (e.data && e.data.type === "kjeldby-focus-update" && Array.isArray(e.data.products)) {
    liveFocusProducts = e.data.products;
    renderFocusPanel();
  }
});

// Display only — lets the customer see filtering would be possible here,
// but nothing is actually wired to filter the list yet. The focus panel
// only exists on landing_page.html (search.html has no chat/focus UI),
// so these are guarded rather than assumed present — this file is shared
// between both pages.
document.getElementById("focus-filters-toggle")?.addEventListener("click", () => {
  document.getElementById("focus-filters-panel").classList.toggle("open");
});
document.addEventListener("click", e => {
  const wrap = document.getElementById("focus-filters");
  if (wrap && !wrap.contains(e.target)) {
    document.getElementById("focus-filters-panel").classList.remove("open");
  }
});

function updateFocusFilters(products) {
  const wrap = document.getElementById("focus-filters");
  wrap.style.display = products.length > 1 ? "block" : "none";
  if (products.length <= 1) return;

  const colorSelect = document.getElementById("focus-filter-color");
  const colors = [...new Set(products.map(p => p.selected_color).filter(Boolean))];
  colorSelect.innerHTML = '<option>Alle farver</option>' + colors.map(c => `<option>${c}</option>`).join("");
}

function renderFocusPanel() {
  const list = document.getElementById("focus-list");
  const seen = new Set();
  const combined = [];
  for (const p of liveFocusProducts) {
    seen.add(p.sku);
    combined.push(p);
  }
  for (const sku in basketProducts) {
    if (!seen.has(sku)) combined.push(basketProducts[sku]);
  }

  updateFocusFilters(combined);

  if (!combined.length) {
    list.innerHTML = '<div class="focus-empty">Spørg om et produkt, så dukker det op her.</div>';
    return;
  }
  list.innerHTML = combined.map(p => {
    const img = productImage(p.sku, p.category, p.selected_color);
    const priceHtml = p.discount_percent
      ? fmtKr(p.price) + ' <span class="was">' + fmtKr(p.normal_price) + "</span>"
      : fmtKr(p.price);
    const inBasket = !!basketProducts[p.sku];
    const colorHtml = p.selected_color ? '<div class="focus-color">Farve: ' + p.selected_color + '</div>' : '';
    return (
      '<div class="focus-card" data-sku="' + p.sku + '" data-color="' + (p.selected_color || '') + '">' +
        '<img src="' + img + '" alt="">' +
        '<div class="focus-info">' +
          '<div class="focus-name">' + p.name + '</div>' +
          '<div class="focus-price">' + priceHtml + '</div>' +
          colorHtml +
          '<button class="focus-add-btn' + (inBasket ? ' added' : '') + '" type="button">' +
            (inBasket ? "Tilføjet ✓" : "Læg i kurv") +
          '</button>' +
        '</div>' +
      '</div>'
    );
  }).join("");
  list.querySelectorAll(".focus-card").forEach(el => {
    el.addEventListener("click", () => openProductModal(el.dataset.sku, el.dataset.color));
  });
  list.querySelectorAll(".focus-add-btn").forEach((btn, i) => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const p = combined[i];
      basketProducts[p.sku] = p;
      btn.textContent = "Tilføjet ✓";
      btn.classList.add("added");
    });
  });
}
