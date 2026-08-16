// Talks to /api/autocomplete and /api/search on the Gradio/FastAPI process
// (port 7860) — a genuine cross-origin call from this page (served on
// :8080), allowed via the CORS middleware registered in app/gradio_app.py.
const API_BASE = "http://localhost:7860";

const searchInput = document.getElementById("search-input");
const autocompleteList = document.getElementById("autocomplete-list");
const resultsSection = document.getElementById("results-section");
const disambigSection = document.getElementById("disambig-section");
const resultsSummary = document.getElementById("results-summary");

let searchSort = { key: "name", dir: 1 };
let currentPoolSkus = [];
let debounceTimer = null;

searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  const q = searchInput.value.trim();
  if (!q) {
    hideAutocomplete();
    return;
  }
  debounceTimer = setTimeout(() => fetchAutocomplete(q), 200);
});

function hideAutocomplete() {
  autocompleteList.style.display = "none";
  autocompleteList.innerHTML = "";
}

function fetchAutocomplete(q) {
  fetch(API_BASE + "/api/autocomplete?q=" + encodeURIComponent(q))
    .then(r => r.json())
    .then(items => {
      if (!items.length) {
        hideAutocomplete();
        return;
      }
      autocompleteList.innerHTML = items.map(it =>
        '<div class="autocomplete-item" data-sku="' + it.sku + '">' +
          '<span class="ac-name">' + it.name + '</span>' +
          '<span class="ac-cat">' + (CATEGORY_LABELS[it.category] || it.category) + '</span>' +
        '</div>'
      ).join("");
      autocompleteList.style.display = "block";
      autocompleteList.querySelectorAll(".autocomplete-item").forEach(el => {
        el.addEventListener("click", () => {
          const sku = el.dataset.sku;
          hideAutocomplete();
          PRODUCTS_READY.then(() => openProductModal(sku));
        });
      });
    });
}

document.addEventListener("click", e => {
  if (!e.target.closest("#search-box")) hideAutocomplete();
});

document.getElementById("search-form").addEventListener("submit", e => {
  e.preventDefault();
  hideAutocomplete();
  runSearch(searchInput.value.trim());
});

function runSearch(q) {
  if (!q) return;
  fetch(API_BASE + "/api/search?q=" + encodeURIComponent(q))
    .then(r => r.json())
    .then(result => PRODUCTS_READY.then(() => renderResults(result)));
}

function renderResults(result) {
  if (result.category_breakdown) {
    resultsSection.style.display = "none";
    disambigSection.style.display = "block";
    const entries = Object.entries(result.category_breakdown).sort((a, b) => b[1] - a[1]);
    disambigSection.innerHTML =
      '<p class="results-summary">Ingen specifik kategori genkendt (' + result.total_count +
      ' produkter på tværs af ' + entries.length + ' kategorier) — vælg en for at se hele listen:</p>' +
      '<div class="cat-grid">' +
      entries.map(([cat, count]) =>
        '<div class="cat-card" data-category="' + cat + '">' +
          '<div class="label">' + (CATEGORY_LABELS[cat] || cat) + '</div>' +
          '<div class="count">' + count + ' produkter</div>' +
        '</div>'
      ).join("") +
      '</div>';
    disambigSection.querySelectorAll(".cat-card[data-category]").forEach(el => {
      el.addEventListener("click", () => openCategoryModal(el.dataset.category));
    });
    return;
  }

  disambigSection.style.display = "none";
  resultsSection.style.display = "block";
  currentPoolSkus = result.pool;
  searchSort = { key: "name", dir: 1 };
  // Full pool, not just `shown` — this page is for testing the retriever
  // directly, so seeing every match (including the ones a chat reply would
  // never mention) is the point, not a limitation to work around.
  resultsSummary.textContent = result.shown.length < result.total_count
    ? result.total_count + " match i alt — chatten ville selv vise de " + result.shown.length + " højest prioriterede"
    : result.total_count + " match";
  renderSearchTable();
}

function renderSearchTable() {
  renderProductTable({
    skus: currentPoolSkus,
    sortState: searchSort,
    tbodyEl: document.getElementById("search-results-tbody"),
    headEls: document.querySelectorAll("#search-results-table th.sortable"),
  });
}

document.querySelectorAll("#search-results-table th.sortable").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    searchSort = searchSort.key === key ? { key, dir: -searchSort.dir } : { key, dir: 1 };
    renderSearchTable();
  });
});
