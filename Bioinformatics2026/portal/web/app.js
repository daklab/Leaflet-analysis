// Splicing atlas portal: static, all filtering client-side. Data layout written by build_portal.py.
const $ = (id) => document.getElementById(id);
const slug = (s) => s.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
const MAKO = ["#0B0405", "#382A54", "#395D9C", "#3497A9", "#60CEAC", "#DEF5E5"];   // sequential map used for psi in the paper
let INDEX = null, DS = null, GENE = null, CTMETA = {};

function hex2rgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }
function mako(v) {
  if (v === null || v === undefined) return null;
  const x = Math.min(1, Math.max(0, v)) * (MAKO.length - 1), i = Math.min(MAKO.length - 2, Math.floor(x)), f = x - i;
  const a = hex2rgb(MAKO[i]), b = hex2rgb(MAKO[i + 1]);
  return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(",")})`;
}
const fmt = (v) => (v === null || v === undefined ? "–" : v.toFixed(2));
async function getJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }

async function init() {
  INDEX = await getJSON("data/index.json");
  const dss = Object.keys(INDEX.datasets);
  $("dataset").innerHTML = dss.map((d) => `<option>${d}</option>`).join("");
  $("dataset").onchange = () => selectDataset($("dataset").value);
  $("downloads").innerHTML = INDEX.zenodo ? `<a href="${INDEX.zenodo}">Download the catalog (Zenodo)</a>` : `<span class="hint">Zenodo download: link added at release</span>`;
  document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    $("view-gene").hidden = b.dataset.view !== "gene"; $("view-celltype").hidden = b.dataset.view !== "celltype";
  });
  ["f-tissue", "f-lineage", "f-name", "f-cells", "f-reads", "f-specific"].forEach((id) => $(id).oninput = renderGene);
  $("gene").onchange = () => loadGene($("gene").value.trim());
  $("ct-filter").oninput = fillCellTypes;
  $("ct-show").onclick = showCellTypes;
  selectDataset(dss[0]);
  const q = new URLSearchParams(location.search).get("gene"); if (q) { $("gene").value = q; loadGene(q); }
}

function selectDataset(ds) {
  DS = ds; const D = INDEX.datasets[ds];
  CTMETA = Object.fromEntries(D.cell_types.map((c) => [c.cell_type, c]));
  const pre = D.source.filter((s) => s.includes("PREVIEW"));
  $("banner").hidden = !pre.length;
  $("banner").textContent = pre.length ? `${pre.join("; ")}: preview data for testing the portal, not LeafletFA estimates.` : "";
  $("genelist").innerHTML = D.genes.map((g) => `<option value="${g}">`).join("");
  const uniq = (k) => [...new Set(D.cell_types.map((c) => c[k]))].sort();
  $("f-tissue").innerHTML = `<option value="">all</option>` + uniq("tissue").map((t) => `<option>${t}</option>`).join("");
  $("f-lineage").innerHTML = `<option value="">all</option>` + uniq("lineage").map((t) => `<option>${t}</option>`).join("");
  fillCellTypes(); GENE = null; $("gene-out").innerHTML = `<p class="hint">Type a gene name to see its events.</p>`;
}

async function loadGene(g) {
  if (!g) return;
  try { GENE = await getJSON(`data/genes/${DS}/${slug(g)}.json`); renderGene(); }
  catch (e) { $("gene-out").innerHTML = `<p class="hint">No events for “${g}” in ${DS}.</p>`; }
}

function keep(c) {
  const m = CTMETA[c.ct] || {}, t = $("f-tissue").value, l = $("f-lineage").value, n = $("f-name").value.toLowerCase();
  return (!t || m.tissue === t) && (!l || m.lineage === l) && (!n || c.ct.toLowerCase().includes(n)) &&
    c.cells >= (+$("f-cells").value || 0) && c.reads >= (+$("f-reads").value || 0) && (!$("f-specific").checked || c.specific);
}

function arcs(js) {
  const lo = Math.min(...js.map((j) => j.start)), hi = Math.max(...js.map((j) => j.end)), W = 1000, pad = 20;
  const x = (p) => pad + ((p - lo) / Math.max(1, hi - lo)) * (W - 2 * pad);
  const paths = js.map((j, i) => {
    const a = x(j.start), b = x(j.end), h = 12 + 40 * ((i + 1) / js.length);
    return `<path d="M${a},62 Q${(a + b) / 2},${62 - 2 * h} ${b},62" fill="none" stroke="var(--accent)" stroke-width="2"/>` +
      `<text x="${(a + b) / 2}" y="${66 - h}" font-size="12" text-anchor="middle" fill="currentColor">J${i + 1}</text>`;
  }).join("");
  return `<svg class="arcs" viewBox="0 0 ${W} 70" preserveAspectRatio="none" role="img" aria-label="junction arcs">` +
    `<line x1="${pad}" y1="62" x2="${W - pad}" y2="62" stroke="var(--line)"/>${paths}</svg>`;
}

function renderGene() {
  if (!GENE) return;
  const out = GENE.events.map((e) => {
    const rows = e.cell_types.filter(keep).sort((a, b) => (b.tv ?? 0) - (a.tv ?? 0));
    if (!rows.length) return "";
    const head = e.junctions.map((j, i) => `<th title="${j.id}">J${i + 1}</th>`).join("");
    const sub = (ct) => (CTMETA[ct] || {}).official_subclass ? ` <span class="meta">(${CTMETA[ct].official_subclass})</span>` : "";
    const body = rows.map((c) => `<tr><td class="name">${c.specific ? '<span class="star" title="cell-type-specific">★</span> ' : ""}${c.ct}${sub(c.ct)}</td>` +
      c.psi.map((v, i) => `<td class="psi${v !== null && v > 0.7 ? " light" : ""}" style="background:${mako(v) ?? "transparent"}" ` +
        `title="ψ ${fmt(v)} [${fmt(c.lo[i])}–${fmt(c.hi[i])}]; rest of atlas ${fmt(c.rest[i])}">${fmt(v)}</td>`).join("") +
      `<td>${fmt(c.tv)}</td><td>${c.cells}</td><td>${c.cov}</td><td>${c.reads}</td></tr>`).join("");
    const coords = e.junctions.map((j, i) => `J${i + 1} ${j.chrom}:${j.start}-${j.end} (${j.strand})`).join(" · ");
    return `<div class="event"><h3>${e.event_id}</h3><div class="meta">${e.annotation} · ${coords}</div>${arcs(e.junctions)}` +
      `<table><thead><tr><th class="name">${INDEX.datasets[DS].type_label || "Cell type"}</th>${head}<th title="total variation vs the rest of the atlas">TV</th>` +
      `<th>cells</th><th>covering</th><th>reads</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }).join("");
  $("gene-out").innerHTML = `<h2>${GENE.gene} <span class="meta">${DS}</span></h2>` + (out || `<p class="hint">No cell types pass the filters.</p>`);
}

function fillCellTypes() {
  const f = $("ct-filter").value.toLowerCase();
  $("ct-select").innerHTML = INDEX.datasets[DS].cell_types.filter((c) => c.cell_type.toLowerCase().includes(f))
    .map((c) => `<option value="${c.cell_type}">${c.cell_type}${c.official_subclass ? " [" + c.official_subclass + "]" : ""} (${c.cells})</option>`).join("");
}

let CTROWS = [], SORT = { k: "delta", d: -1 };
async function showCellTypes() {
  const sel = [...$("ct-select").selectedOptions].map((o) => o.value);
  CTROWS = [];
  for (const t of sel) {
    try { (await getJSON(`data/celltypes/${DS}/${slug(t)}.json`)).forEach((r) => CTROWS.push({ ct: t, ...r })); } catch (e) { /* no specific events */ }
  }
  renderCellTypes();
}
function renderCellTypes() {
  const cols = [["ct", "Cell type"], ["gene", "Gene"], ["event_id", "Event"], ["delta", "Δψ"], ["psi", "ψ"], ["rest", "ψ rest"], ["reads", "reads"], ["cov", "covering"]];
  const rows = [...CTROWS].sort((a, b) => (a[SORT.k] > b[SORT.k] ? 1 : a[SORT.k] < b[SORT.k] ? -1 : 0) * SORT.d);
  $("ct-out").innerHTML = rows.length ? `<p class="meta">${rows.length} specific events</p><table><thead><tr>` +
    cols.map(([k, l]) => `<th class="sortable name" data-k="${k}">${l}</th>`).join("") + `</tr></thead><tbody>` +
    rows.map((r) => `<tr><td class="name">${r.ct}</td><td class="name"><a href="?gene=${encodeURIComponent(r.gene)}">${r.gene}</a></td>` +
      `<td class="name">${r.event_id}</td><td>${fmt(r.delta)}</td><td>${fmt(r.psi)}</td><td>${fmt(r.rest)}</td><td>${r.reads}</td><td>${r.cov}</td></tr>`).join("") +
    `</tbody></table>` : `<p class="hint">No events flagged specific in the selected cell types.</p>`;
  document.querySelectorAll("#ct-out th.sortable").forEach((th) => th.onclick = () => {
    SORT = { k: th.dataset.k, d: SORT.k === th.dataset.k ? -SORT.d : -1 }; renderCellTypes();
  });
}

init().catch((e) => { document.querySelector("main").innerHTML = `<p>Could not load data/index.json (${e.message}). Serve this folder over http, e.g. python -m http.server.</p>`; });
