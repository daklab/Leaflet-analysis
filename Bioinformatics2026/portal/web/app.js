// Cell-type splicing browser: static, all computation client-side. Data layout written by build_portal.py (see README).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const NA = 65535, SCALE = 10000;
const num = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-US"));
const f2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(2));
const f4 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "NA" : v.toFixed(4));
const dec = (x) => (x === NA ? NaN : x / SCALE);
let INDEX, CAT, JUNC;
const cache = new Map();

async function getBuf(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.arrayBuffer(); }));
  return cache.get(path);
}
async function getJSON(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }));
  return cache.get(path);
}

const isGrey = (t) => (CONFIG.GREY_RULE === "n_mice_lt2" ? t.n_mice < 2 : !t.eligible);
function greyReason(t) {
  if (t.n_mice < 2) return `fewer than 2 mice (${t.n_mice})`;
  return `${t.n_mice} mice, but no TRAIN cells in one of the two frozen mouse halves`;
}
const typeLabel = (ds) => (ds === "Allen" ? "cell group" : "cell type");
const typesLabel = (ds) => (ds === "Allen" ? "cell groups (cluster names)" : "cell types");
function typeByName(ds, name) { return INDEX.datasets[ds].types.find((t) => t.name === name); }
const DARK_MQ = window.matchMedia("(prefers-color-scheme: dark)");
const isDark = () => { const t = document.documentElement.dataset.theme; return t ? t === "dark" : DARK_MQ.matches; };
function jcolor(i) { const c = isDark() ? CONFIG.JUNCTION_COLORS_DARK : CONFIG.JUNCTION_COLORS; return i < c.length ? c[i] : CONFIG.JUNCTION_OVERFLOW_COLOR; }
const eventIdx = (id) => CAT.byId.get(id);
function eventInfo(e) {
  const jo = CAT.jo[e], jn = CAT.jo[e + 1] - jo, g = CAT.gene_idx[e];
  return { e, id: CAT.ids[e], gene: CAT.genes[g], gene_id: CAT.gene_ids[g], cls: CAT.classes[CAT.cls[e]], chrom: CAT.chrom[CAT.chrom_idx[e]], strand: CAT.strand[e] ? "-" : "+", jo, jn };
}
const jStart = (j) => JUNC.start[j], jEnd = (j) => JUNC.end[j];
const jCoord = (j, chrom) => `${chrom}:${jStart(j).toLocaleString("en-US")}-${jEnd(j).toLocaleString("en-US")}`;
const jId = (j, ev) => `${ev.chrom}_${jStart(j)}_${jEnd(j)}_${ev.strand}`;

function setPage(html) { $("main").innerHTML = html; }
function dsToggle(cur, hrefFn) {
  return `<span class="ds" role="group" aria-label="Dataset">` + Object.keys(INDEX.datasets).map((d) =>
    `<a class="${d === cur ? "on" : ""}" href="${hrefFn(d)}"><span class="swatch" style="background:${CONFIG.DATASET_COLOR[d]}"></span>${d}</a>`).join("") + `</span>`;
}

/* ---------- search ---------- */
function searchHits(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  const out = [];
  if (q.startsWith("ensmus") || q.includes("atse")) {
    for (let e = 0; e < CAT.ids.length && out.length < 15; e++) if (CAT.idsLower[e].includes(q)) out.push({ href: `#/event/${curDS()}/${encodeURIComponent(CAT.ids[e])}`, label: CAT.ids[e], sub: CAT.genes[CAT.gene_idx[e]] });
    return out;
  }
  const exact = [], pre = [];
  CAT.genesLower.forEach((g, i) => { if (g === q) exact.push(i); else if (g.startsWith(q)) pre.push(i); });
  for (const i of [...exact, ...pre].slice(0, 15)) out.push({ href: `#/gene/${encodeURIComponent(CAT.genes[i])}`, label: CAT.genes[i], gene: true, sub: `${CAT.gene_events[i].length} ATSE${CAT.gene_events[i].length > 1 ? "s" : ""}` });
  return out;
}
let lastDS = "TMS";
const curDS = () => lastDS;
function initSearch() {
  const box = $("q"), res = $("q-results");
  let sel = -1;
  const show = () => {
    const hits = searchHits(box.value); sel = -1;
    res.hidden = !box.value.trim();
    res.innerHTML = hits.length ? hits.map((h) => `<a href="${h.href}">${h.gene ? `<span class="gene">${esc(h.label)}</span>` : `<span class="mono">${esc(h.label)}</span>`} <span class="meta">${esc(h.sub)}</span></a>`).join("") : `<a class="meta">No gene or ATSE matches.</a>`;
  };
  box.oninput = show;
  box.onkeydown = (ev) => {
    const items = [...res.querySelectorAll("a[href]")];
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault(); if (!items.length) return;
      sel = (sel + (ev.key === "ArrowDown" ? 1 : items.length - 1 + (sel < 0 ? 1 : 0))) % items.length;
      items.forEach((a, i) => a.classList.toggle("sel", i === sel));
    } else if (ev.key === "Enter" && items.length) { location.hash = items[Math.max(0, sel)].getAttribute("href"); res.hidden = true; box.blur(); }
    else if (ev.key === "Escape") res.hidden = true;
  };
  res.onclick = () => { res.hidden = true; };
  document.addEventListener("click", (ev) => { if (!ev.target.closest(".search")) res.hidden = true; });
}

/* ---------- landing ---------- */
function viewHome() {
  const D = INDEX.datasets;
  const stats = Object.entries(D).map(([d, x]) => `<div class="stat"><span class="swatch" style="background:${CONFIG.DATASET_COLOR[d]}"></span><div><b>${d}</b>: ${num(x.types.length)} ${typesLabel(d)}<br><span class="meta">${num(x.train_cells)} TRAIN cells; seed 0, K = 20</span></div></div>`).join("");
  const cards = INDEX.featured.map((f) => `<div class="card"><h3><span class="gene">${esc(f.gene)}</span> <span class="meta">${esc(f.title)}</span></h3><p>${esc(f.text)}</p><p class="links">` +
    f.events.map((e) => `<a class="mono" href="#/event/${f.dataset}/${encodeURIComponent(e)}">${esc(e)}</a> <span class="meta">${CONFIG.CLASS_LABEL[eventInfo(eventIdx(e)).cls]}, ${f.dataset}</span>`).join("<br>") + `</p></div>`).join("");
  const ex = INDEX.example_compare;
  setPage(`<h1>Cell-type splicing browser</h1>
  <p>Junction usage (ψ) per alternative transcript structure event (ATSE), estimated with LeafletFA, next to raw TRAIN pseudobulk. ${num(CAT.ids.length)} ATSEs, ${num(JUNC.start.length)} junctions, ${num(CAT.genes.length)} genes.</p>
  <div class="stats">${stats}</div>
  <h2>Start here</h2>
  <p><a href="#/compare/Allen/${encodeURIComponent(ex.a)}/${encodeURIComponent(ex.b)}">Compare two</a> ${typesLabel("Allen")} or cell types (example: ${esc(ex.a)} vs. ${esc(ex.b)}), or rank ATSEs for <a href="#/one/TMS">one vs. all others</a>. Differences are point estimates with intervals, for exploration.</p>
  <h2>Featured examples</h2><div class="cards">${cards}</div>`);
}

/* ---------- gene page ---------- */
function viewGene(name) {
  const i = CAT.genes.indexOf(name);
  if (i < 0) return setPage(`<p class="hint">No gene named “${esc(name)}” in the catalog.</p>`);
  const rows = CAT.gene_events[i].map((e) => {
    const v = eventInfo(e); let lo = Infinity, hi = 0;
    for (let j = v.jo; j < v.jo + v.jn; j++) { lo = Math.min(lo, jStart(j)); hi = Math.max(hi, jEnd(j)); }
    return `<tr><td class="l mono">${esc(v.id)}</td><td class="l">${CONFIG.CLASS_LABEL[v.cls]}</td><td>${v.jn}</td><td class="l mono">${v.chrom}:${lo.toLocaleString("en-US")}-${hi.toLocaleString("en-US")} (${v.strand})</td><td class="l">` +
      Object.keys(INDEX.datasets).map((d) => `<a href="#/event/${d}/${encodeURIComponent(v.id)}">${d}</a>`).join(" · ") + `</td></tr>`;
  }).join("");
  setPage(`<h1><span class="gene">${esc(name)}</span> <span class="meta">${esc(CAT.gene_ids[i])}</span></h1>
  <div class="wrap"><table><thead><tr><th class="l">ATSE</th><th class="l">Event class</th><th>Junctions</th><th class="l">Span</th><th class="l">Open</th></tr></thead><tbody>${rows}</tbody></table></div>`);
}

/* ---------- event page ---------- */
async function loadEvent(ds, e) {
  const D = INDEX.datasets[ds], B = INDEX.bucket_size, b = Math.floor(e / B), T = D.types.length;
  const e0 = b * B, e1 = Math.min(CAT.ids.length, e0 + B), j0 = CAT.jo[e0], j1 = CAT.jo[e1], nj = j1 - j0, ne = e1 - e0;
  const [buf, st] = await Promise.all([getBuf(`data/${ds}/events/${b}.bin`), getJSON(`data/catalog/struct/${b}.json`)]);
  let off = 0;
  const take = (Arr, n) => { const a = new Arr(buf, off, n); off += n * Arr.BYTES_PER_ELEMENT; return a; };
  const jreads = take(Uint32Array, T * nj), ereads = take(Uint32Array, T * ne), treads = take(Uint32Array, T * ne);
  const psi = take(Uint16Array, T * nj), lo = take(Uint16Array, T * nj), hi = take(Uint16Array, T * nj), raw = take(Uint16Array, T * nj);
  const cov = take(Uint16Array, T * ne), mice = take(Uint16Array, T * ne), tcells = take(Uint16Array, T * ne);
  if (off !== buf.byteLength) throw new Error(`bucket ${b}: expected ${off} bytes, got ${buf.byteLength}`);
  const v = eventInfo(e), jl = v.jo - j0, el = e - e0;
  return {
    v, D, exons: st[CAT.ids[e]],
    psi: (t, k) => dec(psi[t * nj + jl + k]), lo: (t, k) => dec(lo[t * nj + jl + k]), hi: (t, k) => dec(hi[t * nj + jl + k]), raw: (t, k) => dec(raw[t * nj + jl + k]),
    jreads: (t, k) => jreads[t * nj + jl + k], reads: (t) => ereads[t * ne + el], cov: (t) => cov[t * ne + el], mice: (t) => mice[t * ne + el],
    treads: (t) => treads[t * ne + el], tcells: (t) => tcells[t * ne + el],
  };
}

function structureSVG(ev, exons) {
  const n = ev.jn, W = Math.max(340, Math.min(1400, $("main").clientWidth - 40));
  const J = []; for (let k = 0; k < n; k++) J.push({ k, l: jStart(ev.jo + k) + 1, r: jEnd(ev.jo + k) + 1 });   // half-open exon coordinates
  const ex = exons.map(([a, b, tx, kind]) => ({ l: a, r: b + 1, tx, kind }));
  const pts = [...new Set([...J.flatMap((j) => [j.l, j.r]), ...ex.flatMap((x) => [x.l, x.r])])].sort((a, b) => a - b);
  const inExon = (u, v) => ex.some((x) => x.l <= u && x.r >= v);
  const w = []; for (let i = 0; i + 1 < pts.length; i++) { const len = pts[i + 1] - pts[i]; w.push(inExon(pts[i], pts[i + 1]) ? 10 + Math.sqrt(len) : 8 + 7 * Math.log10(len + 1)); }
  const pad = 34, tot = w.reduce((a, b) => a + b, 0), sc = (W - 2 * pad) / tot, pos = new Map();
  let c = pad; pts.forEach((p, i) => { pos.set(p, c); if (i < w.length) c += w[i] * sc; });
  const x = (p) => pos.get(p);
  const order = [...J].sort((a, b) => (x(a.r) - x(a.l)) - (x(b.r) - x(b.l)));
  const step = Math.max(13, Math.min(26, 150 / Math.max(1, n)));
  order.forEach((j, i) => { j.h = 18 + i * step; });
  const top = 18 + (n - 1) * step + 16, y0 = top + 4;
  // exon lanes: greedy, non-overlapping within a lane
  const lanes = [], laneH = 14;
  for (const e of [...ex].sort((a, b) => a.kind - b.kind || a.l - b.l)) {
    let i = lanes.findIndex((L) => L.every((o) => o.r <= e.l || o.l >= e.r));
    if (i < 0) { lanes.push([]); i = lanes.length - 1; }
    lanes[i].push(e); e.lane = i;
  }
  const narrow = W < 600, nl = Math.max(1, lanes.length), H = y0 + nl * (laneH + 3) + (narrow ? 40 : 26);
  const leftEnds = new Set(ex.filter((e) => e.kind === 0).map((e) => e.r)), rightEnds = new Set(ex.filter((e) => e.kind === 0).map((e) => e.l));
  let s = `<svg class="structure" viewBox="0 0 ${W} ${H}" role="img" aria-label="Exons as boxes and junctions as arcs for ${esc(ev.id)}">`;
  s += `<line x1="${pad - 20}" x2="${W - pad + 20}" y1="${y0 + laneH / 2}" y2="${y0 + laneH / 2}" stroke="var(--line)"/>`;
  for (const e of ex) {
    const y = y0 + e.lane * (laneH + 3), a = x(e.l), wd = Math.max(2, x(e.r) - a);
    const fill = e.kind === 0 ? "var(--muted)" : "none", stroke = e.kind === 0 ? "var(--muted)" : "var(--muted)";
    s += `<rect x="${a}" y="${y}" width="${wd}" height="${laneH}" rx="1" fill="${fill}" stroke="${stroke}" ${e.kind ? 'stroke-dasharray="3 2"' : ""}><title>exon ${e.l.toLocaleString("en-US")}-${(e.r - 1).toLocaleString("en-US")}, ${e.r - e.l} nt; ${e.tx} GENCODE vM19 transcript${e.tx > 1 ? "s" : ""}${e.kind ? "; no catalog junction ends at it" : ""}</title></rect>`;
    if (e.lane > 0) { const dot = `stroke="var(--line)" stroke-dasharray="2 2"`; s += `<line x1="${a}" x2="${a}" y1="${y0}" y2="${y}" ${dot}/><line x1="${a + wd}" x2="${a + wd}" y1="${y0}" y2="${y}" ${dot}/>`; }
  }
  for (const j of J) {                                                     // ends with no annotated exon: short open stub
    if (!leftEnds.has(j.l)) s += `<rect x="${x(j.l) - 9}" y="${y0}" width="9" height="${laneH}" fill="var(--bg)" stroke="var(--muted)" stroke-width="0.8"><title>no GENCODE vM19 exon ends here</title></rect>`;
    if (!rightEnds.has(j.r)) s += `<rect x="${x(j.r)}" y="${y0}" width="9" height="${laneH}" fill="var(--bg)" stroke="var(--muted)" stroke-width="0.8"><title>no GENCODE vM19 exon starts here</title></rect>`;
  }
  for (const j of order.slice().reverse()) {
    const a = x(j.l), b = x(j.r), col = jcolor(j.k), jx = ev.jo + j.k;
    s += `<path d="M${a},${y0} C${a},${y0 - j.h * 1.33} ${b},${y0 - j.h * 1.33} ${b},${y0}" fill="none" stroke="${col}" stroke-width="1.8"><title>J${j.k + 1} ${jCoord(jx, ev.chrom)}</title></path>`;
  }
  for (const j of order) s += `<text class="jl" x="${(x(j.l) + x(j.r)) / 2}" y="${y0 - j.h + 4}" text-anchor="middle">J${j.k + 1}</text>`;
  const by = H - (narrow ? 22 : 8);
  s += `<text x="${pad - 20}" y="${by}" fill="var(--muted)">${pts[0].toLocaleString("en-US")}</text><text x="${W - pad + 20}" y="${by}" text-anchor="end">${pts[pts.length - 1].toLocaleString("en-US")}</text>`;
  s += `<text x="${W / 2}" y="${narrow ? H - 6 : by}" text-anchor="middle">${ev.strand === "+" ? "transcription 5′ → 3′" : "transcription 3′ ← 5′"}; exons and introns not to scale</text>`;
  return s + `</svg>`;
}

const groupKeys = (ds) => (ds === "Allen" ? [["group1", "Class"], ["group2", "Official subclass"]] : [["group1", "Tissue"], ["group2", "Cell type"]]);
function groupOrder(ds, key, names) {
  const pref = key === "group1" && ds === "Allen" ? ["excitatory", "inhibitory", "non-neuronal"] : null;
  return [...names].sort((a, b) => (pref ? pref.indexOf(a) - pref.indexOf(b) : a.localeCompare(b)));
}

async function viewEvent(ds, id, hl) {
  const e = eventIdx(id);
  if (e === undefined) return setPage(`<p class="hint">No ATSE “${esc(id)}” in the catalog.</p>`);
  lastDS = ds;
  const E = await loadEvent(ds, e), v = E.v, T = E.D.types;
  const partners = CAT.partners[id] || [];
  const juncRows = Array.from({ length: v.jn }, (_, k) => { const j = v.jo + k; return `<tr><td class="l"><span class="swatch" style="background:${jcolor(k)}"></span> J${k + 1}</td><td class="l mono">${jCoord(j, v.chrom)}</td><td>${(jEnd(j) - jStart(j)).toLocaleString("en-US")}</td><td class="l">${CONFIG.ANNOTATION_LABEL[JUNC.status[j]]}</td><td class="l mono">${jId(j, v)}</td></tr>`; }).join("");
  const hlSet = new Set(hl || []);
  const state = { group: "group1", minReads: 1, name: "" };
  setPage(`<h1><span class="gene"><a href="#/gene/${encodeURIComponent(v.gene)}">${esc(v.gene)}</a></span> <span class="mono">${esc(v.id)}</span> ${dsToggle(ds, (d) => `#/event/${d}/${encodeURIComponent(id)}`)}</h1>
  <p class="meta">${CONFIG.CLASS_LABEL[v.cls]} · ${v.jn} junctions · ${v.chrom} (${v.strand}) · ${esc(v.gene_id)} · GENCODE vM19${partners.length ? ` · MXE-style pair split across ATSEs; other half: ${partners.map((p) => `<a class="mono" href="#/event/${ds}/${encodeURIComponent(p)}">${esc(p)}</a>`).join(", ")}` : ""}</p>
  ${structureSVG(v, E.exons)}
  <p class="meta">Filled box: GENCODE vM19 exon at a junction end. Dashed box: annotated exon inside the span with no catalog junction ending at it. Small open box: junction end with no annotated exon.</p>
  <div class="wrap" style="display:inline-block"><table><thead><tr><th class="l">Junction</th><th class="l">Coordinates</th><th>Length (nt)</th><th class="l">Annotation</th><th class="l">Catalog id</th></tr></thead><tbody>${juncRows}</tbody></table></div>
  <h2>ψ per ${typeLabel(ds)} for each junction</h2>
  <div class="controls">
    <label>Group by <select id="ev-group">${groupKeys(ds).map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select></label>
    <label>${ds === "Allen" ? "Cell group" : "Cell type"} contains <input id="ev-name" size="16"></label>
    <label>Min ATSE reads (0 also shows types with no reads) <input id="ev-reads" type="number" min="0" value="1"></label>
    <button id="ev-dl">Download TSV</button>
  </div>
  ${dotLegend(ds)}
  <div id="ev-plot"></div>`);
  const draw = () => { $("ev-plot").innerHTML = dotTable(ds, E, state, hlSet); };
  $("ev-group").onchange = (x) => { state.group = x.target.value; draw(); };
  $("ev-name").oninput = (x) => { state.name = x.target.value.toLowerCase(); draw(); };
  $("ev-reads").oninput = (x) => { state.minReads = +x.target.value || 0; draw(); };
  $("ev-dl").onclick = () => downloadEvent(ds, E);
  draw();
  attachDotTip(ds, E);
}

function dotLegend(ds) {
  const r = (reads) => rawR(reads);
  const circ = (reads) => `<svg width="${2 * r(reads) + 3}" height="16"><circle cx="${r(reads) + 1.5}" cy="8" r="${r(reads)}" fill="none" stroke="var(--raw)" stroke-width="1"/></svg>`;
  const grey = INDEX.datasets[ds].types.filter(isGrey);
  return `<div class="legend"><span><svg width="34" height="12"><line x1="2" x2="32" y1="6" y2="6" stroke="var(--ochre)" stroke-width="2"/><circle cx="17" cy="6" r="3.6" fill="var(--ochre)"/></svg> LeafletFA estimate with ${esc(CONFIG.INTERVAL_LABEL)}</span>
  <span>raw TRAIN pseudobulk, sized by junction reads: ${circ(10)} 10 ${circ(1000)} 1,000 ${circ(100000)} 100,000</span>
  <span><svg width="34" height="12"><line x1="2" x2="32" y1="6" y2="6" stroke="var(--grey)" stroke-width="2"/><circle cx="17" cy="6" r="3.6" fill="var(--grey)"/></svg> grey: ${greyRuleText()} (${grey.length} here)</span></div>`;
}
const greyRuleText = () => (CONFIG.GREY_RULE === "n_mice_lt2" ? "fewer than 2 mice" : "fewer than 2 mice, or no TRAIN cells in one of the two frozen mouse halves");
const rawR = (reads) => (reads <= 0 ? 0 : 1.4 + 0.9 * Math.log10(reads + 1));

function dotTable(ds, E, st, hlSet) {
  const v = E.v, T = E.D.types, PW = 132, RH = 15;
  const rows = T.map((t, i) => ({ t, i })).filter(({ t, i }) => E.reads(i) >= st.minReads && (!st.name || t.name.toLowerCase().includes(st.name) || t.label.toLowerCase().includes(st.name)));
  if (!rows.length) return `<p class="hint">No ${typesLabel(ds)} pass these filters.</p>`;
  const groups = new Map();
  for (const r of rows) { const g = r.t[st.group] || "unassigned"; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(r); }
  const X = (p) => 6 + p * (PW - 12);
  const axis = `<svg width="${PW}" height="14" aria-hidden="true"><text x="6" y="11" font-size="9" fill="currentColor">0</text><text x="${X(0.5)}" y="11" font-size="9" text-anchor="middle" fill="currentColor">0.5</text><text x="${X(1)}" y="11" font-size="9" text-anchor="end" fill="currentColor">1</text></svg>`;
  const head = `<tr><th class="label">${ds === "Allen" ? "Cell group" : "Cell type"}<span class="sub">ψ on each panel, 0 to 1</span></th><th title="reads at this ATSE, TRAIN cells">Reads</th><th title="TRAIN cells with reads at this ATSE">Cells</th><th title="mice with reads at this ATSE, TRAIN cells">Mice</th>` +
    Array.from({ length: v.jn }, (_, k) => `<th class="panel"><span class="swatch" style="background:${jcolor(k)}"></span> <b>J${k + 1}</b> <span class="mono" style="font-size:10px">${jStart(v.jo + k)}-${jEnd(v.jo + k)}</span>${axis}</th>`).join("") + `</tr>`;
  const ref = [0, 0.5, 1].map((p) => `<line x1="${X(p)}" x2="${X(p)}" y1="0" y2="${RH}" stroke="var(--line)" stroke-width="0.7"/>`).join("");
  let body = "";
  for (const g of groupOrder(ds, st.group, groups.keys())) {
    const R = groups.get(g).sort((a, b) => a.t.name.localeCompare(b.t.name));
    body += `<tr class="group"><th class="label">${esc(g)} <span class="meta">(${R.length})</span></th><td colspan="${3 + v.jn}"></td></tr>`;
    for (const { t, i } of R) {
      const grey = isGrey(t), col = grey ? "var(--grey)" : "var(--ochre)", rawc = grey ? "var(--grey)" : "var(--raw)";
      body += `<tr class="${grey ? "grey" : ""}${hlSet.has(t.name) ? " hl" : ""}"><th class="label" title="${esc(t.label)}${grey ? `; grey: ${esc(greyReason(t))}` : ""}">${esc(st.group === "group1" && ds === "TMS" ? t.group2 : t.name)}</th><td class="n">${num(E.reads(i))}</td><td class="n">${num(E.cov(i))}</td><td class="n">${num(E.mice(i))}</td>`;
      for (let k = 0; k < v.jn; k++) {
        const p = E.psi(i, k), lo = E.lo(i, k), hi = E.hi(i, k), raw = E.raw(i, k), rr = E.jreads(i, k);
        let m = ref + (hasIv(lo, hi) ? `<line x1="${X(lo)}" x2="${X(hi)}" y1="7.5" y2="7.5" stroke="${col}" stroke-width="1.6"/>` : "") + `<circle cx="${X(p)}" cy="7.5" r="2.8" fill="${col}"/>`;
        if (!Number.isNaN(raw)) m += `<circle cx="${X(raw)}" cy="7.5" r="${Math.max(1.5, rawR(rr))}" fill="none" stroke="${rawc}" stroke-width="0.9"/>`;
        body += `<td data-t="${i}" data-k="${k}"><svg width="${PW}" height="${RH}">${m}</svg></td>`;
      }
      body += `</tr>`;
    }
  }
  return `<p class="meta">${rows.length} of ${T.length} ${typesLabel(ds)} shown.</p><div class="wrap dotwrap" style="max-height:82vh"><table class="dots"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function attachDotTip(ds, E) {
  const tip = $("tip"), host = $("ev-plot");
  host.onmousemove = (ev) => {
    const td = ev.target.closest("td[data-t]");
    if (!td) { tip.hidden = true; return; }
    const i = +td.dataset.t, k = +td.dataset.k, t = E.D.types[i];
    const ivs = ivStatus(E.treads(i), E.tcells(i));
    tip.innerHTML = `<b>${esc(t.label)}</b>, J${k + 1}<br>LeafletFA ψ ${ivText({ p: E.psi(i, k), lo: E.lo(i, k), hi: E.hi(i, k) })}<br><span class="meta">${ivs ? esc(ivs) : esc(CONFIG.INTERVAL_LABEL)}</span><br>held-out (TEST): ${num(E.treads(i))} reads, ${num(E.tcells(i))} cells with reads at this ATSE<br>raw pseudobulk ψ ${f2(E.raw(i, k))}; ${num(E.jreads(i, k))} of ${num(E.reads(i))} reads<br>${num(E.cov(i))} cells and ${num(E.mice(i))} mice with reads at this ATSE${isGrey(t) ? `<br><span class="meta">grey: ${esc(greyReason(t))}</span>` : ""}`;
    tip.hidden = false;
    tip.style.left = Math.min(innerWidth - 370, ev.clientX + 14) + "px"; tip.style.top = Math.min(innerHeight - 130, ev.clientY + 12) + "px";
  };
  host.onmouseleave = () => { tip.hidden = true; };
}

/* ---------- type shards ---------- */
async function loadType(ds, i) {
  const buf = await getBuf(`data/${ds}/types/${i}.bin`), L = INDEX.type_layout, J = JUNC.start.length, n = CAT.ids.length;
  const o = {}; let off = 0;
  for (const [name, kind, unit] of L) {
    const len = unit === "J" ? J : n, A = kind === "u32" ? Uint32Array : Uint16Array;
    o[name] = new A(buf, off, len); off += len * A.BYTES_PER_ELEMENT;
  }
  if (off !== buf.byteLength) throw new Error(`type shard ${ds}/${i}: expected ${off} bytes, got ${buf.byteLength}`);
  return o;
}

async function loadSpliceVI(ds, key) {
  const S = INDEX.datasets[ds].splicevi_ds;
  if (!S) return { map: {}, note: CONFIG.SPLICEVI_EMPTY_NOTE };
  let map = {};
  try { map = await getJSON(`data/${ds}/splicevi_ds/${key}.json`); } catch (x) { /* pair not in the file */ }
  return { map, note: `source: ${S.source}` };
}

/* ---------- shared table machinery ---------- */
function typeSelect(id, ds, cur, emptyLabel) {
  const T = INDEX.datasets[ds].types, g = new Map();
  T.forEach((t) => { const k = t.group1 || "unassigned"; if (!g.has(k)) g.set(k, []); g.get(k).push(t); });
  const opts = groupOrder(ds, "group1", g.keys()).map((k) => `<optgroup label="${esc(k)}">` + g.get(k).sort((a, b) => a.label.localeCompare(b.label)).map((t) =>
    `<option value="${esc(t.name)}"${t.name === cur ? " selected" : ""}>${esc(ds === "TMS" ? t.group2 : t.name)}${isGrey(t) ? " (grey)" : ""}, ${num(t.cells)} cells</option>`).join("") + `</optgroup>`).join("");
  return `<select id="${id}"><option value="">${esc(emptyLabel)}</option>${opts}</select>`;
}
function filterControls(st) {
  return `<label>Min reads per type <input id="f-reads" type="number" min="0" value="${st.minReads}"></label>
  <label>Min cells with reads per type <input id="f-cells" type="number" min="0" value="${st.minCells}"></label>
  <label>Min mice with reads per type <input id="f-mice" type="number" min="0" value="${st.minMice}"></label>
  <label>Event class <select id="f-class"><option value="">all</option>${CAT.classes.map((c) => `<option value="${c}"${st.cls === c ? " selected" : ""}>${CONFIG.CLASS_LABEL[c]}</option>`).join("")}</select></label>
  <label>Gene contains <input id="f-gene" size="10" value="${esc(st.gene)}"></label>`;
}
function bindFilters(st, rerun) {
  const b = (id, k, f) => { $(id).oninput = (x) => { st[k] = f(x.target.value); rerun(); }; };
  b("f-reads", "minReads", (s) => +s || 0); b("f-cells", "minCells", (s) => +s || 0); b("f-mice", "minMice", (s) => +s || 0);
  b("f-class", "cls", (s) => s); b("f-gene", "gene", (s) => s.toLowerCase());
}
const newFilters = () => ({ minReads: CONFIG.DEFAULT_MIN_READS, minCells: CONFIG.DEFAULT_MIN_CELLS, minMice: 0, cls: "", gene: "", page: 0 });
const passesMeta = (st, e) => (!st.cls || CAT.classes[CAT.cls[e]] === st.cls) && (!st.gene || CAT.genesLower[CAT.gene_idx[e]].includes(st.gene));

function intervalGlyph(a, b) {
  const W = 104, X = (p) => 3 + p * (W - 6);
  const seg = (x, y, col) => (hasIv(x.lo, x.hi) ? `<line x1="${X(x.lo)}" x2="${X(x.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2"/>` : "") + `<circle cx="${X(x.p)}" cy="${y}" r="2.4" fill="${col}"/>`;
  return `<svg width="${W}" height="15" aria-hidden="true"><line x1="${X(0)}" x2="${X(1)}" y1="14" y2="14" stroke="var(--line)"/>${seg(a, 4, "var(--ochre)")}${seg(b, 10, "var(--raw)")}</svg>`;
}
const hasIv = (lo, hi) => !Number.isNaN(lo) && !Number.isNaN(hi);
// 1 = overlap, 0 = no overlap, -1 = an interval is missing (shown and downloaded as NA)
const overlap = (a, b) => (hasIv(a.lo, a.hi) && hasIv(b.lo, b.hi) ? (a.lo <= b.hi && b.lo <= a.hi ? 1 : 0) : -1);
const ovlText = (o) => (o < 0 ? "NA" : o ? "yes" : "no");
const ivText = (x) => (hasIv(x.lo, x.hi) ? `${f2(x.p)} [${f2(x.lo)}, ${f2(x.hi)}]` : `${f2(x.p)} <span class="meta">[no interval]</span>`);
const ivStatus = (tr, tc) => (tr < CONFIG.IV_MIN_TEST_READS ? `no interval: fewer than ${CONFIG.IV_MIN_TEST_READS} held-out (TEST) reads`
  : tc < CONFIG.IV_MIN_TEST_CELLS ? `no interval: fewer than ${CONFIG.IV_MIN_TEST_CELLS} held-out (TEST) cells with reads` : "");

function renderTable(st, cols, rows, host) {
  const key = st.sort, dir = st.dir;
  const sorted = [...rows].sort((a, b) => { const x = a[key], y = b[key]; if (x === y) return 0; return (x > y ? 1 : -1) * dir; });
  st.sorted = sorted;
  const P = CONFIG.PAGE_SIZE, pages = Math.max(1, Math.ceil(sorted.length / P)); st.page = Math.min(st.page, pages - 1);
  const pg = sorted.slice(st.page * P, (st.page + 1) * P);
  const head = cols.map((c) => `<th class="${c.l ? "l " : ""}${c.wrap ? "wraptext " : ""}${c.sort ? "sortable " : ""}${c.sort === key ? "sorted" : ""}" ${c.sort ? `data-s="${c.sort}"` : ""} ${c.title ? `title="${esc(c.title)}"` : ""}>${c.h}${c.sort === key ? (dir < 0 ? " ▼" : " ▲") : ""}${c.sub ? `<span class="sub">${c.sub}</span>` : ""}</th>`).join("");
  const body = pg.map((r, i) => `<tr>` + cols.map((c) => `<td class="${c.l ? "l " : ""}${c.cls ? c.cls(r) : ""}">${c.f(r, st.page * P + i)}</td>`).join("") + `</tr>`).join("");
  host.innerHTML = `<div class="pager"><b>${num(sorted.length)}</b> <span>of ${num(CAT.ids.length)} ATSEs pass the filters</span>
    <button id="pg-prev" ${st.page ? "" : "disabled"}>Previous</button><span class="meta">page ${st.page + 1} of ${pages}</span><button id="pg-next" ${st.page + 1 < pages ? "" : "disabled"}>Next</button>
    <button id="dl">Download TSV (all ${num(sorted.length)} rows)</button></div>` +
    (sorted.length ? `<div class="wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>` : `<p class="hint">No ATSEs pass these filters. Lower the read or cell minimum.</p>`);
  host.querySelectorAll("th[data-s]").forEach((th) => { th.onclick = () => { const s = th.dataset.s; if (st.sort === s) st.dir = -st.dir; else { st.sort = s; st.dir = -1; } st.page = 0; renderTable(st, cols, rows, host); }; });
  $("pg-prev").onclick = () => { st.page--; renderTable(st, cols, rows, host); };
  $("pg-next").onclick = () => { st.page++; renderTable(st, cols, rows, host); };
  $("dl").onclick = () => st.download(sorted);
}
const spliceviCol = (sv) => ({ h: CONFIG.SPLICEVI_HEADER, sub: esc(sv.note), title: sv.note, l: 1, wrap: 1, f: (r) => esc(sv.map[CAT.ids[r.e]] ?? "") });
const greyBanner = (list) => { const g = list.filter(isGrey); return g.length ? `<div class="greynote">Grey: ${g.map((t) => `${esc(t.label)} (${esc(greyReason(t))})`).join("; ")}. Estimates shown for reference.</div>` : ""; };
const geneCell = (r) => `<a class="gene" href="#/gene/${encodeURIComponent(r.gene)}">${esc(r.gene)}</a>`;

/* ---------- compare two ---------- */
async function viewCompare(ds, an, bn) {
  ds = ds || lastDS; lastDS = ds;
  const D = INDEX.datasets[ds], A = an ? typeByName(ds, an) : null, B = bn ? typeByName(ds, bn) : null;
  const go = () => { const a = $("sel-a").value, b = $("sel-b").value; location.hash = `#/compare/${ds}${a ? "/" + encodeURIComponent(a) : ""}${b ? "/" + encodeURIComponent(b) : ""}`; };
  setPage(`<h1>Compare two ${typesLabel(ds)}</h1>
  <div class="note">${esc(CONFIG.COMPARE_NOTE)}</div>
  <div class="controls"><label>Dataset ${dsToggle(ds, (d) => `#/compare/${d}`)}</label>
    <label>A (ochre) ${typeSelect("sel-a", ds, A ? A.name : "", `Pick ${typeLabel(ds)} A`)}</label>
    <label>B (dark) ${typeSelect("sel-b", ds, B ? B.name : "", `Pick ${typeLabel(ds)} B`)}</label></div>
  <div id="cmp-body"><p class="hint">Pick two ${typesLabel(ds)}.</p></div>`);
  $("sel-a").onchange = go; $("sel-b").onchange = go;
  if (!A || !B) return;
  if (A === B) { $("cmp-body").innerHTML = `<p class="hint">Pick two different ${typesLabel(ds)}.</p>`; return; }
  $("cmp-body").innerHTML = `<p class="hint">Loading…</p>`;
  const [X, Y, sv] = await Promise.all([loadType(ds, A.i), loadType(ds, B.i), loadSpliceVI(ds, [A.i, B.i].sort((p, q) => p - q).join("__"))]);
  const st = { ...newFilters(), sort: "delta", dir: -1 };
  $("cmp-body").innerHTML = `${greyBanner([A, B])}<div class="controls">${filterControls(st)}</div>
  <p class="meta">Filters apply to both types. Δψ = ψ(A) − ψ(B), LeafletFA point estimates, at the junction with the largest |Δψ| in the ATSE (lead). Interval: ${esc(CONFIG.INTERVAL_LABEL)}. Reads, cells and mice count TRAIN cells at the ATSE.</p><div id="cmp-table"></div>`;
  const greyA = isGrey(A), greyB = isGrey(B);
  const cols = [
    { h: "#", f: (r, i) => i + 1 },
    { h: "Gene", l: 1, sort: "gene", f: geneCell },
    { h: "ATSE", l: 1, sort: "id", f: (r) => `<a class="mono" href="#/event/${ds}/${encodeURIComponent(r.id)}/${encodeURIComponent(A.name + "|" + B.name)}">${esc(r.id)}</a>` },
    { h: "Class", l: 1, sort: "cls", f: (r) => CONFIG.CLASS_LABEL[r.cls] },
    { h: "Lead", l: 1, title: "junction with the largest |Δψ|", f: (r) => `J${r.lead + 1}<span class="meta"> of ${r.jn}</span>` },
    { h: "Max |Δψ|", sort: "delta", f: (r) => `<b>${f2(r.delta)}</b>` },
    { h: "ψ A", sub: "[interval]", sort: "pa", cls: () => (greyA ? "grey" : ""), f: (r) => ivText(r.a) },
    { h: "ψ B", sub: "[interval]", sort: "pb", cls: () => (greyB ? "grey" : ""), f: (r) => ivText(r.b) },
    { h: "Raw ψ A", sub: "lead junction", sort: "rawa", title: "raw TRAIN pseudobulk ψ of A at the lead junction", f: (r) => f2(r.rawa) },
    { h: "Raw ψ B", sub: "lead junction", sort: "rawb", title: "raw TRAIN pseudobulk ψ of B at the lead junction", f: (r) => f2(r.rawb) },
    { h: "Raw Δψ", sub: "lead junction", sort: "rawd", title: "raw TRAIN pseudobulk at the lead junction", f: (r) => f2(r.rawd) },
    { h: "Intervals", sub: "A, B on 0 to 1", l: 1, f: (r) => intervalGlyph(r.a, r.b) },
    { h: "Overlap", sort: "ovl", title: `at the lead junction; ${CONFIG.INTERVAL_LABEL}; NA if either interval is missing`, f: (r) => ovlText(r.ovl) },
    { h: "Reads A", sort: "ra", f: (r) => num(r.ra) }, { h: "Cells A", sort: "ca", f: (r) => num(r.ca) },
    { h: "Reads B", sort: "rb", f: (r) => num(r.rb) }, { h: "Cells B", sort: "cb", f: (r) => num(r.cb) },
    spliceviCol(sv),
    { h: "Mice A", sort: "ma", f: (r) => num(r.ma) }, { h: "Mice B", sort: "mb", f: (r) => num(r.mb) },
  ];
  const run = () => {
    const rows = [];
    for (let e = 0; e < CAT.ids.length; e++) {
      if (X.reads[e] < st.minReads || Y.reads[e] < st.minReads || X.cov[e] < st.minCells || Y.cov[e] < st.minCells || X.mice[e] < st.minMice || Y.mice[e] < st.minMice || !passesMeta(st, e)) continue;
      const jo = CAT.jo[e], jn = CAT.jo[e + 1] - jo; let best = -1, lead = 0;
      for (let k = 0; k < jn; k++) { const d = Math.abs(X.psi[jo + k] - Y.psi[jo + k]); if (d > best) { best = d; lead = k; } }
      const j = jo + lead, a = { p: dec(X.psi[j]), lo: dec(X.lo[j]), hi: dec(X.hi[j]) }, b = { p: dec(Y.psi[j]), lo: dec(Y.lo[j]), hi: dec(Y.hi[j]) };
      rows.push({ e, id: CAT.ids[e], gene: CAT.genes[CAT.gene_idx[e]], cls: CAT.classes[CAT.cls[e]], jn, lead, j, delta: best / SCALE, a, b, pa: a.p, pb: b.p, sd: a.p - b.p, rawa: dec(X.raw[j]), rawb: dec(Y.raw[j]), rawd: dec(X.raw[j]) - dec(Y.raw[j]), ovl: overlap(a, b),
        ra: X.reads[e], ca: X.cov[e], ma: X.mice[e], rb: Y.reads[e], cb: Y.cov[e], mb: Y.mice[e] });
    }
    st.download = (sorted) => downloadTSV(`compare_${ds}_${slug(A.name)}_vs_${slug(B.name)}.tsv`, ds, [`A = ${A.label}`, `B = ${B.label}`, filterLine(st), CONFIG.COMPARE_NOTE],
      ["gene", "event_id", "event_class", "lead_junction", "max_abs_delta_psi", "psi_A", "lo_A", "hi_A", "psi_B", "lo_B", "hi_B", "signed_delta_psi", "raw_psi_A", "raw_psi_B", "raw_delta_psi", "intervals_overlap", "reads_A", "cells_A", "mice_A", "test_reads_A", "test_cells_A", "reads_B", "cells_B", "mice_B", "test_reads_B", "test_cells_B", "A_grey", "B_grey", "splicevi_ds_call"],
      sorted.map((r) => [r.gene, r.id, r.cls, jId(r.j, eventInfo(r.e)), f4(r.delta), f4(r.a.p), f4(r.a.lo), f4(r.a.hi), f4(r.b.p), f4(r.b.lo), f4(r.b.hi), f4(r.sd), f4(r.rawa), f4(r.rawb), f4(r.rawd), ovlText(r.ovl), r.ra, r.ca, r.ma, X.treads[r.e], X.tcells[r.e], r.rb, r.cb, r.mb, Y.treads[r.e], Y.tcells[r.e], greyA, greyB, sv.map[r.id] ?? ""]));
    renderTable(st, cols, rows, $("cmp-table"));
  };
  bindFilters(st, () => { st.page = 0; run(); });
  run();
}

/* ---------- one vs all ---------- */
async function viewOne(ds, an) {
  ds = ds || lastDS; lastDS = ds;
  const A = an ? typeByName(ds, an) : null;
  setPage(`<h1>One ${typeLabel(ds)} vs. all others</h1>
  <div class="note">${esc(CONFIG.COMPARE_NOTE)}</div>
  <div class="controls"><label>Dataset ${dsToggle(ds, (d) => `#/one/${d}`)}</label><label>${ds === "Allen" ? "Cell group" : "Cell type"} ${typeSelect("sel-a", ds, A ? A.name : "", `Pick a ${typeLabel(ds)}`)}</label></div>
  <div id="one-body"><p class="hint">Pick a ${typeLabel(ds)}.</p></div>`);
  $("sel-a").onchange = () => { const a = $("sel-a").value; location.hash = `#/one/${ds}${a ? "/" + encodeURIComponent(a) : ""}`; };
  if (!A) return;
  const [X, sv] = await Promise.all([loadType(ds, A.i), loadSpliceVI(ds, `${A.i}__rest`)]);
  const st = { ...newFilters(), sort: "delta", dir: -1 }, grey = isGrey(A);
  $("one-body").innerHTML = `${greyBanner([A])}<div class="controls">${filterControls(st)}</div>
  <p class="meta">Filters apply to this ${typeLabel(ds)}. Rest ψ = mean over all TRAIN cells of the other ${typesLabel(ds)} (saved in check 10). Range = lowest to highest LeafletFA ψ among other eligible ${typesLabel(ds)} with at least ${INDEX.range_min_reads} reads and ${INDEX.range_min_cells} cells at the ATSE (fixed; does not follow the filters). Beyond range = distance from this ψ to that range, 0 if inside. Lead = junction with the largest |ψ − rest ψ|. Interval: ${esc(CONFIG.INTERVAL_LABEL)}; ${esc(CONFIG.REST_INTERVAL_NOTE)}.</p><div id="one-table"></div>`;
  const cols = [
    { h: "#", f: (r, i) => i + 1 },
    { h: "Gene", l: 1, sort: "gene", f: geneCell },
    { h: "ATSE", l: 1, sort: "id", f: (r) => `<a class="mono" href="#/event/${ds}/${encodeURIComponent(r.id)}/${encodeURIComponent(A.name)}">${esc(r.id)}</a>` },
    { h: "Class", l: 1, sort: "cls", f: (r) => CONFIG.CLASS_LABEL[r.cls] },
    { h: "Lead", l: 1, f: (r) => `J${r.lead + 1}<span class="meta"> of ${r.jn}</span>` },
    { h: "Max |ψ − rest|", sort: "delta", f: (r) => `<b>${f2(r.delta)}</b>` },
    { h: "ψ", sub: "[interval]", sort: "pa", cls: () => (grey ? "grey" : ""), f: (r) => ivText(r.a) },
    { h: "Rest ψ", sort: "pb", title: CONFIG.REST_INTERVAL_NOTE, f: (r) => f2(r.b.p) },
    { h: "Range in others", f: (r) => `${f2(r.mn)} to ${f2(r.mx)}` },
    { h: "Beyond range", sort: "gap", f: (r) => f2(r.gap) },
    { h: "Intervals", sub: "this (interval), rest (point) on 0 to 1", l: 1, f: (r) => intervalGlyph(r.a, r.b) },
    { h: "Raw ψ", sub: "lead junction", sort: "raw", f: (r) => f2(r.raw) },
    { h: "Reads", sort: "ra", f: (r) => num(r.ra) }, { h: "Cells", sort: "ca", f: (r) => num(r.ca) },
    spliceviCol(sv),
    { h: "Mice", sort: "ma", f: (r) => num(r.ma) },
  ];
  const run = () => {
    const rows = [];
    for (let e = 0; e < CAT.ids.length; e++) {
      if (X.reads[e] < st.minReads || X.cov[e] < st.minCells || X.mice[e] < st.minMice || !passesMeta(st, e)) continue;
      const jo = CAT.jo[e], jn = CAT.jo[e + 1] - jo; let best = -1, lead = 0;
      for (let k = 0; k < jn; k++) { const d = Math.abs(X.psi[jo + k] - X.rest[jo + k]); if (d > best) { best = d; lead = k; } }
      const j = jo + lead, a = { p: dec(X.psi[j]), lo: dec(X.lo[j]), hi: dec(X.hi[j]) }, b = { p: dec(X.rest[j]), lo: dec(X.rest_lo[j]), hi: dec(X.rest_hi[j]) };
      const mn = dec(X.oth_min[j]), mx = dec(X.oth_max[j]);
      const gap = Number.isNaN(mn) ? -1 : Math.max(0, a.p - mx, mn - a.p);
      rows.push({ e, id: CAT.ids[e], gene: CAT.genes[CAT.gene_idx[e]], cls: CAT.classes[CAT.cls[e]], jn, lead, j, delta: best / SCALE, a, b, pa: a.p, pb: b.p, mn, mx, gap, raw: dec(X.raw[j]), ovl: overlap(a, b), ra: X.reads[e], ca: X.cov[e], ma: X.mice[e] });
    }
    st.download = (sorted) => downloadTSV(`one_vs_rest_${ds}_${slug(A.name)}.tsv`, ds, [`type = ${A.label}`, filterLine(st), CONFIG.COMPARE_NOTE],
      ["gene", "event_id", "event_class", "lead_junction", "max_abs_delta_psi", "psi", "lo", "hi", "psi_rest", "others_min", "others_max", "beyond_range", "raw_psi", "reads", "cells", "mice", "test_reads", "test_cells", "grey", "splicevi_ds_call"],
      sorted.map((r) => [r.gene, r.id, r.cls, jId(r.j, eventInfo(r.e)), f4(r.delta), f4(r.a.p), f4(r.a.lo), f4(r.a.hi), f4(r.b.p), f4(r.mn), f4(r.mx), r.gap < 0 ? "NA" : f4(r.gap), f4(r.raw), r.ra, r.ca, r.ma, X.treads[r.e], X.tcells[r.e], grey, sv.map[r.id] ?? ""]));
    renderTable(st, cols, rows, $("one-table"));
  };
  bindFilters(st, () => { st.page = 0; run(); });
  run();
}

/* ---------- downloads ---------- */
const slug = (s) => s.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
const filterLine = (st) => `filters: min reads ${st.minReads}, min cells with reads ${st.minCells}, min mice with reads ${st.minMice}, event class ${st.cls || "all"}, gene contains "${st.gene}"`;
function downloadTSV(name, ds, notes, header, rows) {
  const S = INDEX.datasets[ds].source;
  const head = [`LeafletFA cell-type splicing browser, version ${INDEX.version}`, `dataset ${ds}; seed ${S.seed}, K = ${S.K}`, `fit.npz md5 ${S.fit_md5}`, `fit commit ${S.fit_commit}${S.fit_dirty ? " (recorded dirty)" : ""}`,
    `estimates md5 ${S.estimates_md5}`, `intervals md5 ${S.intervals_md5}; tau^2 by TEST read bin ${JSON.stringify(S.tau2)}`, `interval: ${CONFIG.INTERVAL_LABEL}`, ...notes].map((l) => `# ${l}`);
  const txt = [...head, header.join("\t"), ...rows.map((r) => r.join("\t"))].join("\n") + "\n";
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([txt], { type: "text/tab-separated-values" })); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}
function downloadEvent(ds, E) {
  const v = E.v, rows = [];
  E.D.types.forEach((t, i) => { for (let k = 0; k < v.jn; k++) rows.push([t.label, t.group1 ?? "", isGrey(t), v.id, v.gene, jId(v.jo + k, v), f4(E.psi(i, k)), f4(E.lo(i, k)), f4(E.hi(i, k)), ivStatus(E.treads(i), E.tcells(i)) || "check 12 (a)+(b)", f4(E.raw(i, k)), E.jreads(i, k), E.reads(i), E.cov(i), E.mice(i), E.treads(i), E.tcells(i)]); });
  downloadTSV(`event_${ds}_${slug(v.id)}.tsv`, ds, [`event class ${v.cls}`], [typeLabel(ds).replace(" ", "_"), ds === "Allen" ? "class" : "tissue", "grey", "event_id", "gene", "junction_id", "psi", "lo", "hi", "interval_status", "raw_psi", "junction_reads", "atse_reads", "cells_with_reads", "mice_with_reads", "test_reads", "test_cells_with_reads"], rows);
}

/* ---------- routing ---------- */
async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === parts[0]));
  $("tip").hidden = true; $("q").value = ""; $("q-results").hidden = true; window.scrollTo(0, 0);
  try {
    if (parts[0] === "gene") viewGene(parts[1]);
    else if (parts[0] === "event") await viewEvent(parts[1], parts[2], parts[3] ? parts[3].split("|") : null);
    else if (parts[0] === "compare") await viewCompare(parts[1], parts[2], parts[3]);
    else if (parts[0] === "one") await viewOne(parts[1], parts[2]);
    else viewHome();
  } catch (x) { setPage(`<p>Could not load the page data (${esc(x.message)}).</p>`); console.error(x); }
}

async function init() {
  INDEX = await getJSON("data/index.json");
  const [c, jb] = await Promise.all([getJSON("data/catalog/events.json"), getBuf("data/catalog/junctions.bin")]);
  CAT = c; CAT.jo = Uint32Array.from(c.jo);
  CAT.byId = new Map(c.ids.map((id, i) => [id, i])); CAT.idsLower = c.ids.map((s) => s.toLowerCase()); CAT.genesLower = c.genes.map((s) => s.toLowerCase());
  CAT.gene_events = c.genes.map(() => []); c.gene_idx.forEach((g, e) => CAT.gene_events[g].push(e));
  const J = c.ids.length ? CAT.jo[c.ids.length] : 0;
  JUNC = { start: new Uint32Array(jb, 0, J), end: new Uint32Array(jb, 4 * J, J), status: new Uint8Array(jb, 8 * J, J) };
  $("footer").innerHTML = `Version ${esc(INDEX.version)}. Local preview; not published. Seed-0 TMS and Allen K = 20 fits, TRAIN cells only. Provenance: <a href="data/SOURCE.json">SOURCE.json</a>. Interval: ${esc(CONFIG.INTERVAL_LABEL)}.`;
  initSearch();
  window.addEventListener("hashchange", route);
  DARK_MQ.addEventListener("change", route);   // re-render so junction colors follow the color scheme
  route();
}
init().catch((e) => { $("main").innerHTML = `<p>Could not load data/index.json (${esc(e.message)}). This site needs http, not file://: run <code>python -m http.server -d SITE 8000</code>.</p>`; });
