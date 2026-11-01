# Splicing atlas portal (static)

Generated from the catalog export (check 10.5); no server code, all filtering client-side. Not published: local preview only.

```bash
python make_preview_catalog.py                       # PREVIEW data (raw TMS pseudobulk, ~100 genes) until check 10 exists
python build_portal.py preview_catalog.tsv.gz        # -> site/  (prints build time, total size, largest file, Parquet size)
python -m http.server -d site 8000                   # open http://localhost:8000  (fetch() needs http, not file://)
```

- `web/`: the app (index.html, app.js, style.css), copied into `site/`.
- `site/data/index.json`: datasets, genes, cell types; `site/data/genes/<dataset>/<gene>.json`: one file per gene;
  `site/data/celltypes/<dataset>/<cell type>.json`: events flagged specific in that type.
- Gene view: gene search; per ATSE the junction arcs and a cell type × junction ψ heatmap (hover: 95 % interval and
  rest-of-atlas ψ; ★ = specific); filters by tissue, lineage, name, minimum cells and reads, specific only.
- Cell-type view: one or several cell types; every specific event, sortable, linking to the gene view.
- Preview data carry `source = "PREVIEW raw pseudobulk"` and the site shows a banner.

**Build report (2026-11-01, preview: 99 genes, 246 events, 132 TMS cell types, 57,867 rows):** 27.9 s; 233 files;
3.86 MB total; largest file `data/genes/TMS/Picalm.json`, 114.7 kB; the same rows as Parquet 1.84 MB.
Extrapolated to the full TMS catalog (24,475 events, ~100× the preview): ~400 MB of per-gene JSON, largest gene files
a few MB; Parquet ~180 MB per dataset. A single Parquet queried with DuckDB-WASM is feasible at that size (range
requests), but not built.
