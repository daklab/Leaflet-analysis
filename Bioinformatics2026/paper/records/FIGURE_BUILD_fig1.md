# Figure 1 build record

- Data panels 1b–1d: `figures/panels/fig1.py`, inputs `inputs.fig1` in `paper_config.yaml`, written by
  `scripts/fig1_catalog_summaries.py` (job 22597637, 2026-11-01; June 2026 build, TMS-only and AB-only TRAIN + TEST files,
  all 78,135 TMS cells and 54,571 Allen nuclei; per-dataset junctions / ATSEs / genes = those with ≥ 1 read in the dataset).
- Event types: geometric rules of `DEFINITIONS.md` 1.6 (alt 3′ / alt 5′ / skipped exon / complex) on the catalog var.
- Cross-check: median reads per observed (cell, ATSE) 20 (TMS) and 4 (Allen), as in SpliceVI Supp. Fig. S1 (TEST cells).
- Slots: 1a ATSEmapper schematic (Karin, Affinity) 70 × 50 mm; 1e duplicate reads 58 × 40 mm (rebuild or drop by 8 Nov).
  Figure 2: 2a model schematic (Karin, Affinity) 70 × 55 mm.
- Numbers quoted in the text: ledger rows `Catalog*TMS/Allen`, `Median*`, `EventShare*`.
- Output: `<output_dir>/fig1/` (panels as PDF/SVG/PNG with clean metadata; assembled preview `fig1.pdf`).
