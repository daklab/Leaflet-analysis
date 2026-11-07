# Application Note: one main figure (revised 2026-11-06 to Karin's layout; layout PNG for approval before building)

Limit from Karin: about 2,000 words plus one figure. Only panel a exists in part; the rest are not drawn. Width 183 mm, height 165 mm (layout mock-up).

Layout mock-up (no data): `paper/figures/layout/fig1_appnote_layout.png` (`fig1_appnote_layout.py`). Bioinformatics: 4 pages, about 2,600 words, or 2,000 words plus one figure (Karin).

| Panel | Shows | Source | Status |
|---|---|---|---|
| a | Catalog and model: ATSEmapper schematic, catalog per atlas and event types, LeafletFA in one line | `paper/figures/panels/fig1.py` (1b, 1c), `DEFINITIONS.md`, ledger | schematic to draw |
| b | Cell-type accuracy vs raw pseudobulk (10.1), by TRAIN reads and type size, TMS and Allen, 95 % CI | `tsv/c10_{TMS,Allen}_s0_accuracy.tsv`, `c10/*_accuracy_units.tsv.gz` | to draw |
| c | *Gria2* flip/flop, linked MXE (Allen `ENSMUSG00000033981.14_atse_3`), neurons by class | portal shards, structure from `precompute.py` | to draw |
| d | *Fgfr2* IIIb/IIIc (TMS atse_3), epithelial vs mesenchymal, covered and uncovered cells apart | `fgfr2_IIIb_covered_uncovered.tsv` (12 mesenchymal types) | to rebuild in this format |
| e | *Nrxn1* SS4 (atse_2), excitatory vs inhibitory (Allen); chosen by Karin 2026-11-06 (best-known SLM2 target; matches SpliceVI Fig. 2 switches) | `tsv/c10_Allen_s0_nrxn_pairs_note4.tsv` | chosen, to draw |
| f | *Ptprc* in lymphoid cells (TMS atse_2), B and T vs other immune types | `tsv/c10_TMS_s0_ptprc_direction.tsv` | to draw |

Each of c to f: exon structure on top (the browser's structure diagram, GENCODE vM19), ψ per cell type below (LeafletFA point with interval labeled not calibrated, raw TRAIN pseudobulk sized by reads). The browser gap diagnosis (TMS ranking vs raw) must finish before d and f are drawn, since both are TMS.

## Supplement gallery (10 to 20 ATSEs)
Selection rule, approved by Karin as proposed (2026-11-06); event list draft in GALLERY_KNOWN_BIOLOGY_DRAFT.md: event classes spread (alt 3', alt 5', skipped exon, MXE linked, complex), both atlases, at least 20 reads and 3 cells in the shown types, and LeafletFA and raw Δψ of the same sign; known-biology events first (literature list to be agreed), not the top of a ranking.

## Moves out of the main figure
Interval coverage and width (10.3), duplicate reads, sparsity per cell, representation checks, program stability, benchmarks, DS null, Allen donor composition: Supplement. Everything else: browser.

## Word budget at 2,000
Unchanged from 2026-11-05; current counts in the Overleaf comments.
