# Application Note: one main figure (proposal, 2026-11-05; for Karin's decision)

Limit from Karin: about 2,000 words plus one figure. Only panel a exists in part; the rest are not drawn. Width 183 mm, height about 110 mm.

| Panel | Shows | Source | Status |
|---|---|---|---|
| a | ATSEmapper: junctions to ATSE (schematic), with event types and catalog size | `paper/figures/panels/fig1.py` 1b, 1c; `CATALOG_NUMBERS.md`, `DEFINITIONS.md` | schematic to draw; 1b, 1c have code |
| b | Accuracy vs held-out pseudobulk, by type size and read depth, TMS and Allen (10.1, passed) | `tsv/c10_{TMS,Allen}_s0_accuracy.tsv` | to draw |
| c | Limits: real vs permuted confirmation under (i) and (ii), TMS and Allen, with intervals; labeled as failing the pre-registered rule | `VALIDATION_REPORT.md` 10.2 diagnostics; `c10_null_diag/*_summary.json` | to draw; LeafCutter added only if the control is finished and Karin wants it |
| d | One event, *Fgfr2*: structure and ψ by lineage, covered vs uncovered cells (10.4) | `c10_TMS_s0_known_biology.json`; `GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/build_fgfr2_example.py` | to rebuild for this format; the uncovered range conflict (0.35-0.63 vs 0.3-0.8) must be settled first |
| e | Browser: compare view for L2/3 IT CTX vs Pvalb, with the exploration note visible | `portal_browser/screenshots/05_*.png` | exists as a preview screenshot; replace with a real render after deployment |

## Moves out of the main figure

| Item | Goes to | Why |
|---|---|---|
| Model schematic | Text (one equation block) and Supplement | Space; the equation carries it |
| Representation checks (held-out likelihood, per-cell ψ across seeds, kNN) | Supplement (S3, S4) | Not the deliverable; the Plan's panel e is replaced by the browser |
| Interval coverage and width (10.3; `c10_diag_*`) | Supplement, one sentence in text | Limit stated in c's text; figures already drawn |
| Duplicate reads (old 1e) | Supplement S2 and Note 1 | Rebuild still owed by 8 Nov |
| Sparsity per cell (1d) | Supplement S1 | Secondary to accuracy |
| Program stability, consensus, K and prior, simulation | Supplement S4 to S7 | Plan as written |
| Benchmarks (10.6) | Supplementary table, one sentence in text | Plan as written |
| DS null (check 9), Allen donor composition table | Supplement | One sentence in text; wording of the Allen sentence is Karin's |
| Pairwise ranking, all featured examples (*Gria2*, *Nrxn1-3*, *Ptprc*), human and bulk catalogs | Browser | Exploration, no calls |

## Word budget at 2,000

| Section | Words | Ported so far |
|---|---|---|
| Abstract | 150 | 0 |
| Introduction | 200 | 187 |
| Data and catalog | 250 | 511 (shared with SpliceVI; cuts are Karin's) |
| ATSEmapper | 150 | 157 |
| Model and fitting | 200 | 205 |
| Cell-type usage and intervals | 150 | 92 |
| Results: catalog 120, accuracy 180, limits 170, biology and browser 120, scale 60 | 650 | 238 |
| Discussion | 250 | 131 |
| **Total** | **2,000** | **1,521** |

## Decisions needed
1. Is the failure shown as a panel (c, my recommendation, because the claim limits rest on it) or only in text?
2. Is e kept? Dropping it frees room for larger b and c.
3. Cuts to the shared data text, 261 words over its budget.
