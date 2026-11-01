# LeafletFA — Bioinformatics Original Paper plan

2026-11-01. Companion to the SpliceVI paper; both are submitted together by **30 November 2026**; writing starts **15 November**.
The Genome Biology manuscript (`GenomeBiology_Submission2026/`) was never submitted. It is source material only and stays
in place.

**Rule for this paper: only validated results go in the main text.** "Validated" means a pre-registered check passed
(`LeafletFA_validation/VALIDATION_PREREG.md`, amendments 1–4; results in `VALIDATION_REPORT.md`), or the result was rebuilt
and checked this month (*Fgfr2*). Everything else is dropped or held for the supplement until it is re-verified.

Folder layout:

```
Bioinformatics2026/
  BIOINFORMATICS_PLAN.md   this file
  manuscript/              Overleaf-ready: leafletfa_main.tex, leafletfa_supplement.tex, OUP class + bst, references.bib,
                           numbers.tex (generated), latexmkrc, Figures/
  paper/                   figures and numbers: paper_config.yaml (style shared with SpliceVI), figures/ (style.py,
                           make_figures.py, panels/), ledger/results_ledger.tsv, scripts/make_numbers_tex.py, captions/, records/
```

---

## 1. Title, abstract, figures

### Title options
1. **LeafletFA: a probabilistic factor model and event catalog for sparse single-cell splicing** (working title in the .tex)
2. Modeling duplicated, sparse junction reads in single cells with a Dirichlet-multinomial factor model
3. ATSEmapper and LeafletFA: consistent splicing events and per-cell splicing representations for full-length single-cell data
4. A validated factor model of single-cell junction usage

Option 1 names both deliverables (model + catalog). Option 2 leads with the modeling problem. Option 3 suits the
one-package framing.

### Structured abstract (outline; Bioinformatics, ≤ 250 words)
- **Motivation:** single-cell splicing arrives as a few PCR-duplicated reads per event, detected mainly where the gene is
  expressed. A per-cell model needs (i) events defined the same way in every cell and dataset and (ii) a likelihood that
  does not count duplicate reads as independent evidence.
- **Results:**
  - ATSEmapper defines alternative transcript structure events (ATSEs) from junction reads. The released catalog covers
    \CatalogATSEsMouse{} ATSEs in \CatalogCellsMouse{} mouse cells, plus human brain and bulk catalogs.
  - LeafletFA is a Dirichlet-multinomial factor model. Junction usage is a simplex within each ATSE, and a per-event
    concentration floor caps the weight duplicate reads carry.
  - Its per-cell representation reads junction usage rather than detection (accuracy drops \KnnDropRatiosTMS{} when ratios
    are redrawn). It fits held-out reads better than population, cell-type and one-program nulls in two atlases.
  - The representation is stable across seeds (kNN range \SeedKnnRangeTMS{}). On data simulated from the model at
    full-length depth, one fit recovers the parameters (ψ r \SimPsiR{}, per-cell φ r \SimPhiR{}).
  - Programs that recur across 15 seeds are broad lineage axes. For two of them (brain, epithelial), top events are
    confirmed by pseudobulk tests. Consensus over seeds plus a pseudobulk check is the recommended route to program-level claims.
- **Availability:** one Python package (ATSEmapper + LeafletFA) on GitHub/PyPI; catalog and fitted models on Zenodo.

### Main figures (4). Every panel mapped to an existing figure, table or script

Status: **built** = exists in a usable form; **plot** = data exist, panel to draw in paper style; **build** = needs new work.

**Figure 1 — ATSEmapper and the ATSE catalog**

| Panel | Content | Source | Status |
|---|---|---|---|
| a | Schematic: junction reads → splice graph → ATSE; annotation against the long-read GTF | none (GB Fig. 1 plan, `figures/FIGURE_PLAN_FINAL.md`); draw in Affinity | build |
| b | Catalog per dataset: cells, junctions, ATSEs, genes (mouse TMS + Allen, human Allen, bulk RNA atlas) | `supplemental/data_quality/supp_dataset_coverage.py` → `output/supp_dataset_coverage_*_summary.csv` | plot |
| c | ATSE structure: junctions per ATSE (44 % two-junction, mean 3.26) and ATSEs per gene | `supp_exons_vs_atses.py` → `output/supp_exons_vs_atses_*.csv`; cells `db4acb28`, `57b82209` of `Archive/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1/FIGURE1.ipynb` (2025 submission; old atlas) | plot (re-derive 44 % / 3.26 into the ledger) |
| d | Coverage: junctions detected per cell (median 3,349), reads per observed event | `supp_dataset_coverage_mouse_smartseq2_summary.csv` | plot |
| e | Duplicate reads: fraction of (cell, ATSE) pairs with every read on one junction vs read depth, against the independent-read expectation (\DuplicateAllOnOne{}; \DuplicateMolecules{} molecules) | Supplementary Note 1 (`GenomeBiology_Submission2026/manuscript/supplement/supplementary_notes.tex`), from `NOTE_duplicate_reads.md`; producing script to locate (ledger rows pending) | build (locate script, recompute, ledger) |

**Figure 2 — The model and what its per-cell representation reads**

| Panel | Content | Source | Status |
|---|---|---|---|
| a | Model: ψ simplex per ATSE, φ per cell, Dirichlet-multinomial likelihood with per-ATSE concentration (floor 2); E-step for new cells | none; `METHODS_LEAFLETFA_CURRENT.md` (LeafletFA) | build (schematic) |
| b | Held-out log-likelihood gain per read over the population, cell-type (supervised) and K = 1 nulls, TMS and Allen, 95 % CI over mice | check 1: `tsv/c1_{TMS,Allen}_K20_s0_gains.tsv` (`c1_score.py`, `c1_nulls.py`) | plot |
| c | Cell-type kNN accuracy on real data, with ratios redrawn and at the exact average, K = 5/10/20/30 | check 2: `tsv/c2_K_prior.tsv` (`c2_summary.py`, `knn_check.py`) | plot |
| d | Seed stability: kNN real / redrawn per seed, TMS and Allen | check 3: `tsv/c3_{TMS,Allen}_seed_knn.tsv` (`seed_replication.py`) | plot |

The ratio-redrawing test is the SpliceVI paper's (Section 3). The validation numbers use the **all-20-program** embedding
(`knn_check.py`). SpliceVI's Fig. 1 uses LeafletFA's **live** programs. Quote one embedding per sentence and say which.

**Figure 3 — Recovery on data simulated from the model**

| Panel | Content | Source | Status |
|---|---|---|---|
| a | True vs fitted ψ, live programs (median r 0.979; centered cosine 0.884; 6 of 7 programs) | check 4: `tsv/c4_recovery.tsv`, `c4_recovery_programs.tsv`; arrays `LeafletFA_validation/check4/` (`c4_simulate.py`, `c4_recover.py`) | plot |
| b | True vs fitted per-cell φ (r 0.982) | `check4/truth.npz`, `check4/fit_simA/` | plot |
| c | Depth limit: same at EasySci-like depth (×0.011): ψ r 0.745, 0 of 7 programs | `c4_recovery.tsv` (sim B), `check4/fit_simB/` | plot |

**Figure 4 — Consensus programs are broad lineage axes (and imputation at a known event)**

| Panel | Content | Source | Status |
|---|---|---|---|
| a | Consensus programs over 15 TMS fits: mean φ per tissue–cell type for CP1–CP4; recurrence (15, 12, 11, 10 of 15) and top-50 Jaccard marked | check 8: `tsv/c8_celltype_usage.tsv`, `c8_consensus_programs.tsv` (`c8_consensus.py`) | plot |
| b | Pseudobulk confirmation of top ATSEs, CP1 (brain, 9/10) and CP3 (epithelial, 8/10), against random ATSEs (0.20–0.23) | `tsv/c8_pseudobulk_top_atses.tsv`, `c8_pseudobulk_null.tsv` | plot |
| c | *Fgfr2* IIIb/IIIc: covered vs uncovered cells, raw vs LeafletFA, epithelial and mesenchymal types | `FIGURE2_simulation_imputation/code/build_fgfr2_example.py` panel b (branch `leafletfa-methods-fixes`, 52499bd); `data/fgfr2_IIIb_covered_uncovered.tsv` | built (GB theme); port to paper style |

### Programs section (Karin, 2026-11-01)

**Paper center**, in order:
1. The per-cell representation reads junction usage.
2. Held-out likelihood beats fair nulls.
3. Simulation recovery at full-length depth.
4. Consensus programs.
5. The ATSE catalog and ATSEmapper.

**The paper is not framed around named programs.** Programs appear as broad lineage axes, as the 15-seed consensus found them
(check 8: 4 clusters recur in ≥ 60 % of fits; 3 keep their top-50 junctions, Jaccard ≥ 0.5; the pre-registered bar of 8
was not met).

| Consensus program | Lineage axis | Recurrence | How the text uses it |
|---|---|---|---|
| CP1 | brain glia and neurons | 15 / 15 | **example confirmed by pseudobulk**: 9 of 10 top ATSEs confirmed (random ATSEs 0.23) |
| CP3 | intestinal and kidney epithelium, hepatocytes | 11 / 15 | **example confirmed by pseudobulk**: 8 of 10 (random 0.20) |
| CP2 | endothelial cells, lung smooth muscle | 12 / 15 | lineage axis only, **no junction-level claim** (used by 95 of 96 cell types, so the pseudobulk test has one non-user type) |
| CP4 | stromal and mesenchymal cells | 10 / 15 | lineage axis only, **no junction-level claim** (top-50 Jaccard 0.42) |

- **Supplement:** the keratinocyte/basal program is the worked example of **usage reproducing while content does not**. It
  is in 14 of 15 fits, in two compositions (centered cosine 0.77, never both in one fit), with the same top cell types.
  Show it with check 6's usage-vs-content table (S5–S6).
- **Recommended user workflow for program-level claims** (Methods and package docs): fit several seeds; build consensus
  programs over seeds (`c8_consensus.py` becomes a package function); make a program-level claim only for consensus
  programs; check a program's top ATSEs with a pseudobulk test between the cell types that use it and those that do not.
  A single fit's programs are not claims.
- **Figure 4** becomes "Consensus programs are broad lineage axes": a = cell-type usage of CP1–CP4 (recurrence marked);
  b = pseudobulk confirmation for CP1 and CP3 against random ATSEs. *Fgfr2* (c) stays as the imputation example, or moves
  to S8 if pages are short.
- Resolved: the check 6 four (with keratinocyte/basal) is not used.
- **Figure 3 is pending the DS checks** (amendment 5, below). If differential splicing passes, it may take main Figure 3
  and the simulation moves into Fig. 2 or the supplement. If it fails, DS stays out of the main text.

### One package
- **What:** one GitHub repository and one PyPI package, `leafletfa`.
  - `leafletfa.mapper`: ATSEmapper, moved in with its history (`git subtree add --prefix leafletfa/mapper ATSEmapper main`;
    daklab/ATSEmapper stays as an archived pointer).
  - `leafletfa` (model, E-step, differential splicing).
  - `leafletfa.catalog`: loader for the released catalog, e.g. `leafletfa.catalog.load("mouse_smartseq2_2026")`. It
    downloads from Zenodo and checks the md5.
- **CLI:** `leafletfa map` (ATSEmapper), `leafletfa fit`, `leafletfa embed` (E-step).
- **Catalog release (Zenodo):** per dataset, an ATSE table and the model-ready matrices' var tables.
  - ATSE table columns: event id, gene, junction coordinates, annotation status, cells observed, reads.
  - Also released: the fitted models (TMS and Allen K20 seeds 0–2, the 15-seed consensus fit) and the validation tables.
- **Fixes the package needs before release:**
  - `pyproject.toml` says "Bayesian Beta-Dirichlet Factor Analysis" (stale: the model is Dirichlet-multinomial).
  - The version reads 0.1.0 in `pyproject.toml` but 0.2.0 in the run records.
  - ATSEmapper's README still says 10x support is "under development".
  - The catalog definition drops real constitutive junctions at shared splice sites (the singleton limitation, e.g.
    *COL1A1*). Document it.

---

## 2. Dropped, kept, moved

| Item (GB source) | Decision | Why |
|---|---|---|
| Aging analysis, aging program, *Kcnip3* example (`FIGURE4_aging/`, all) | **drop** | age coincides with sequencing campaign and library format in TMS (Karin, 2026-10-26) |
| Former aging program as a partially reproducible program (Supp. SX, `build_supp_partial_program.py`) | **drop** | only existed to replace the aging claim |
| Named non-reproducing programs: SP-by-cell-type heatmaps with names, *Fbln2* decomposition, program sharing, *Gna13* | **drop** | fail checks 3, 6 and 8 |
| Human vs mouse conservation (GB Fig. 5) | **drop** | not built; the conserved-pair number did not reproduce (check 5b) |
| Differential splicing vs LeafCutter on the atlas (GB Fig. 3) | **hold; supplement only if re-verified by 15 Nov, else drop** | not in a pre-registered check (5b re-derived only the 840 vs 641 count); also the SpliceVI paper's ground |
| GB simulation vs NMF and DS calibration (GB Fig. 2 a–h) | **hold; supplement only if re-verified** | not validated (check 5b: "not checked"); different simulation from check 4 |
| Held-out imputation grid, uncertainty calibration (GB S10, `_imputation_uncertainty`) | **hold** | "not checked" in 5b; re-score on the validated fits if wanted |
| Catalog (GB S1 data quality), duplicate reads (Supp. Note 1) | **keep, main Fig. 1** | descriptive; reproduced (5b atlas size) |
| Held-out likelihood, kNN with ratios redrawn, seed stability, simulation recovery | **keep, main Figs 2–3** | checks 1–4 passed |
| Consensus programs, *Fgfr2* covered vs uncovered | **keep, main Fig. 4** | check 8; rebuilt 2026-10-31 |

**Supplement (planned):**

| # | Content | Source |
|---|---|---|
| S1 | Catalog coverage per dataset, exons vs ATSEs | `supplemental/data_quality/output/` (built) |
| S2 | Duplicate reads (with Supplementary Note 1) | Supp. Note 1 |
| S3 | Held-out likelihood checks: scorer agrees with the package, true parameters beat the null on simulated data, shuffled labels remove the cell-type null's advantage | `tsv/c1_verify_*` |
| S4 | K and prior: kNN, participation ratio, programs in use, replication by K | `tsv/c2_K_prior.tsv`, `c2_replication_*` |
| S5 | Program stability up to resolution: usage reproduces, junction content does not | `tsv/c6_*` |
| S6 | Consensus details: sensitivity to clustering, CP top junctions, the two keratinocyte compositions, pseudobulk table | `tsv/c8_*` |
| S7 | Simulation at EasySci-like depth (if Fig. 3c moves out) | `c4_*` |
| S8 | *Fgfr2* full figure: gene model, all types, per-seed φ (panels a, c) | `build_fgfr2_example.py` |

**Does it fit about 7 pages?** Yes, if Methods stay near 1,500 words.
- **Budget:** 4 double-column figures take about 2.5 pages. Abstract, text and references take about 4.5 pages,
  roughly 4,000 words.
- **Supplementary Methods:** the full model, inference settings and the validation protocol.
- **If over, in this order:**
  1. Move Fig. 3 to the supplement; quote its numbers in the Fig. 2 text.
  2. Move Fig. 4c (*Fgfr2*) to S8.
  3. Shorten Fig. 1 to a + b.

---

## 3. Overlap with the SpliceVI paper

**Shared data and inputs (both papers; say so in both):**
- **Raw data:** TMS and Allen Brain Smart-seq2, the same ATSEmapper quantification (SpliceVI Methods "AS and GE
  Quantification").
- **Model-ready files:** the same files and 70/30 splits (`train_70_30_*` / `test_30_70_*`, TMS_ONLY and AB_ONLY).
- **Fits:** the same LeafletFA fits (`LeafletFA_splicevi_baseline/{TMS,Allen}/full_K20_s{0,1,2}_f64`). They are
  SpliceVI's baseline and this paper's validated model.
- **Test inputs:** the same ratio-redrawn, exact-average and planted test inputs (SpliceVI STAGE5 §170).

**Must not be reused in the LeafletFA paper:**
- **SpliceVI Fig. 1c–f.** Fig. 1d–e is the ratio-redrawing test across methods.
- **SpliceVI's LeafletFA supplementary figures:** `figS_leafletfa_programs` and `figS_umaps`.
- **SpliceVI Supp. Fig. S1:** molecules per event, M_eff 1.19 / 1.26. Cite it for the data-measured value; do not redraw it.
- **SpliceVI's *Fgfr2* exon IIIb supplementary figure** (S5 in the current Fig. 2 caption). Ours (Fig. 4c) is a different
  figure on the atlas fits that shows covered vs uncovered cells; cross-reference it.
- **Text:** SpliceVI's Data / Quantification paragraphs and its LeafletFA baseline paragraph. Write ours fresh.

**Numbers that must agree between the two papers (reconcile before 15 Nov):**
- **Catalog size.** SpliceVI says 89,831 junctions, 28,015 ATSEs, 10,572 genes and 142,315 cells. The model-ready object
  has 79,768 junctions, 24,475 ATSEs, 11,233 genes and 132,706 cells. SpliceVI's figures look pre-filter; both papers
  should quote the same stage, ideally from this paper's ledger.
- **LeafletFA kNN numbers:** live programs (SpliceVI Fig. 1) vs all 20 programs (this paper's checks 2–3).
- **K sentence:** "K = 20 performed as well as K = 10 and better than K = 30" (SpliceVI Methods). Same fact here; reword,
  do not copy.

**Proposed cross-citation wording**

In the SpliceVI paper:
> Splicing events were defined with ATSEmapper and quantified as described in the companion paper (LeafletFA2026), which
> also releases the event catalog used here.

> LeafletFA (LeafletFA2026), a Bayesian factor model of junction usage developed alongside SpliceVI on the same
> quantification, was fitted with K = 20 programs...

This replaces "(Isaev et al., manuscript in submission)" in SpliceVI Methods.

In the LeafletFA paper:
> The companion paper (SpliceVI2026) introduces a test of whether a splicing representation reflects junction usage or
> junction detection, by redrawing each cell's junction ratios while keeping which junctions were detected; we apply it
> to LeafletFA here.

> SpliceVI (SpliceVI2026) models gene expression and splicing jointly with neural decoders; LeafletFA models splicing
> alone with a linear factor model whose programs and per-event concentrations can be read directly. Both were built on
> the event catalog described here.

---

## 4. Setup (done 2026-11-01, branch `bioinformatics`)

- **Branch:** `bioinformatics`, cut from `leafletfa-methods-fixes` (= `main` + the *Fgfr2* commits), in the worktree
  `~/Leaflet-analysis-bioinformatics`. Not pushed.
- **`manuscript/`:** Overleaf-ready.
  - `leafletfa_main.tex` on the OUP template with SpliceVI's class options (`webpdf,traditional,large,namedate`).
  - `leafletfa_supplement.tex`; `xr` cross-references, with the main text pointing to the supplement as in SpliceVI.
  - `oup-authoring-template.cls` and `oup-abbrvnat.bst`, copied byte-identical from the SpliceVI Overleaf project.
  - `references.bib`: seeded with 9 entries copied from SpliceVI's file, plus SpliceVI2026 and Kotliar2019 (both to
    verify).
  - `numbers.tex` and `latexmkrc`.
  - **To create the Overleaf project:** `git subtree push --prefix Bioinformatics2026/manuscript <overleaf-remote> master`,
    or upload the folder.
  - **Not compiled yet:** no LaTeX on the cluster; compile once in Overleaf.
- **`paper/`:** self-contained.
  - `paper_config.yaml`: style, palette and method conventions copied verbatim from SpliceVI's config; inputs point to the
    validation tables.
  - `figures/style.py` and `make_figures.py`, copied from SpliceVI; no figure modules yet.
  - Liberation Sans fonts (OFL).
  - `ledger/results_ledger.tsv`: 28 rows; 26 final with source md5, 2 pending (duplicate-read source).
  - `scripts/make_numbers_tex.py` (writes `numbers.tex` and copies it into `manuscript/`).
- **Color decision for Karin:** LeafletFA is ochre in SpliceVI's figures, and teal is SpliceVI. I kept ochre here so a
  reader of both papers sees one color per method. The LeafletFA theme would make it teal.

---

## 5. Before 15 November (so writing can start then)

| # | What | Kind | Est. |
|---|---|---|---|
| 1 | **Decide the fit set for Fig. 4c.** Validated fits are per-dataset (TMS, Allen); the *Fgfr2* panel uses the pooled atlas fits. Recommend: rebuild *Fgfr2* on the TMS fits (seeds 0–2, training cells) so every result is on validated fits. | decision + CPU | < 1 h |
| 2 | **Reconcile catalog numbers** with SpliceVI (Section 3) and put them in the ledger | check | 2 h |
| 3 | **Fig. 1:** schematic; panels b–d from the data-quality outputs; re-derive "44 % two-junction, mean 3.26" | plot + draw | 1 day |
| 4 | **Duplicate reads:** locate the script behind `NOTE_duplicate_reads.md`, recompute, ledger md5 (Fig. 1e, Supp. Note 1) | CPU | half day |
| 5 | **Fig. 2:** model schematic; b–d from the check 1–3 tables | plot + draw | 1 day |
| 6 | **Fig. 3:** a–c from `check4/` arrays | plot | half day |
| 7 | **Fig. 4:** a–b from the check 8 tables; c ported (or rebuilt, item 1) in paper style | plot | 1 day |
| 8 | **Package:** move ATSEmapper in; fix `pyproject`; CLI; tests; catalog export script and Zenodo draft | code | 2–3 days |
| 9 | **DS vs LeafCutter and the GB simulation:** re-verify for the supplement, or drop | decision (+ CPU if kept) | — |
| 10 | **Bibliography:** the GB `.bib` is not on this filesystem (an Overleaf project?). Build ours from SpliceVI's plus additions | text | half day |
| 11 | **SpliceVI side:** cross-citation edits and the catalog numbers (coordinate with the SpliceVI session) | text | 1 h |

No new GPU training is needed. Everything above is plotting, CPU, or code.

---

## Decisions for Karin
1. ~~Programs to report~~ — resolved 2026-11-01: consensus lineage axes; CP1 and CP3 as pseudobulk-confirmed examples.
2. Fit set for *Fgfr2* (Fig. 4c): rebuild on the TMS fits (recommended) or keep the atlas fits and say so.
3. DS vs LeafCutter and the GB simulation: re-verify for the supplement, or drop.
4. LeafletFA color: ochre (consistent with SpliceVI, current) or teal (LeafletFA theme).
5. Title.
6. Author list and order (left as placeholders).
