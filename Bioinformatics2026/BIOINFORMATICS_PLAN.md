# LeafletFA — Bioinformatics plan: cell-type-resolved splicing atlas (Original Paper; Application Note fallback)

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

## 1. Format, title, abstract, figures (Karin, 2026-11-01)

**Decision (Karin, 2026-11-02): Application Note, 4 pages, about 2,600 words.** Check 10: 10.1 passed; 10.2 and 10.7 failed their
pre-registered rule, so the Application Note reading applies. The outline is `~/LeafletFA_overleaf/main.tex`. Sections below written
for the Original Paper are kept as source material. The "Two pages" and "specificity replication" of the Application Note block below are superseded.

**Format: Original Paper (resource + method)** if check 10 (amendment 6) passes 10.1 and 10.2 a/b; **Application Note**
as fallback. The resource is the **ATSE catalog with cell-type-resolved junction usage**: for every ATSE × cell type
(TMS tissue–cell types; Allen cell groups (cluster names), with the official Allen subclass as an extra column), LeafletFA cell-type ψ with an uncertainty interval and a cell-type-specific
flag. Programs are no longer the headline; consensus programs move to the supplement. Human brain and bulk catalogs:
supplement and portal only.

### Title (working; final after check 10)
**A cell-type-resolved atlas of alternative splicing in mouse full-length single-cell data with LeafletFA**
(Earlier options 1–4 superseded.)

### Structured abstract (outline)
- **Motivation:** cell-type splicing from full-length single-cell data is measured as a few duplicated reads per event;
  pseudobulk per cell type is noisy for small or shallow types and gives no uncertainty.
- **Results:** ATSEmapper defines events; LeafletFA, a Dirichlet-multinomial factor model, estimates junction usage per
  cell and per cell type with intervals. Against held-out pseudobulk it is [TBD: as or more accurate, most for small /
  shallow types] (10.1); its cell-type-specific calls [TBD: replicate across mice] (10.2); known switches (*Fgfr2*,
  *Ptprc*, *Nrxn* SS4) are recovered (10.4). Its per-cell representation beats fair nulls on held-out reads and reads junction usage
  (shown in the companion paper), and per-cell usage reproduces across 15 seeds (r \PerCellPsiSeedR{}) although programs do not.
- **Availability:** catalog (Zenodo) with a static portal; ATSEmapper and LeafletFA (two repositories, one toolkit).

### Main figures (Original Paper)

| Fig | Content | Panels and sources | Status |
|---|---|---|---|
| **1 ATSEmapper and the catalog** | events, catalog size, coverage | a schematic (draw); b catalog per dataset, mouse only (`supplemental/data_quality/output/supp_dataset_coverage_mouse_smartseq2_summary.csv`); c junctions per ATSE, ATSEs per gene (`supp_exons_vs_atses_*.csv`); d coverage per cell; e duplicate reads (drop if not rebuilt by 8 Nov; cite SpliceVI Supp. Fig. S1) | plot / draw |
| **2 Model, scale and representation checks** | what LeafletFA is and that its per-cell representation is sound | a model schematic; b held-out log-likelihood vs fair nulls (check 1, `tsv/c1_*`); **c per-cell ψ stability across 15 seeds, enlarged** (r per seed pair, 0.998; `c9a_jaccard.tsv`, recompute on all cells) beside program recurrence (check 8: 4 of 110 live-program clusters recur in ≥ 60 % of fits), with kNN stability on real data per seed (check 3) as a small inset; d benchmark: time and memory, LeafletFA vs scQuint and the LeafCutter catalog (10.6). **The ratio-redrawing test is not shown** (Karin, 2026-11-01): it is in SpliceVI Fig. 1, where LeafletFA is one of the models; Results cite it, and the all-K numbers are a supplementary table (`paper/supplementary/table_ratio_redrawing_all_programs.tsv`) | plot; benchmark to run |
| **3 Cell-type ψ accuracy vs pseudobulk** | the resource is more accurate than the obvious alternative | a error vs TEST pseudobulk by type size; b by read depth; c interval coverage (10.3); d example events | check 10.1, 10.3 (to run) |
| **4 Cell-type-specific splicing** | calls, replication, known biology, portal | a δ and number of specific (event, type) per type; b replication across mice vs permuted null (10.2a/b), across seeds (10.2c); c *Fgfr2* (the upstream junction pair of the IIIb/IIIc mutually exclusive exons), *Ptprc*, *Nrxn1–3* SS4 (10.4; *Fgfr2* rebuilt on TMS fits); d portal screenshot | check 10.2, 10.4 (to run); portal |

**Supplement:** simulation recovery (check 4), consensus programs as broad lineage axes (check 8, CP1/CP3 pseudobulk
confirmed; keratinocyte two-composition example of usage reproducing while content does not), program stability (check 6),
K and prior (check 2), held-out likelihood verification, DS null result (check 9), in Karin's wording: "LeafCutter is anti-conservative on mouse-split nulls; LeafletFA's posterior test made no calls, consistent with its conservative behavior" (not as LeafletFA being calibrated), human brain and bulk catalogs,
Supplementary Note 1 (duplicate reads), catalog filter chain (`CATALOG_NUMBERS.md`).

### Application Note (fallback if check 10 fails 10.1 or 10.2 a/b)
One figure: a ATSEmapper and catalog; b model; c estimation accuracy vs pseudobulk by cell-type size; d specificity
replication and known-biology examples; e representation checks (held-out likelihood, per-cell ψ across seeds; ratio redrawing cited from SpliceVI). Two pages; the portal and catalog as the deliverable; claims limited to what passed.

### Paper center (Original Paper)
1. The catalog: cell-type ψ with intervals, more accurate than pseudobulk where data are thin (10.1, 10.3).
2. Cell-type-specific splicing that replicates across mice, with known biology recovered (10.2, 10.4).
3. The model's per-cell representation is sound: beats fair nulls on held-out reads, per-cell usage reproduces across seeds
   although programs do not; that it reads junction usage is shown in the SpliceVI paper (Fig. 1) and cited.
4. ATSEmapper and the toolkit; the portal.

### Portal (start now; Karin, 2026-11-01)
- Static site in `Bioinformatics2026/portal/` (build script; local preview; no GitHub repo, not published).
- Data: precomputed per-gene JSON and one small JSON index per cell type, generated from the catalog export.
- **Gene view:** search; per ATSE the junction structure, cell-type ψ heatmap with intervals and specific flags;
  client-side filters by tissue, broad lineage, name, minimum cells and reads, "specific only".
- **Cell-type view:** pick one or several cell types; list every ATSE flagged specific, sortable by Δψ and coverage,
  linking to the gene view.
- Download links to Zenodo. Build report: total size, largest file, build time. Later option (not built): one Parquet of
  the full catalog queried with DuckDB-WASM (feasibility and size noted in the build report).

### Interval caution (Karin, 2026-11-01)
The cell-type intervals come from the variational posterior (draws of q(φ) and q(ψ); amendment 6 note 2), which is often
too narrow. **10.3 interval coverage against TEST pseudobulk and the mouse-half replication are the primary safeguards**;
coverage is reported prominently (main Fig. 3c, abstract if it falls short), and specificity calls use P(TV > δ) ≥ 0.95
from per-draw TV.

### Methods text shared with SpliceVI
Mutually exclusive exons (corrected 2026-11-01): "A mutually exclusive exon pair forms one ATSE only if another junction links its two halves; otherwise it appears as two ATSEs, an alternative-3' and an alternative-5' pair." (DEFINITIONS.md 1.6; *Gria2* flip/flop linked, *Fgfr2* IIIb/IIIc unlinked).
The ratio-redrawing test is shown only in the SpliceVI paper (Fig. 1). The LeafletFA Methods describe it in two sentences
and cite SpliceVI; the LeafletFA Results cite SpliceVI Fig. 1 for the result (Karin, 2026-11-01, superseding the
self-contained description).
The ATSE definition paragraph in `manuscript/leafletfa_main.tex` (Methods, "ATSEmapper and the ATSE catalog"; from
`DEFINITIONS.md`) is the wording both papers use for the definition. The event-type table is Supplementary Table
(`tab:eventtypes`).

### Benchmarks (check 10.6)
Wall-clock and peak memory, hardware recorded: LeafletFA fit at TMS and atlas size and E-step; scQuint on TMS; LeafCutter
producing the same cell type × event catalog (one-vs-rest per type, total). LeafletSC (original Leaflet) dropped (Karin, 2026-11-01). No claims beyond the
measured numbers. The atlas-size LeafletFA fit needs ~0.5 GPU-h (approval).

### Check 10 compute (CPU) and target
| Part | Estimate |
|---|---|
| TMS seed 0: estimates, intervals (200 draws), leave-one-out, δ, halves A/B, 20 permutations, calibration, known biology | ~6–8 CPU-h |
| Allen seed 0: same | ~8–10 CPU-h |
| Seeds 1–14 (TMS) and 1–2 (Allen), call sets for 10.2c | ~1 CPU-h each, ~16 CPU-h |
| Benchmarks: LeafCutter one-vs-rest catalog on TMS (≈ 130 runs × ~1 h × 4 cores); scQuint | ~550 core-h (≈ 6 h wall at 25 concurrent) + ~2 h |
| **Total** | **~40 CPU-h analysis + ~550 core-h benchmark; target Tue 10 Nov** |

### Toolkit: two repositories, described as one (Karin, 2026-11-01)
No package merge before submission. ATSEmapper and LeafletFA are released as **two repositories presented as one
toolkit** (ATSEmapper defines and counts events; LeafletFA models them). Merge into one package after acceptance.

- **ATSEmapper** (daklab/ATSEmapper):
  - Version tag `v0.1.0` (matching `pyproject.toml`) **at the final checks on 29 Nov, together with the Zenodo release**
    (Karin, 2026-11-01); not before.
  - Clean repository root: remove the stray output folders (`LeafletFA_ATSE_mapper_output_2026051*`) and the `.csv` /
    `.tsv` files there (`multi_gene_junctions_*.csv`, `singleton_junctions.tsv`, `filtered_low_usage_junctions.tsv`), and
    the stale `leafletfa_utils.egg-info`.
  - README: no cluster paths; state 10x status (not supported in this release); document the singleton limitation
    (junctions at shared splice sites that form no alternative event are dropped, which removes real constitutive
    junctions such as *COL1A1*'s); tests pass.
  - The uncommitted demo rework already in the working tree (chr19 example, `setup_sample_data.sh`, `run_example.sh`,
    junction-parser columns) is reviewed and committed first.
- **LeafletFA** (daklab/LeafletFA):
  - `pyproject.toml`: description fixed (no "Beta-Dirichlet": a Dirichlet-multinomial factor model of junction usage);
    one version everywhere (`pyproject.toml`, `leafletfa.__version__`, run records).
  - `consensus_programs()` as a package function (from `scripts/validation/c8_consensus.py`): pool live centered ψ over
    fits, cluster, keep recurring clusters, return consensus ψ / π / concentration ready for the E-step; plus the
    pseudobulk check as a helper.
  - A Zenodo catalog export script: per dataset, the ATSE table (event id, gene, junction coordinates, annotation status,
    cells observed, reads) and the model-ready var tables, plus the fitted models (TMS and Allen K20 seeds 0–2, the
    15-seed consensus fit) and the validation tables, with md5s.
- **Text:** the paper names "the LeafletFA toolkit (ATSEmapper and LeafletFA)" with both repository URLs and one Zenodo DOI.

---

## 2. Dropped, kept, moved

| Item (GB source) | Decision | Why |
|---|---|---|
| Aging analysis, aging program, *Kcnip3* example (`FIGURE4_aging/`, all) | **drop** | age coincides with sequencing campaign and library format in TMS (Karin, 2026-10-26) |
| Former aging program as a partially reproducible program (Supp. SX, `build_supp_partial_program.py`) | **drop** | only existed to replace the aging claim |
| Named non-reproducing programs: SP-by-cell-type heatmaps with names, *Fbln2* decomposition, program sharing, *Gna13* | **drop** | fail checks 3, 6 and 8 |
| Human vs mouse conservation (GB Fig. 5) | **drop** | not built; the conserved-pair number did not reproduce (check 5b) |
| Differential splicing vs LeafCutter (GB Fig. 3) | **out of the paper** (check 9 failed 9a; no LRT check; Karin 2026-11-01); one Discussion line: the posterior test is very conservative at this depth | posterior test: 0–1 calls at seed 0 in the 5 Fig. 3b contrasts, calls do not reproduce across seeds (median Jaccard 0.0); calibrated on the real-data null. The LRT behind GB Fig. 3b was not tested |
| GB simulation vs NMF and DS calibration (GB Fig. 2 a–h) | **drop** (Karin, 2026-11-01) | not validated (check 5b: "not checked"); different simulation from check 4 |
| Held-out imputation grid, uncertainty calibration (GB S10, `_imputation_uncertainty`) | **hold** | "not checked" in 5b; re-score on the validated fits if wanted |
| Catalog (GB S1 data quality), duplicate reads (Supp. Note 1) | **keep, main Fig. 1** | descriptive; reproduced (5b atlas size) |
| Held-out likelihood, seed stability, per-cell ψ across seeds | **keep, main Fig. 2** | checks 1, 3, 9a |
| Ratio-redrawing test (kNN with ratios redrawn) | **cited from SpliceVI Fig. 1**; all-K numbers in a supplementary table | shown once, in the SpliceVI paper (Karin, 2026-11-01) |
| Simulation recovery | supplement | check 4 |
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

**Both papers compare to LeafCutter** (no head-to-head between SpliceVI and LeafletFA):
- SpliceVI does differential splicing on sparse EasySci metacells (and on Allen thinned to EasySci depth); LeafletFA does
  per-cell differential splicing on the full-length atlas (check 9 pending).
- Different data and depth regimes; neither paper compares the two methods' DS calls.
- Cross-citation sentences:
  - In SpliceVI: "On full-length data at atlas depth, the companion paper (LeafletFA2026) tests differential splicing per
    cell against LeafCutter; here we address the sparse regime of combinatorial-indexing atlases."
  - In LeafletFA: "The companion paper (SpliceVI2026) evaluates differential splicing against LeafCutter in sparse
    combinatorial-indexing data; here we test it per cell on full-length Smart-seq2 data."
  - **Update after check 9:** LeafletFA's DS is not in the main text, so the LeafletFA sentence becomes a Discussion
    pointer instead: "For differential splicing between groups of cells, see the companion paper (SpliceVI2026); here,
    LeafletFA's posterior effect-size test made no calls on mouse-split nulls, consistent with its conservative behavior,
    and its calls between cell types did not reproduce across
    seeds (Supplementary Note), so we use LeafletFA for per-cell representation and imputation." The SpliceVI sentence
    should then drop the reference to LeafletFA's DS.

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
- **Color (decided 2026-11-01): ochre in both papers.** LeafletFA is ochre in SpliceVI's figures, and teal is SpliceVI; so a
  reader of both papers sees one color per method. The LeafletFA theme would make it teal.

---

## 5. Before 15 November (so writing can start then)

| # | What | Kind | Est. |
|---|---|---|---|
| 1 | **Rebuild *Fgfr2* on the TMS fits** (seeds 0–2, training cells; Karin 2026-11-01), so every result is on validated fits | CPU | < 1 h |
| 2 | **Reconcile catalog numbers** with SpliceVI (Section 3) and put them in the ledger | check | 2 h |
| 3 | **Fig. 1:** schematic; panels b–d from the data-quality outputs; re-derive "44 % two-junction, mean 3.26" | plot + draw | 1 day |
| 4 | **Duplicate reads:** locate the script behind `NOTE_duplicate_reads.md`, recompute, ledger md5 (Fig. 1e, Supp. Note 1). **Deadline Sun 8 Nov**, else drop Fig. 1e and cite SpliceVI Supp. Fig. S1 | CPU | half day |
| 5 | **Fig. 2:** model schematic; b–d from the check 1–3 tables | plot + draw | 1 day |
| 6 | **Fig. 3:** a–c from `check4/` arrays | plot | half day |
| 7 | **Fig. 4:** a–b from the check 8 tables; c ported (or rebuilt, item 1) in paper style | plot | 1 day |
| 8 | **Toolkit (two repos, by Sun 15 Nov):** ATSEmapper tag + clean root + README (10x status, singleton limitation) + tests; LeafletFA pyproject description and version, `consensus_programs()`, Zenodo catalog export script. No merge before acceptance | code | 2 days |
| 9 | **DS checks (check 9) and CATALOG_NUMBERS.md** — done 2026-11-01 (check 9 fails 9a; LeafCutter null still running) | CPU | done |
| 10 | **Bibliography:** the GB `.bib` is not on this filesystem (an Overleaf project?). Build ours from SpliceVI's plus additions | text | half day |
| 11 | **SpliceVI side:** cross-citation edits and the catalog numbers (coordinate with the SpliceVI session) | text | 1 h |

No new GPU training is needed. Everything above is plotting, CPU, or code.

---

## Decisions for Karin
1. ~~Programs to report~~ — resolved 2026-11-01: consensus lineage axes; CP1 and CP3 as pseudobulk-confirmed examples.
2. ~~Fit set for *Fgfr2*~~ — resolved 2026-11-01: rebuild on the TMS per-dataset fits (seeds 0–2, training cells).
3. ~~DS vs LeafCutter~~ — 2026-11-01: re-verify as candidate main Fig. 3 (check 9, amendment 5: seed reproducibility, median Jaccard ≥ 0.7; real-data null, false-call rate ≤ 0.05). If it passes, simulation recovery moves to the supplement. GB simulation vs NMF: dropped.
4. ~~LeafletFA color~~ — resolved 2026-11-01: ochre in both papers.
5. ~~Title~~ — working title 2026-11-01: "A cell-type-resolved atlas of alternative splicing in mouse full-length single-cell
   data with LeafletFA"; final after check 10.
7. ~~DS~~ — resolved 2026-11-01: out of the paper, no LRT check; one Discussion line.
8. ~~ATSEmapper tag~~ — resolved 2026-11-01: tag `v0.1.0` on 29 Nov with the Zenodo release.
6. Author list and order (left as placeholders).
9. ~~Allen cell types~~ — resolved 2026-11-01: the 70 cell groups (cluster names, the cluster label without its number), called
   "cell groups (cluster names)" in all text, not subclasses; official `subclass_label` as an extra column in export and portal.
10. Atlas-size LeafletFA benchmark fit: ~0.5 GPU-h (b6k) — approve or use the existing dense atlas fit's log.
