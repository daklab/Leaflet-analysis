# Figure plan — FINAL, 2026-08-15

The single authority for what goes in the paper. Replaces `FIGURE_OUTLINE_ALL.md` (2026-08-01,
archived), which described the BetaBinomial pipeline. Everything superseded now lives in
`_archive_2026-08-15_pre_final/` with a manifest of why.

**The shipped model, which every figure validates:** Dirichlet-multinomial likelihood, within-ATSE
simplex psi, per-ATSE concentration floored at 2.0 (the measured effective sample size), Dirichlet
concentration on phi pinned at K, NO likelihood weight, K=20 seed 0 primary with {10, 30} x 3 seeds
as robustness. Differential splicing at delta = 0.15, derived from permuted negative-control ATSEs.

Status: DONE · BUILT-NEEDS-CAPTION · DATA-READY · RUNNING · TO-BUILD

---

## MAIN FIGURES

### Figure 1 — The mouse splicing atlas  [TO-BUILD, descriptive, no modelling]

| panel | content |
|---|---|
| a | schematic: tissues, Smart-seq2, junction -> ATSE construction |
| b | atlas composition: cells per tissue / cell type |
| c | ATSE structure: junctions per event (44% m=2, mean 3.26) |
| d | coverage: reads per cell; **the read non-independence result** -- 68% of (cell, ATSE) pairs put all reads on one junction at median 48 reads; PCR-duplication signature; solving P(all-on-one) gives 1.7-3.1 effective molecules per event |

Panel d is where the paper's central modelling decision is motivated -- the concentration floor is
the measured effective sample size -- so it belongs in Figure 1, not the supplement.

Numbers established: 132,706 cells x 79,768 junctions; 24,475 ATSEs; 70/30 split 92,894 / 39,812;
junction coverage 13.80%.

### Figure 2 — Simulation and held-out imputation  [REDESIGNED 2026-08-16: v2, one idea per panel]

`FIGURE2_simulation_imputation/plots/FIG2_v2.{pdf,png}` from `code/build_fig2_v2.py`
(prep: `code/prep_fig2_v2.py` -> `data/fig2_v2_prep.npz`). Caption: `CAPTION_FIG2_v2.md`.
FIG2_final (9 dense panels) is superseded; its extra panels go to the supplement.

Story in three rows -- setup / recovery / test+imputation:

| panel | content | status | results-text message |
|---|---|---|---|
| a | UMAP of TRUE phi, coloured by program (6,986 heart cells, K=5, 4,268 cassette exons, real depth) | DONE | this is what was simulated |
| b | one DIFFERENTIAL exon: true skip-PSI per program + observed per-cell violins by program | DONE | what the model has to find from counts |
| c | one NON-differential exon | DONE | what it must not call |
| d | UMAP of LEARNED phi (LeafletFA), coloured by true program, silhouette 0.82 (truth 0.87) | DONE | recovers the programs |
| e | NMF at the same rank, mean-imputed input, silhouette 0.41 | DONE | NMF merges programs even with imputation |
| f | between-program delta-psi recovery vs NMF (best gauge) by effect | DONE | 0.21->0.85 vs 0.0->0.24 |
| g | DS calibration: LFA LRT, LFA posterior P(TV>0.15) [from effect 0.25], LeafCutter | DONE | LFA tests reach 1.0 by effect 0.35; LeafCutter 2.2x at 1.0 |
| h | DS power beside g | DONE | LRT >= LeafCutter everywhere; posterior conservative by design |
| i | programs TRANSFER to unseen cells: mean SP activity per (tissue-cell-type, SP), training 70% vs held-out 30% (E-step), Pearson 0.998 vs 0.49 shuffled | DONE | plus SANITY_imputed_vs_raw_heldout.png: prior gets junction LEVEL (r 0.87) not within-junction ranking (0.11); posterior mean 0.98/0.95/0.74 at 10/50/90% removed; on entries with no reads left it reverts to the prior (0.14) |

To supplement from FIG2_final: concentration recovery, psi vs depth, phi by regime, delta trade-off,
atlas reach, conc-2 DS. Cell-type-mean baseline DROPPED everywhere (uses labels).

### Figure 3 — Differential splicing on the atlas, vs LeafCutter  [DATA-READY, TO-BUILD]

Four arms, one thing changing per row. `--design phi` LRT is DROPPED from the head-to-head: phi
enters through the posterior test, which is the only arm that uses what LeafletFA learns.

| panel | content | data | results-text message |
|---|---|---|---|
| a | calibration on each method's OWN null (QQ) | A52 + LC perm null | 1.42x vs 4.48x at 0.05. Caveat in caption: LeafletFA's null is contaminated by real signal so its ratio is an UPPER bound |
| b | calibration by event size m | A52 | **the sharpest result**: LeafletFA 1.15 -> 1.93 (m=2 -> 9), LeafCutter 3.23 -> 11.35. Seed-invariant to 3 decimals |
| c | calls at MATCHED realised FDP, group arm | A50 analysis_group | LeafletFA 840 vs LeafCutter 641 per contrast, 840/841/842 across seeds. Nominal-q counts (197 vs 1,124) shown greyed: that 5.7x gap is miscalibration, not sensitivity. **Framed as a data-design result** -- single cells vs 7v8 pseudobulk with the test held fixed |
| d | Jaccard decomposition by LeafCutter-measured effect and by m | `analysis_group/jaccard_decomposition.tsv.gz` | agreement rises monotonically with effect (19% -> 61% both-called); disagreement splits along m: LeafletFA-only at m=2 (55%), LeafCutter-only at m>=6 (38%) where its null is 11x inflated |
| e | posterior DS at delta 0.15: consensus set, COVERAGE-GATED, LeafletFA-only split by data support | A51 x3 seeds; `analysis_matched/posterior_consensus_gated.tsv.gz`; A65 `analysis_consistency/` | gate >=25 covered cells/arm (ungated 49,865 -> 7,041; 86% of ungated calls were extrapolation on <25 cells). Half of the gated calls are LeafCutter-confirmed. Of the 3,490 LeafletFA-only, **2,191 have observed pooled TV >= 0.15 in the contrast (real effects 7v8 pseudobulk missed), 1,299 do NOT** (hatched): psi is a global program property, so TV can be large because the PROGRAMS differ at that event while the two arms' own cells do not follow them (Gna13 ATSE1: posterior 0.33-0.41 every seed, cells 0.09). Say this in the text as a limitation of a K-program model, not hide it |
| f | posterior test vs LeafCutter on the SAME question (A63) | A63 at delta 0.15 (re-run) | on LeafCutter's own large-effect set: lower recall, higher precision (0.81 vs 0.68 at 0.10). Conservative and precise rather than sensitive |
| supp | per-ATSE effect sizes by contrast: posterior TV vs LeafCutter TV vs observed pooled TV, 6 contrasts | `code/build_effect_sizes_by_contrast.py` | LeafCutter TV tracks observed (Spearman 0.65-0.71); posterior TV saturates at a CONTRAST-SPECIFIC ceiling set by how far apart the groups' phi are (0.7 brain endo/oligo, 0.42 heart endo/fibro, ~0.1 skin basal/keratinocyte and intestine) -- cell types sharing programs cannot clear delta at any event. State as a property of the test |
| g | worked examples with genomic context, SAME contrast (brain endothelial vs oligodendrocyte): Apbb2 (both) and Mbd3 (LeafletFA-only) | `code/build_examples_v2.py` -> `plots/FIG3_examples_v2.png` | junction arcs over transcripts + per-junction single-cell violins with per-mouse pseudobulk dots. Apbb2: LC q 3e-18, posterior TV 0.64-0.76, observed 0.98. Mbd3: 238/414 cells shift J1 0.19->0.37 (observed TV 0.35, posterior 0.25-0.28 all seeds) but the 7v8 pseudobulks overlap and LC q = 0.20. Examples gated on >=50 cells/arm AND observed TV >= 0.25. Gna13 (was here) and Usp54 (before that) both REMOVED: Usp54 was 14 covered cells, Gna13 was a program-only call. Fbln2 program decomposition moves to supp |

> **PENDING (Karin, 2026-10-25): age is confounded with sequencing batch; this aging text is to be replaced by draft (a) or (b) in manuscript/AGING_SUPPLEMENT_DRAFTS.md. Do not cite it as an aging result.**

### Figure 4 — Aging  [TO-BUILD, biology]

Program 5 (K20) rises with age in every cell type tested; complete separation 3m vs 24m in male
mice (p = 0.0159, the minimum attainable with n=5 vs 4); replicates in 5/7 fits; one of the
reproducible programs (r 0.982). 2m EXCLUDED (all allen_brain -- a lab contrast, not age). Missing:
what the program IS -- its high-psi junctions and genes on the final model. Redo on shipped fit.

### Figure 5 — Human vs mouse brain  [TO-BUILD, biology, not scoped this session]

---

## SUPPLEMENTARY FIGURES

| # | content | source | status |
|---|---|---|---|
| S1 | data quality: exons vs ATSEs, dataset coverage, per dataset | `supplemental/data_quality/output/` | DONE |
| S2 | the model-selection factorial: BB vs DM x beta vs simplex -- calibration, psi validity (86.9% -> 100%) | A36/A37 tables | DATA-READY |
| S3 | phi collapse and the recipe: `FIG2_regime_phi_collapse` (contrast magnitude vs run-to-run CV; phi vs true_phi per arm; psi survives / phi collapses) | built | DONE |
| S4 | delta derivation: the permuted-ATSE gate on sims (permuted vs true-null TV, ratio 0.93-1.02) and the atlas floor per contrast | A64 outputs | DATA-READY |
| S5 | program replication: cross-seed matched r per program at K=10/20/30 (smooth gradient, NOT bimodal -- r>=0.9 is a chosen cutoff), and cross-K (14 of 20 survive K=20 -> K=30) | `a61_replication.tsv` | DATA-READY |
| S6 | phi structure: variance 92.8% cell type / 7.2% donor; averaging cells does NOT rescue phi (A60 flat n=1 -> 1000) | A59, A60 | DATA-READY |
| S7 | atlas phi UMAP on the shipped fit | `FIG3_supp_atlas_phi_umap_FINAL` | DONE |
| S8 | posterior test diagnostics: why delta 0.05 fails (psi posterior sd 0.024-0.09 per junction; three ruled-out fixes); the two TV functionals | delta sweep + null-floor tables | DATA-READY |
| S9 | LeafCutter on the DM simulations: full FDP/power table across effects and pseudobulk design | `LC_SIM_DM/summary_leafcutter_dm.tsv` | DATA-READY |
| S10 | held-out imputation: full method x depth x metric grid, both scoring sets (HVJ and all junctions), and the lost-to-downsampling entry set | A3 outputs | RUNNING |
| S11 | the m/(m-1) tempered arm as the ablation: same fits, same scores, why it was NOT shipped | runs_final vs runs_final_deltaonly | DATA-READY |

---

## What is NOT in the paper, and why

- `--design phi` LRT. Reduces LeafletFA to a covariate generator; phi's contribution is the
  posterior test.
- A53's "0.76% called on a true null" as calibration. That null has ||dphi|| ~ 0.12 against ~1.0
  for a real contrast; the floor it measures is 8x too small.
- Every DS number from the collapsed-phi model, including "FDP 0.000/0.018/0.016, conservative at
  every effect size". Invalid.
- Any imputation number scored on models trained on all cells.
- BetaBinomial vs Binomial validation panels. Neither ships.
