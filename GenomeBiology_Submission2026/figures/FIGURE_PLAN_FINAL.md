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

### Figure 2 — Simulation and held-out imputation  [BUILT (a-h) / RUNNING (i-j)]

`FIGURE2_simulation_imputation/plots/FIG2_final.{pdf,png}` from `code/build_fig2_final.py`.

| panel | content | status | results-text message |
|---|---|---|---|
| a | psi effect-size recovery vs NMF at its BEST gauge | DONE | recovery 0.21 -> 0.85 vs NMF -0.01 -> 0.24: a factor model on compositions, not a matrix factorisation on ratios |
| b | concentration recovery 10/50/200 -> 10/49/183 | DONE | the overdispersion parameter is identified, not absorbing slack |
| c | psi recovery vs read depth | DONE | recovery RISES with coverage; a method at 1.0 on 1-2 reads would be fitting noise |
| d | DS calibration vs LeafCutter, same sims, same truth | DONE | LeafletFA's LRT is better calibrated at every effect size (5.99 -> 0.33 vs 7.94 -> 2.16) and reaches nominal around effect 0.30; **LeafCutter never does** |
| e | DS power at alpha 0.05 | DONE | more powerful than LeafCutter from effect 0.25 up; read ONLY beside d |
| f | phi vs truth by regime | DONE | the limitation panel: the recipe recovers 58% of phi's accuracy, not all; delta = 5K is WORSE than doing nothing (0.08 vs 0.25) |
| g | delta: calibration-power trade-off | DONE | delta 0.15 sits at ratio 1.2-2.0 with power 0.46-0.90; the operating point is READ off, not chosen |
| h | fraction of atlas above each TV (LeafCutter-measured) | DONE | 28% of atlas pairs exceed 0.10, 18% exceed 0.15 -- the posterior test speaks to the top fifth |
| **i** | **held-out imputation vs depth: LeafletFA / NMF / naive / mb_ct on the 30% split at 10-100% reads** | RUNNING (A3 sweep) | the ONLY imputation number scored on cells the model never saw |
| **j** | **the same at dm_conc = 2 (measured effective sample size) for panels d/e** | RUNNING (`c2_*`) | answers "you tested calibration on data cleaner than the atlas" |

Captions MUST carry: sims at median 33 reads/ATSE vs atlas 16 (every power number is an optimistic
bound); both methods anti-conservative below effect 0.20; delta 0.15 is derived (gate passed at
ratio 0.93-1.02; atlas floor p99 = 0.13). Panel h is scoping, and its number belongs in the text.

### Figure 3 — Differential splicing on the atlas, vs LeafCutter  [DATA-READY, TO-BUILD]

Four arms, one thing changing per row. `--design phi` LRT is DROPPED from the head-to-head: phi
enters through the posterior test, which is the only arm that uses what LeafletFA learns.

| panel | content | data | results-text message |
|---|---|---|---|
| a | calibration on each method's OWN null (QQ) | A52 + LC perm null | 1.42x vs 4.48x at 0.05. Caveat in caption: LeafletFA's null is contaminated by real signal so its ratio is an UPPER bound |
| b | calibration by event size m | A52 | **the sharpest result**: LeafletFA 1.15 -> 1.93 (m=2 -> 9), LeafCutter 3.23 -> 11.35. Seed-invariant to 3 decimals |
| c | calls at MATCHED realised FDP, group arm | A50 analysis_group | LeafletFA 840 vs LeafCutter 641 per contrast, 840/841/842 across seeds. Nominal-q counts (197 vs 1,124) shown greyed: that 5.7x gap is miscalibration, not sensitivity. **Framed as a data-design result** -- single cells vs 7v8 pseudobulk with the test held fixed |
| d | Jaccard decomposition by LeafCutter-measured effect and by m | `analysis_group/jaccard_decomposition.tsv.gz` | agreement rises monotonically with effect (19% -> 61% both-called); disagreement splits along m: LeafletFA-only at m=2 (55%), LeafCutter-only at m>=6 (38%) where its null is 11x inflated |
| e | posterior DS at delta 0.15: consensus set | A51 x3 seeds | 49,865 ATSEs called in >=2 of 3 seeds (2.9% of tested, median TV 0.256); rank consistency of TV across seeds Spearman 0.90; scale wobble stated as the reason for consensus |
| f | posterior test vs LeafCutter on the SAME question (A63) | A63 at delta 0.15 (re-run) | on LeafCutter's own large-effect set: lower recall, higher precision (0.81 vs 0.68 at 0.10). Conservative and precise rather than sensitive |
| g | worked example: Fbln2, heart endothelial vs fibroblast | `FIG3_supp_Fbln2_..._FINAL_K20_seed0` | program decomposition + posterior; predicted skip 0.55/0.10 vs LeafCutter 0.92/0.02, direction recovered in 3/3 seeds |

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
