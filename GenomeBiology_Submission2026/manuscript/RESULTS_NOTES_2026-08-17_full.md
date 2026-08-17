# Results notes — everything established 2026-08-16/17 (for writing Results; numbers are quotable)

Conventions: K = 20 seed 0 is the shipped atlas fit; every other claim carries its K 10/20/30 and
3-seed spread. Programs are SP1..SPK by fitted pi (SP1 = most used). TV_e = 1/2 sum_j |Delta_j| is
THE effect size; delta = 0.15 is the minimum effect to call; P(TV_e > delta) + Bayesian FDR is the
confidence; calls need q <= 0.05 in >= 2 of 3 seeds and >= 25 covered cells per arm.

## Figure 2 (simulation + held-out imputation) — v2

Setup. 6,986 heart cells keep their real per-ATSE depth; K = 5 programs (phi ~ Dir(20)); 4,268
cassette-exon ATSEs, 30% differential; DM emission conc 50 (atlas-like conc-2 set in supp).
- a: UMAP of true phi shows five groups. b/c: a differential ATSE (max between-program dpsi 0.47) and a
  null ATSE, per-program truth as bars over per-cell observed violins.
Recovery (K = 5 = true K, 3 model seeds).
- d/e: learned-phi UMAP silhouette 0.82 (truth 0.87); NMF at the same rank on the mean-imputed
  matrix 0.41 — NMF merges programs even with imputation.
- f: label-free detection of differential ATSEs from learned psi (max-min across programs): AUROC/
  AUPRC vs effect, LeafletFA >> NMF at every effect (mean +- SD over seeds).
Test (same simulations, same truth, same null tolerance).
- g calibration (realised FDP / nominal at 0.05): LeafletFA DM LRT on single cells reaches 1.0 by
  effect 0.35 and is conservative above; the posterior test P(TV > 0.15) (shown from effect 0.25,
  the smallest effect at which any true delta can exceed 0.15) sits at 1.2-2.0x; LeafCutter on 6 v 6
  random pseudobulks (its best case: no donor variance) is 2.2x above nominal even at effect 1.0 and
  8x at 0.1. h power: LRT >= LeafCutter everywhere; posterior test conservative by construction.
- conc-2 regime (supp): LFA 6.6/2.9/0.91 vs LC 5.4/4.3/4.0 at effects 0.25/0.5/1.0; power 0.52/0.79/
  0.95 vs 0.39/0.60/0.82.
- K misspecification on sims (model K = 3, 10): fits done, scoring running -> supp panel.
Imputation (i + supp).
- 70/30 split (92,894 / 39,812 cells; test cells NEVER in training — verified 100% of the earlier
  numbers were on seen cells and were discarded). Reads removed at 10..100%. Imputed = DM posterior
  mean (gamma_e rho + y)/(gamma_e + N); rho = phi.psi with phi from the E-step on the cell's own reads.
- Metrics: quote WITHIN-junction (across cells) and within-cell correlations; the pooled all-entry r
  is dominated by between-junction level (a per-junction constant scores 0.86, prior 0.87) and is not
  quoted.
- Entries with NO reads left (imputation proper): LeafletFA prior 0.05-0.06 within-junction; NMF
  mean-imputed 0.05->0.04; NMF 0-imputed 0.03; junction mean 0; label-based cell-type mean 0.04. All at
  the floor: a held-out cell's usage at one ATSE is ~2 effective molecules (98% of covered cells are
  all-on-one junction), so per-cell estimates cannot be SCORED against per-cell observations. This
  is the observation's limit, not the estimate's.
- With the cell's remaining reads (all covered entries): posterior mean 0.95 / 0.90 / 0.66 at 10 /
  50 / 90% removed; junction mean + same reads 0.86 / 0.80 / 0.51; NMF 0.10 / 0.09 / 0.07 (A3 seed 0;
  seeds 1-2 landing).
- Latent structure transfers: E-step phi of unseen cells reproduces the training cell-type x program
  pattern (Pearson 0.997, Spearman 0.87 over 21 broad types; 0.94 over 36 tissue-cell-types);
  nearest-training-centroid recovers the broad cell type from phi alone 62% (21 classes). phi barely
  changes when 90% of reads are removed (a cell has reads at thousands of ATSEs).
- Cell-type-level within-junction r (mean phi.psi per type vs pooled observed per type): A66,
  pending -> this is the "latent variables alone give useful usage" number.
- Uncertainty: posterior sd of the imputed value tracks across-seed spread (Fgfr2 event: Spearman
  0.74 over cells, 0.76 over cell types, ratio 1.2 = slightly conservative). A68 generalising over
  400 ATSEs x 6,000 cells + 90% CI coverage by sd bin: pending.
- Fgfr2 IIIb/IIIc (supp, 3 seeds): epithelial recovered (imputed 0.90-0.97 vs raw ~1.0 in 14 types
  with 11-290 covered cells); mesenchymal direction-only and seed-dependent (0.3-0.8 vs raw 0.0; 5-59
  covered cells) because mesenchymal types also load SP1/SP4 whose psi_IIIb is unconstrained (their
  cells do not express Fgfr2). Mammary basal clusters with mesenchymal in all seeds (myoepithelial);
  bladder epithelial only in seed 0. Old submission's Fgfr2 panel was empty (N=0).
- Wording to use: "LeafletFA denoises junction usage at single-cell resolution by pooling across all
  cells and ATSEs ... validated where the noise averages out: per cell type and per group, which is
  also where DS and program analyses use them."
- 70/30 refit seed 0 merged endothelial with mesenchymal (||dphi||_1 0.08 vs 0.97-1.27 elsewhere;
  silhouette 0.009); psi still matches the atlas 16/20 at r >= 0.9. Known across-seed phi-encoding
  degeneracy; report mean and range over seeds; K10/K30 x 3 refits launched.

## Figure 3 (atlas DS vs LeafCutter) — main = 4 panels

- a calibration by m on each method's own null: LeafletFA 1.15 -> 1.93 (m 2 -> 9), LeafCutter 3.23 ->
  11.35; seed-invariant. b matched realised FDP 0.05: LeafletFA 840 vs LeafCutter 641 calls per
  contrast (nominal q: 197 vs 1,124 — the 5.7x is miscalibration).
- c posterior test, coverage-gated (>= 25 covered cells/arm; 406,288 eligible pairs): 7,041 calls,
  50% also LeafCutter; LeafCutter-only 25,431. LeafletFA-only 3,490 split: 2,191 cell-supported
  (observed pooled TV >= 0.15 — effects the 7 v 8 pseudobulk lacked power for) and 1,299 program-level
  (observed < 0.15). Gna13 ATSE1 heart endo vs fibro is the canonical program-level call: posterior
  0.33-0.41 every seed, cells 0.09; direction real atlas-wide (endothelial 0.55-0.77 vs mesenchymal
  0.34-0.54 -> ~0.15) but amplified 3x because psi_SP9 (mesenchymal) sits below every mesenchymal
  type's observed usage. Say "program-level, amplified", not "wrong".
- d examples, same contrast (brain endothelial vs oligodendrocyte), genomic context + per-junction
  single-cell violins + per-mouse pseudobulks: Apbb2 (both; LC q 3e-18; posterior 0.64-0.76; observed
  0.98) and Mbd3 (LeafletFA-only; 238/414 cells, J1 0.19 -> 0.37, observed TV 0.35, posterior 0.25-
  0.28 all seeds; LC q 0.20 because pseudobulks overlap). Examples gated on observed TV >= 0.25 and
  >= 50 cells/arm. Usp54 (14 cells) and Gna13 removed.
- Property of the test (supp): TV_e <= 1/2 ||dphi_bar||_1; per contrast the 99th percentile TV =
  0.34 ||dphi||_1 - 0.04 (r 0.95); 29 of 70 contrasts have NO posterior call at any ATSE (LeafCutter
  makes 8,345 there): marrow B vs monocyte (||dphi|| 0.60, ceiling 0.09, 450 LC calls), skin basal vs
  keratinocyte, intestine epithelial vs intestinal. Cell types sharing programs cannot clear delta.
  Mean-phi heatmap: SP1 myeloid, SP2/3 broad, SP4 endothelial, SP6 lymphoid, SP9 mesenchymal, SP15
  keratinocyte/basal, SP16 astrocyte/oligo, SP17 hepatocyte, SP18 islet, SP20 intestinal.
- Effect sizes per contrast (supp): LeafCutter TV tracks observed pooled TV (Spearman 0.65-0.71);
  posterior TV saturates at the contrast ceiling.
- K sweep (supp): posterior consensus calls 6,360 / 7,041 / 6,666 at K 10/20/30; LC agreement 46 /
  50 / 50%; call-set Jaccard vs K20: K10 0.52, K30 0.71; per-contrast TV rank Spearman 0.93 / 0.97;
  contrasts with zero calls 33 / 29 / 29. LRT and LeafCutter arms do not depend on K.
- A63 (supp): on LeafCutter's own TV > 0.15 set, posterior recall 0.07 vs 0.22, precision 0.75 vs 0.56.
- Duplicate reads: 68% of covered cells all-on-one at median 48 reads -> 1.7-3.1 molecules; the per-
  ATSE concentration IS the effective sample size; floor 2 removes ~20x overconfidence at source.

## Data / models
- Atlas: 132,706 cells (TMS 78,135 + Allen Brain mouse 54,571), 79,768 junctions, 24,475 ATSEs.
- TMS B cells: 7,794 (marrow 3,448, spleen 2,266), 15 mice; 24m is male-only, 3m skews female.
- Training rule: every model K 10/20/30 x 3 seeds. Launched 2026-08-17: TMS-only (54,694 cells),
  mouse-AB-only (54,571), atlas 70/30 K10/K30, human AB full (43,442; K10/20/30). Human AB input
  ready: HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad.
- Package: DM model + simplex + per-ATSE conc are in leafletfa/minibatch.py with tests; defaults,
  __init__ exports, README, E-step and training driver still to be packaged (after the queue drains).
