# Results draft — Figures 2 and 3, bullet form (2026-08-18)
Each bullet = one claim, with the panel it rests on and the number to quote. Karin converts to prose.
"S2.x" = Figure 2 supplement files in FIGURE2_simulation_imputation/plots; "S3.x" = FIGURE3_leafcutter_ds/plots.

## Section: LeafletFA recovers splicing programs and differential ATSEs on realistic simulations (Fig. 2)

**Setup (Fig. 2a–c)**
- Simulations start from 6,986 real heart cells: each keeps its actual per-ATSE read depth; K = 5 latent
  programs (φ ~ Dirichlet(20)); 4,268 cassette-exon ATSEs of which 30% differ between programs by a
  stated effect ε ∈ {0.10 … 1.0}; counts drawn per cell and ATSE from a Dirichlet-multinomial over
  the three junctions (conc 50 main; conc 2 = the atlas's effective sample size in S2). [2a; Methods]
- 2a: UMAP of true φ — five clean groups. 2b: a differential ATSE (max Δψ 0.47) — per-program truth
  (bars) over the observed per-cell skip-junction ratios (violins) — what has to be found from counts.
  2c: a non-differential ATSE — same usage in every program — what must not be called.

**Program recovery without imputation (Fig. 2d–f)**
- LeafletFA's learned φ separates the five true programs (silhouette 0.82 vs 0.87 for the truth) [2d];
  NMF at the same rank on the mean-imputed usage matrix merges them (0.41) [2e].
- Which ATSEs differ between programs, scored label-free from learned ψ (max−min across programs vs
  the simulated label): AUROC/AUPRC rise with effect and are far above NMF at every effect
  (mean ± SD, 3 seeds) [2f]. ψ recovery r ≥ 0.97 at every effect (S2 sim-K panel).

**Differential-splicing test (Fig. 2g–h)**
- Same simulations, same truth, same null tolerance for all three tests [Methods].
- Calibration [2g]: LeafletFA's DM likelihood-ratio test on single cells reaches nominal FDP by effect
  0.35 and is conservative above; the posterior test P(TV > 0.15) sits at 1.2–2.0× (shown from effect
  0.25 — below that no true effect can exceed δ); LeafCutter on 6-vs-6 random pseudobulks (no donor
  variance, its best case) is 2.2× above nominal even at effect 1.0 and ~8× at 0.1.
- Power [2h]: the LRT matches or exceeds LeafCutter at every effect; the posterior test is deliberately
  conservative (it demands TV > 0.15, not > 0).
- At the atlas's overdispersion (conc 2; S2 v1-panels f): LFA 6.6/2.9/0.91× vs LC 5.4/4.3/4.0× at
  effects 0.25/0.5/1.0; power 0.52/0.79/0.95 vs 0.39/0.60/0.82.
- Rank misspecification (S2 sim-K panel; true K 5, model K 3/5/10, 3 seeds): ψ recovery is robust
  (r 0.93–0.99); under-K merges programs (φ silhouette 0.55–0.66 vs 0.67–0.87 at K 5), over-K splits
  them (0.62–0.86); DS-ATSE AUROC 0.95–0.99 / 0.91–1.00 / 0.84–0.97. The tests react in opposite ways:
  the LRT (φ as design) becomes over-conservative under-K (0.23–0.50×) and ANTI-conservative over-K
  (3.6–3.9×: nine design columns overfit at these counts), while at K 5 it is 1.8× / 0.71× / 0.33× at
  effects 0.25/0.5/1.0; the posterior test P(TV > 0.15) keeps its calibration under-K (2–3×) and
  over-K (0.6–2×) but loses power in both directions (effect 0.5: 0.26 / 0.72 / 0.41 at K 3/5/10)
  because merged or split programs shrink Δφ. Choosing K by program reproducibility across seeds and
  ranks (Methods) and reporting the K sweep on the atlas is therefore part of the method, not a
  tuning detail.
- Choosing δ (S2 v1-panels d/e): δ = 0.15 derived from permuted negative-control ATSEs (floor p99 = 0.13
  on the atlas); δ 0.05 sits below ψ's own posterior sd (0.02–0.09) and is 5–6× anti-conservative;
  on LeafCutter's own atlas measurements 18% of (contrast, ATSE) pairs exceed TV 0.15 — the reach of the test.

**Programs transfer to unseen cells (Fig. 2i)**
- Atlas split 70/30 (92,894 / 39,812 cells; test cells never in training). φ for held-out cells is
  inferred by the E-step from their own reads with ψ, π, γ frozen.
- Mean SP activity per (tissue-cell-type, SP): training vs held-out Pearson 0.998 (0.991 with 90% of the
  held-out cells' reads removed); with cell labels shuffled 0.49; 110 types × 20 SPs [2i; S2 SP
  train-vs-heldout heatmap, one per seed]. Held-out φ assigns the broad cell type by nearest training
  centroid 62% of the time (21 classes, chance 5%).
- Imputation proper (S2 imputation-metrics): LeafletFA denoises usage at single-cell resolution by
  pooling across all cells and ATSEs. The prior φ·ψ alone recovers cell-type-level usage (within-
  junction r across cell types 0.39, all 3 seeds; 0.88 → 0.83 with the cell's own reads folded in at
  0 → 90% removed; NMF 0.41 at both levels). With remaining reads the posterior mean degrades gracefully (per-cell
  within-junction r 0.95 / 0.90 / 0.66 at 10 / 50 / 90% removed vs junction-mean+reads 0.86 / 0.80 /
  0.51 and NMF 0.10 / 0.09 / 0.07). Because one cell's observed usage at one ATSE rests on ~2 effective
  molecules, per-cell estimates cannot be scored against per-cell observations (label-based cell-type
  mean scores 0.04 on the same entries): all label-free methods sit at 0.03–0.06 there. Validation is
  where the noise averages out — per cell type and group — which is where DS uses ρ = φ·ψ.
- No-leakage check (S2 imputation metrics / per-dataset): a model trained on TMS alone (same junctions)
  imputes the same TMS test cells identically to the atlas model (posterior mean 0.923/0.876/0.678 vs
  0.919/0.870/0.668 at 10/50/90% removed) -- co-training with Allen Brain costs nothing, so programs
  are not batch artefacts.
- Uncertainty is calibrated (S2 imputation-uncertainty): posterior sd of the imputed value tracks
  seed-to-seed sd (Spearman 0.88 over 7.5 M cell-junction pairs, 400 ATSEs); 90% credible intervals
  contain other seeds' values 87% (83–91% across sd deciles); sd is largest where the cell has no reads
  at the ATSE (0.012 vs 0.003 with > 5 reads).
- Fgfr2 IIIb/IIIc (S2 Fgfr2, 3 seeds): the epithelial arm of the canonical switch is recovered
  (imputed 0.90–0.97 vs raw ~1.0 across 14 epithelial types); the mesenchymal arm is direction-only
  and seed-dependent (0.3–0.8 vs raw 0.0; 5–59 covered cells) because mesenchymal types also load
  programs whose ψ at this event is unconstrained (their cells do not express Fgfr2). Mammary basal
  cells cluster with mesenchymal types in all seeds (myoepithelial).

## Section: On the atlas, single-cell LeafletFA is calibrated where pseudobulk LeafCutter is not, and its posterior test asks a program-level question (Fig. 3)

**Design [Methods; Fig. 3a–b]**
- 70 within-tissue cell-type contrasts. LeafCutter on 7-vs-8 per-mouse pseudobulks (the realistic
  Smart-seq2 design); LeafletFA's DM LRT on every cell with the same cell-type covariate; each
  method's calibration on its own permutation null; comparison at matched REALISED FDP 0.05.

**Calibration and calls (Fig. 3a–b; S3 calibration_and_K a)**
- Realised/nominal FDP at 0.05 by junction count: LeafCutter 3.2× (m = 2) → 11.4× (m = 9);
  LeafletFA 1.15× → 1.9×; identical across 3 seeds [3a]. Overall on their own nulls: 1.42× vs 4.48×
  (S3 a QQ). LeafletFA's null carries residual donor structure so its ratio is an upper bound.
- At matched realised FDP: LeafletFA 840 vs LeafCutter 641 ATSEs per contrast (840/841/842 across
  seeds); at nominal q ≤ 0.05 LeafCutter's 1,124 vs 197 is miscalibration, not sensitivity [3b].
  Framed as a data-design result: same test, same covariate, single cells vs pseudobulk.
- Where they disagree (S3 calibration_and_K b): agreement rises with LeafCutter-measured effect (19%
  both-called below TV 0.05 → 61% above 0.30); LeafletFA-only calls sit at m = 2, LeafCutter-only at
  m ≥ 6 where its null is 11× inflated.

**The posterior test (Fig. 3c)**
- Uses the model's program compositions: P(TV_e > 0.15), Bayesian FDR q ≤ 0.05, ≥ 2 of 3 seeds,
  ≥ 25 covered cells per arm (406,288 eligible pairs). Ungated, 86% of calls sat on < 25 covered cells
  — the model reporting its prior; gating removed them (49,865 → 7,041).
- 7,041 calls; half (3,551) also made by LeafCutter; LeafCutter calls 25,431 the posterior test does not.
- LeafletFA-only 3,490 split by what the cells in that contrast do: 2,191 cell-supported (pooled
  observed TV ≥ 0.15 — real effects the 7-vs-8 pseudobulk lacked power for) and 1,299 program-level
  (observed < 0.15): the two groups' programs differ at the event but the arm's cells do not follow
  them. Canonical case Gna13 ATSE1 heart endothelial vs fibroblast: posterior 0.33–0.41 every seed,
  cells 0.09; the endothelial-vs-mesenchymal difference is real atlas-wide (0.55–0.77 vs 0.34–0.54,
  ~0.15) but the model reports it 3× amplified because ψ_SP9 (mesenchymal) sits below every mesenchymal
  type's observed usage. Wording: "program-level, amplified", not "wrong"; reported, not merged.
- Effect-size scale: TV_e = ½ Σ_j |Δ_j|, the fraction of the ATSE's splicing mass that moves; Δ_j sum
  to 0 within an ATSE; for two junctions |ΔPSI|; for a cassette exon |ΔPSI_skip| [Methods; worked K=2
  example].
- What the test can see (S3 program_sharing): TV_e ≤ ½‖Δφ̄‖₁ because ψ sums to 1 within an ATSE; per
  contrast the 99th-percentile TV = 0.34 ‖Δφ̄‖₁ − 0.04 (r 0.95). 29 of 70 contrasts have no posterior
  call at any ATSE (LeafCutter makes 8,345 there): marrow B cell vs monocyte (‖Δφ̄‖₁ 0.60, ceiling
  0.09, 450 LC calls), skin basal vs keratinocyte, intestine epithelial vs intestinal — cell types that
  share programs. Mean-φ heatmap by cell type: SP1 myeloid, SP4 endothelial, SP6 lymphoid, SP9
  mesenchymal, SP15 keratinocyte/basal, SP16 astrocyte/oligodendrocyte, SP17 hepatocyte, SP18 islet,
  SP20 intestinal.
- Effect sizes per contrast (S3 effect_sizes_by_contrast, 6 contrasts): LeafCutter TV tracks the
  observed pooled TV (Spearman 0.65–0.71); posterior TV saturates at the contrast's ceiling.
- On LeafCutter's own large-effect set (S3 calibration_and_K c; A63 at δ 0.15): posterior recall 0.07
  vs 0.22, precision 0.75 vs 0.56 — conservative and precise.
- Rank robustness (S3 calibration_and_K d): posterior consensus calls 6,360 / 7,041 / 6,666 at K =
  10/20/30; LeafCutter agreement 46/50/50%; call-set Jaccard vs K = 20: 0.52 (K10), 0.71 (K30);
  per-contrast TV rank Spearman 0.93 / 0.97. LRT and LeafCutter arms do not depend on K.
- Seeds: 3 model seeds; consensus rule; TV rank consistency across seeds Spearman 0.90 within contrast;
  call-set Jaccard 0.30 across seeds due to per-contrast scale wobble → hence the ≥ 2-of-3 rule.

**Worked examples (Fig. 3d)**
- Same contrast, brain endothelial vs oligodendrocyte; junction arcs over annotated transcripts, per-
  junction usage in single cells (violin, one point per covered cell) with per-mouse pseudobulks overlaid.
- Apbb2 ATSE3, found by both: LeafCutter q = 3e-18; posterior TV 0.64–0.76; observed pooled TV 0.98;
  123 vs 314 covered cells.
- Mbd3 ATSE1, LeafletFA-only: 238 and 414 covered cells shift J1 usage 0.19 → 0.37 (observed TV 0.35;
  posterior 0.25–0.28 in all seeds); the 7 and 8 per-mouse pseudobulks overlap and LeafCutter reaches
  only q = 0.20 — the power the pseudobulk design gives up. Examples gated on ≥ 50 covered cells/arm and
  observed TV ≥ 0.25.
- Fbln2 program decomposition (S3 Fbln2, SP-labelled): how ψ per program and Δφ̄ combine into the
  predicted skip 0.55/0.10 vs LeafCutter 0.92/0.02; direction recovered in 3/3 seeds; ‖Δφ̄‖₁ 0.97 →
  ceiling 0.49.

**Duplicate reads / effective sample size (Methods; mention once in Results)**
- 68% of covered cells put all reads on one junction at a median of 48 reads → 1.7–3.1 effective
  molecules; the per-ATSE concentration γ_e (floored at 2) IS the effective sample size and removes
  ~20× overconfidence at source; measured, not assumed.

## Numbers still to fill (pending jobs)
- S2 imputation metrics: NMF cell-type-level r = 0.41 (0 and 90% removed; one NMF fit per level, deterministic).
- Per-dataset models (TMS-only, AB-only, human AB) — not part of Fig 2/3; feed Fig 4/5.
