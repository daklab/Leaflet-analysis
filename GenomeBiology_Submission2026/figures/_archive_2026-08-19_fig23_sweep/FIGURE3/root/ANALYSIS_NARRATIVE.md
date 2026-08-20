# Figure 3 — the scientific account

What was compared, on what data, how each method defines differential splicing, what was
expected at each step, and what was actually found.

---

## The data both methods saw

- **Atlas:** 132,706 cells × 79,768 junctions, grouped into **24,475 ATSEs** (Alternative
  Transcript Structure Events — junctions sharing a splice site, so their usage is a
  composition).
- **Two sources:** Tabula Muris Senis (78,135 cells, SmartSeq2, **18 mice**, ages 3/18/24 m) and
  Allen Brain (54,571 cells, all nominally 2 m, **no usable donor id**).
- **Two count layers:** `y_cj` = reads on junction *j* in cell *c*; `T_cj` = total reads on that
  junction's ATSE. PSI = y/T.
- **Critical check before any comparison:** `T_cj` equals the within-ATSE sum of `y_cj` **exactly**
  (0/79,768 mismatches, three disjoint cell ranges). LeafCutter rebuilds its denominator as the
  within-cluster sum; LeafletFA uses `T_cj` directly. Had these disagreed, the two methods would
  have been estimating different quantities and no comparison would have been meaningful.

## How cells were grouped

- **LeafCutter requires replicates**, so cells were summed into pseudobulk samples:
  - TMS: **donor × tissue_cell_type** (donor is the biological replicate)
  - Allen: **tissue_cell_type × sex** (no donor available; 6 brain regions × 2 sexes)
  - ≥25 cells per sample → **654 pseudobulk samples**, 94.6% of cells retained
- **LeafletFA requires none** — it scores cells directly, grouped by the same labels.
- **Contrasts:** pairs of `tissue_cell_type` **within the same tissue**, so tissue is controlled by
  construction rather than by covariate. 71 TMS pairs + 3 Allen = **74 contrasts**, both methods
  scoring the identical pairs.

## How each method defines "differential splicing"

**LeafCutter** — frequentist, per-event, joint
- Models the whole junction-count vector of an ATSE as **Dirichlet-multinomial**, so PSI lies on
  the simplex *by construction* and ΔPSI sums to exactly 0 within an event.
- Likelihood-ratio test, full vs null → **one p-value per ATSE**; ΔPSI reported per junction.
- Covariates (sex, age) one-hot encoded. Needs ≥4 samples per group.

**LeafletFA** — Bayesian, low-rank, borrows across all cells and junctions
- Cell *c*'s usage of junction *j* is **ρ_cj = Σ_k φ_ck ψ_kj**: φ = per-cell loadings over K=30
  latent programs (simplex), ψ = program × junction PSI.
- **Beta-Binomial per junction, independent across junctions.**
- Group effect: **Δ_j = mean_A ρ_j − mean_B ρ_j = (φ̄_A − φ̄_B) · ψ_:,j**
- Posterior over Δ by sampling ψ from its variational posterior.
- ATSE score = **total variation, TV = ½ Σ_j |Δ_j|** — "what fraction of splicing mass moved".
  For a 2-junction ATSE this equals |ΔPSI| exactly.
- Significance: `prob_de = P(TV > δ)` with δ=0.1, then Bayesian FDR (posterior expected FDP).

---

## Hypotheses and what actually happened

**H1 — The two methods agree on which ATSEs are differential.**
*This was the figure's premise: "cell-type marker ATSEs concordant with pseudobulk LeafCutter".*
→ **Largely rejected.** AUROC for LeafletFA's TV ranking LeafCutter-significant ATSEs is
**0.62–0.66**; top-200 overlap ~20%. Performance is bimodal: 0.68 for well-separated pairs
(oligodendrocyte vs endothelial), **0.54–0.57 for lymphoid pairs** (Marrow B vs NK) — essentially
no signal. The claim as written is not supported.

**H2 — Effect sizes agree in magnitude.**
→ **Rejected.** Median slope of LeafletFA β on LeafCutter ΔΨ = **0.094** (~10× attenuation). On the
Fbln2 reference case, LeafCutter measures ΔΨ = 0.897 for the skip junction; LeafletFA gives 0.065 —
right sign, right rank (1st of 24,475 by prob_de), 14× too small.

**H3 — The attenuation is caused by LeafletFA not enforcing the compositional constraint.**
*Rationale: LeafCutter's ΔΨ sums to 0 within an ATSE; ours summed to −0.284 on Fbln2.*
→ **Tested and rejected.** Renormalising ρ within each ATSE to force the composition left AUROC
unchanged (0.620 vs 0.622). The violation is real and scales sharply with event size — **0.3% of
2-junction ATSEs violate by >0.05 vs 82% of 11+ junction ATSEs** — but it is not the cause.
*Underlying reason:* a per-junction Beta-Binomial is exactly the Dirichlet-multinomial's
**marginal**; what is lost is the joint dependence, not the marginal accuracy.

**H4 — Restricting to clean 2-junction ATSEs should improve agreement.**
→ **Rejected, direction reversed.** 2-junction AUROC 0.61 vs 0.66 for 3+. Simplex violation is not
the bottleneck.

**H5 — The real constraint is the low-rank structure.**
→ **Confirmed, and it is exact algebra.** Since ψ ∈ [0,1], **|β_j| ≤ |Δφ̄|₁** — a hard per-contrast
ceiling set by how differently the two cell types use the K programs. Across contrasts |Δφ̄|₁ spans
**0.07 to 1.69** and predicts the slope at **r = 0.87 (p = 7.6e-23)**. Heart endothelial vs
fibroblast has |Δφ̄|₁ = 0.56, so a true ΔΨ of 0.90 is *above what the model can represent at any ψ*.
Biologically: both are ~50–68% program 0, so they share most of their program mass.

**H6 — Group-vs-group DS reproduces across seeds better than factor-level DS.**
*Structural prediction: cell-type labels are fixed, so nothing depends on matching programs
between seeds.*
→ **Confirmed.** Pairwise **r = 0.87–0.93** across contrasts, versus the factor-level Jaccard of
0.51 that had driven the reproducibility pessimism. Promoting group-vs-group was correct.

**H7 — Cell-level label shuffling gives a falsely clean null.**
→ **Confirmed, starkly.** Under a donor-split null (same cell type both arms, so every call is
false), shuffling **cells** gives **0 false positives everywhere**; shuffling **donors** gives up to
185. Cell shuffling breaks within-donor correlation and would have certified perfect type-I control.

**H8 — LeafCutter is the calibrated anchor.**
*This was assumed throughout the plan.*
→ **Challenged.** On identical donor splits: **LeafCutter falsely calls 7.7% of tested events at
nominal 0.05; LeafletFA 0.14%.** Like-for-like on shared cell types, 8.49% vs 0.30%. Mechanism:
LeafCutter's Dirichlet-multinomial treats pseudobulk samples as exchangeable with only DM
overdispersion — **there is no donor random effect** — so genuine between-donor variance has
nowhere to go except into significance.

**H9 — LeafletFA is better in rare cell types.**
→ **Not established.** The downsampling test appeared to show 85% marker recall at 36 cells, but
the number of calls was **flat across the entire ladder** (2004/2010/1983/2007/2009/2008). Cause:
φ is a point estimate, so `prob_de` draws only from ψ and is **structurally blind to cell count** —
the FDP at 36 cells is identical to that at 1,737. Separately measured, |Δφ̄|₁ = 0.555±0.005 at
n=1,737 vs 0.567±0.069 at n=36: the point estimate is stable, but nothing in the calculation knows
the second is 14× less certain. Being fixed by recovering a φ posterior.

**H10 — Attenuated effect sizes mean worse estimates.**
→ **Rejected.** Predicting held-out donors' pseudobulk ΔΨ (scale-free R², so attenuation is not
penalised): LeafletFA **0.036** vs LeafCutter **0.021**, better on 64.6% of splits, 2.3× better on
well-separated contrasts. Shrinkage buys variance reduction. *Caveat: both R² are small (2–8%), and
splits reuse donors, so the 65% is optimistic.*

---

## What each approach is good and bad at

**LeafCutter**
- *For:* unbiased effect sizes; exact compositional model; established, interpretable p-values; no
  latent-structure assumptions; ΔΨ directly measures the observable.
- *Against:* needs ≥4 pseudobulk replicates — **34 of 96 TMS cell types testable**, and
  `Lung_MONOCYTE` has 14 donors' worth of cells but only 2 pseudobulk samples, so it is untestable
  at any threshold; no donor random effect → 7.7% false-positive rate under donor heterogeneity;
  discards single-cell resolution; refit per contrast (~16 min), so a permutation null costs a
  compute budget; noisy at small n (split-half reliability 0.61–0.68 at n=13).

**LeafletFA**
- *For:* no replicates required; unsupervised, so post-hoc label permutation is free and leaks
  nothing; far more conservative under donor heterogeneity (0.14% vs 7.7%); borrows strength across
  cells and junctions; better held-out prediction; per-cell resolution available; reproduces well
  across seeds (r = 0.87–0.93).
- *Against:* the **|Δφ̄|₁ ceiling** caps effect sizes (~10× attenuation) and makes cell types with
  similar program usage invisible (Marrow B vs NK: |Δφ̄|₁ = 0.07, no signal); models the DM's
  marginal not its joint, so the composition is learned rather than enforced; TV is not size-neutral
  at fixed δ (4.8× more calls for 6+ junction events vs 2-junction); uncertainty currently
  independent of cell count; weak rank agreement with the established tool.

---

## The one-sentence result

**LeafletFA is markedly more conservative than LeafCutter under donor heterogeneity and markedly
less sensitive to real effects — the same shrinkage produces both, so the figure should present a
characterised trade-off, not a concordance claim.**

## Method choices that fell out of the work

- **Statistic:** logit-scale TV (AUROC 0.658 vs 0.623 raw, consistent across 3 seeds) — and
  principled, since LeafCutter's own GLM is logit-link.
- **Posterior draws:** S≈100 suffices for effect sizes and ranking (mean TV r = 0.999 from S=50 to
  1000); **S ≥ 500 for anything FDR-related**, because ATSEs saturating at prob_de = 1.0 drop 28%
  over that range and those report FDP = 0 as a resolution artifact.
- **Replicate unit:** donors, never cells (H7).
- **Calibration reporting:** the permutation null measures **type-I error volume**; a genuine
  nominal-vs-realised FDR curve needs true positives and can only come from simulation.
