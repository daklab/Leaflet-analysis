# Tracker — getting LeafletFA DS to a defensible model

Single source of truth for what we are trying to achieve, what is being tested, and what is
decided. Anything not on this list is out of scope until these are closed.

## The goal, in four properties

A LeafletFA differential-splicing model that is:

1. **Calibrated** — the reported confidence means what it says, comparable to LeafCutter
2. **Reliable** — stable across random seeds
3. **Sensible** — learned psi sums to 1 across the junctions of an ATSE
4. **Additive** — retains the advantages LeafCutter does not have

---

## Where each property stands

### 1. Calibrated — CAUSE CONFIRMED, CORRECTION DEMONSTRATED

Measured on simulations with known truth. Nominal 0.05:

| | effect 0.25 | effect 0.50 |
|---|---|---|
| LeafletFA realised FDP / power | 0.156 / 0.183 | 0.199 / 0.760 |
| LeafCutter realised FDP / power | 0.109 / 0.915 | 0.047 / 0.950 |

Root cause: the posterior is too narrow because the likelihood counts one ATSE observation once
per junction. Scoring m junctions independently against a shared denominator inflates the
effective sample size m-fold.

**A35 tested this directly** (jobs 19460660-65, all completed). Down-weight each junction's
log-likelihood by w and re-measure realised FDP:

realised FDP, effect 0.25 (target: at or below nominal)

| nominal | w=1 | w=(m-1)/m | w=1/m |
|---|---|---|---|
| 0.01 | 0.0298 | **0.0093** | 0.0116 |
| 0.02 | 0.0508 | **0.0163** | 0.0147 |
| 0.05 | 0.1101 | **0.0525** | 0.0203 |
| 0.10 | 0.1749 | **0.1059** | 0.0467 |

realised FDP, effect 0.50

| nominal | w=1 | w=(m-1)/m | w=1/m |
|---|---|---|---|
| 0.01 | 0.0631 | 0.0415 | 0.0152 |
| 0.02 | 0.0995 | 0.0725 | 0.0396 |
| 0.05 | 0.1618 | 0.1443 | 0.1149 |
| 0.10 | 0.2231 | 0.2153 | 0.2003 |

Power at nominal 0.05: effect 0.25 — 0.420 / 0.364 / 0.258; effect 0.50 — 0.749 / 0.739 / 0.703.

Reading:
- **(m-1)/m calibrates the moderate-effect regime**, 0.0525 against a nominal 0.05, and is
  conservative everywhere below it. Power cost 0.420 -> 0.364, a 13% relative loss.
- **1/m over-corrects** and costs 39% of the power for no calibration gain. Not the choice.
- **Neither fixes effect 0.50** (0.144 and 0.115 against 0.05). Effective n is therefore the
  dominant cause at moderate effects and only a partial one at large effects, where saturation
  at prob_de = 1.0 takes over — a different mechanism that weighting cannot reach.

**Decision: the Dirichlet-multinomial is justified.** (m-1)/m is a crude scalar stand-in for
exactly the correction the DM applies, and it already recovers calibration where the low-rank
model has power. The DM applies it per event with the right m and the right correlation
structure, so it should do at least as well without the flat power tax.

Two smaller, separate contributors:
- the estimator had no upper floor — 500/500 recorded as 1.0. **FIXED 2026-08-03**: Jeffreys
  count (n_exceed + 0.5)/(S + 1) applied at junction and ATSE level in `A26` and `A28`. Caps
  prob_de at 0.9990, so lfdr floors at 0.001 instead of 0.
- the FDP rule averages (1 - confidence) assuming ATSEs are independent. They are not; they share
  programs. **Still untested.** No likelihood change would fix this.

### 2. Reliable — PARTIALLY FAILING

- psi composition reproduces well across seeds (r = 0.95–0.96)
- psi within-ATSE **sum** does not (r = 0.51–0.66) — an unidentified direction
- **The reported uncertainty does not cover seed variation.** For the TV statistic, the posterior
  sd within a fit is 0.0137 and the spread across seeds is 0.0130 — a ratio of 0.92, with the
  across-seed spread exceeding the reported posterior sd for 47% of ATSEs. None of that variation
  comes from phi, which is held fixed.

### 3. Sensible — CONSTRAINT WORKS, BUT THE FIRST IMPLEMENTATION WAS UNSTABLE

Baseline: 86% of program-ATSE pairs have psi summing to [0.9, 1.1]; identical across K=10/20/30
and all seeds, so not a K or seed pathology. Scales with event size: 100% valid at 2 junctions,
85% at 3, 72% at 4-5, 47% at 6+.

The within-ATSE softmax (`A33.SimplexPsi`) gives **100.0% valid** — the constraint holds exactly
by construction, as it must. That is the only result from the first round: 1 of 6 jobs finished
(19460573) and even that one hit a non-finite loss at step 1143 and fell back to a step-700
checkpoint.

**Why it diverged, found 2026-08-03:** the max subtracted before exponentiating was a *global*
max over all K x J logits rather than a per-ATSE max. Any single large logit anywhere then drives
every other ATSE's exponentials to zero, its segment denominator to zero, and the fit to NaN.
Fixed with a true segmented softmax (`scatter_reduce(..., 'amax')` per ATSE). The earlier fit is
kept as `logs/PRE_SOFTMAX_FIX_simplex_seed2.npz` and should not be scored.

Seed variance and the high-effect guardrail are **still unmeasured** — they need >=2 completed
seeds per variant.

### 4. Additive — HOLDS, but must be stated honestly

- needs no replicates: LeafCutter cannot test 35 of 96 TMS cell types; `Lung_MONOCYTE` has 14
  donors of cells but only 2 pseudobulk samples
- robust to donor heterogeneity: on a real donor-split null, LeafCutter falsely calls 7.7% of
  events, LeafletFA 0.14%
- unsupervised, so the permutation null is free (no refit); LeafCutter needs ~16 min per permutation
- **but** that robustness is low power, not better calibration — see property 1

---

## Under test right now

| jobs | tests | decides |
|---|---|---|
| 24 x `a36_*` | Dirichlet-multinomial factorial on the Figure 2 sims: {betabinomial, dirichletmultinomial} x {beta, simplex} psi, 2 effect sizes, 3 seeds. Score with `A37_dm_factorial_report.py` | property 1. Does the exact effective-n correction close the gap that A35's (m-1)/m weighting left open at effect 0.50 (0.144 against nominal 0.05)? Decision rule is fixed in the A37 docstring. |
| 19460771-76 | psi within-ATSE simplex on the real atlas, baseline vs constrained, 3 seeds each, with the segmented-softmax fix | properties 2 and 3 at atlas scale. Balance is already answered (100%). These answer: does DS-call seed agreement rise, and does anything regress at high-effect events? |
| 19460778 | frozen-globals E-step recovering q(phi), K=30 seed 1 | property 1 and the downsampling panel. How much of the residual overconfidence is the missing phi uncertainty. |

## Implemented 2026-08-03 -- the Dirichlet-multinomial is in the package

`leafletfa/dirichlet_multinomial.py` plus `likelihood='dirichletmultinomial'` and
`psi_parameterization='simplex'` in `minibatch.py`, both opt-in, committed at `e563a26`. 20 unit
tests pin the segment log-probability against pyro's own `DirichletMultinomial`.

Three findings from building it, none of which were in the plan:

**The DM needs the simplex, and that was established by measurement rather than argument.** The
DM sees alpha only through conc * psi, so on its own the within-event SCALE of psi is
unidentified and drifts. On a compositional simulation, 3 seeds:

| likelihood | psi | recovery r | seed sd | median within-ATSE sum | valid |
|---|---|---|---|---|---|
| betabinomial | beta | 0.9918 | 0.0027 | 0.999 | 99.6% |
| betabinomial | simplex | 0.9975 | 0.0003 | 1.000 | 100.0% |
| dirichletmultinomial | beta | 0.9276 | 0.0040 | 1.259 | 17.5% |
| dirichletmultinomial | simplex | 0.9970 | 0.0003 | 1.000 | 100.0% |

DM alone lands at sums of 1.26 and 17.5% validity. Together they recover psi as well as the best
cell and cut seed-to-seed spread about 9x -- which is property 2, from a change aimed at
properties 1 and 3.

**The Figure 2 simulations do not satisfy the identity the DM needs.**
`00_simulate_splicing.py:366` draws each junction as an INDEPENDENT Binomial against the shared
ATSE total, so the junction counts sum to that total only in expectation. Measured: max gap 168
reads, 7.1% of (cell, ATSE) entries disagree. The TRUE psi is a proper composition (asserted at
line 233), so recovery targets are sound, but the emission is the one the current model assumes,
not the DM's. Handled by deriving N_e = sum_j y_j inside `dm_log_prob` rather than reading
`cell_by_cluster_matrix`. This makes A36 a conservative test: the sim favours the incumbent.

**The first simplex implementation had a numerical bug that would have recurred here.** The
softmax subtracted a global max over all K x J logits instead of a per-ATSE max, which underflows
every other event's denominator to zero and diverges to NaN -- the cause of the step-1143 failure
in the first atlas run. `segment_softmax()` in the package does it per event.

## Pre-existing, NOT caused by the DM work -- separate issue

`tests/test_basic.py` has 4 failures. Verified 2026-08-03 by running the same suite in a scratch
worktree at `9654761`, the commit before the DM work: the identical four fail there too.

    test_recovers_known_programs
    test_health_check_catches_symmetric_collapse
    test_chunked_training_recovers_the_same_psi_as_full_batch
    test_obs_scale_gives_chunked_training_the_full_data_posterior_width

All four exercise the legacy `LeafletFA` class (the `obs_scale` / `use_dense_mode` chunked path),
not `LeafletFAMinibatch`, which is what the atlas models and all of this DS work are trained
with. So they do not affect any current result -- but the first two are the recovery gate and
the health check, so they should not stay red. Out of scope for the calibration work; worth its
own pass.

## Decided, do not revisit

- K makes almost no difference: AUROC 0.643 / 0.652 / 0.658 at K=10 / 20 / 30 across 71 contrasts.
  The Fbln2 K=20 advantage did not generalise.
- LeafCutter is not a calibration target. It over-calls 7.7% under donor heterogeneity.
- Agreement with LeafCutter is worse on ATSEs whose composition is valid (0.575 vs 0.603 at K=20,
  same at K=10 and K=30), because violation tracks real signal. Fixing property 3 is therefore
  justified on identifiability grounds, not concordance grounds.
- The logit TV statistic beats raw TV (0.658 vs 0.623) consistently across seeds.
- S=500 draws is enough. Going to 1000 reduces the saturated set by only 8%.

## Next, once the above land

1. **Dirichlet-multinomial** — decided in favour by A35; write it against the same simulations and
   check it beats (m-1)/m on both FDP and power, at both effect sizes
2. Score the simplex round with `A34_simplex_experiment_report.py` (guarded 2026-08-03 against
   partial runs)
3. Rerun calibration with phi sampled, from 19460778
4. Build the empirical calibration from `DS_CALIBRATION/null_tv_*.npz` — the route that does not
   depend on fixing the model at all, and what Reviewer 1 actually asked for
5. Test the FDP-rule independence assumption (the one untested contributor)

## Explicitly not doing

- retraining the nine atlas models until the DM and the simplex are both settled
- any further exploration of K, of alternative statistics, or of LeafCutter concordance
