# What to test next

Rewritten 2026-08-03 evening, replacing the morning version. The morning plan proposed a
2x2x2 factorial of {simplex, effective-n, phi uncertainty} and listed the empirical null as an
independent side task. Most of that is now answered or reordered.

## What changed today

The Dirichlet-multinomial is implemented, tested and committed. It is not the calibration fix.

| claim in the morning plan | status this evening |
|---|---|
| effective n is the mechanism behind the miscalibration | correct, and now corrected exactly rather than by a scalar weight |
| the DM should close most of the remaining gap | **wrong.** It closes about half and stalls |
| saturation at prob_de = 1.0 is the second cause | **wrong.** Saturated ATSEs are the accurate ones -- 47-59% of calls, 2-5% of false discoveries |
| the FDP rule's independence assumption is an untested contributor | **cleared.** Predicted FDP tracks nominal exactly; dependence moves the variance, not the mean |
| mean-field narrowness is the contingency in reserve | **downgraded.** 1.3-1.7x too narrow on true nulls, and the DM's 1.29x already covers most of it |
| the empirical null is independent of the model work | **wrong.** It is the main line |
| A and B are separable factors that can be crossed | **wrong.** The DM needs the simplex; without it psi sums drift to 1.26 and calibration gets worse |

The cause, from `A38_where_the_miscalibration_lives.py`: TV = 0.5 * sum_j |Delta_j| is a sum of
absolute values, so estimation noise cannot cancel and a truly null ATSE scores TV 0.017-0.025
instead of 0. It is compared against a fixed delta = 0.10. At effect 0.25 the median TRUE
positive scores 0.096 -- below the threshold that 3.9% of nulls clear. Low power and high FDP
are the same fact.

## The one decision that needs input

The null has to be built by resampling DATA, not by drawing from q(psi) -- A39 shows a
posterior-derived null is 1.9x too tight and lands at FDP 0.579 against nominal 0.05. But
"resample the data" has a subtlety worth deciding deliberately rather than by default.

TV depends on the contrast through dphi = phibar_A - phibar_B, and null TV scales with |dphi|.
Any null built by shuffling cells between the two groups drives dphi toward zero, so its TV is
far too small and the test is anti-conservative for the same reason A39 was. Three options:

**Option 1 -- permute and refit, keeping the group structure.** Hold the cell-to-group
assignment fixed but permute junction counts within each ATSE across cells, then refit.
Preserves dphi and coverage, destroys the psi-program association. Most defensible; costs one
refit per permutation, ~23 min on a simulation and hours on the atlas.

**Option 2 -- donor-split null, which already exists.** `DS_CALIBRATION/null_tv_*.npz` holds
per-ATSE null TV from splitting one real cell type by donor. Both arms are the same cell type,
so every call is a false positive, but dphi is whatever donor heterogeneity produces -- NOT the
dphi of a real between-cell-type contrast. Free and already computed; the null sits at a
different effect scale than the test.

**Option 3 -- calibrate the threshold instead of the p-value.** Keep prob_de, but set delta per
contrast from the null TV distribution rather than fixing it at 0.10. Cheapest, directly targets
the diagnosis; weaker as a statistical claim, since it yields a calibrated cutoff rather than an
FDR.

Recommendation: Option 1 on the simulations, to establish that a data-resampled null does
calibrate, then Option 3 on the atlas, where a refit per permutation is not affordable. Report
Option 2 as the robustness result it already is, not as the calibration method.

## Ranked work

1. **Build the data-resampled null on simulations** (Option 1). The one thing that would
   actually deliver a calibrated FDR. Everything else is secondary.
2. **Finish reading the A36 factorial.** The decisive comparison is BB+simplex vs DM+simplex at
   matched power.
3. **Score the atlas simplex round** with `A34_simplex_experiment_report.py` for seed agreement
   and the high-effect guardrail. Balance is already answered at 100%.
4. **Decide whether the simplex is worth its power cost.** It costs about 16% relative power
   (0.420 -> 0.353 at effect 0.25) and buys exact compositions plus ~9x lower seed spread. A
   judgement call, not a measurement.
5. **Rerun calibration with phi sampled**, from the E-step job, to close out that contributor.

## Explicitly dropped

- the 2x2x2 factorial as designed -- the DM and the simplex are not separable
- a structured guide (AutoLowRankMultivariateNormal) as the next model change -- the width gap
  it would close is mostly closed already
- further work on the Jeffreys floor, saturation, or the FDP rule -- all measured as negligible
- retraining the nine atlas models until the null question is settled

## Framing for the paper, if the null work does not land in time

Report prob_de as a ranking statistic with an empirically chosen cutoff, and say so plainly.
Report the DM plus simplex for what they demonstrably deliver: junction usages that sum to 1
within an ATSE, ~9x lower seed-to-seed spread, and a likelihood 3x cheaper than the one it
replaces. Do not report a Bayesian FDR from the current machinery -- it is 1.4-2.4x
anti-conservative even in the best configuration measured today.
