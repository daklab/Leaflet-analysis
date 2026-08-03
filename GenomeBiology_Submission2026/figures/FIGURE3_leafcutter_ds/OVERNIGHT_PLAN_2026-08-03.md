# Overnight plan, 2026-08-03 — and what to read first in the morning

## Read this first: the FDR problem is now diagnosed, and the picture changed

Two results tonight, both uncomfortable, both important.

### LeafCutter beats us on the simulations, on both axes

Same simulations, same ground truth, same events. Nominal 0.05:

| | effect 0.25 | | effect 0.50 | |
|---|---|---|---|---|
| | realised FDP | power | realised FDP | power |
| **LeafCutter** | 0.109 | **0.915** | **0.047** | **0.950** |
| **LeafletFA** | 0.156 | 0.183 | 0.199 | 0.760 |

LeafCutter is calibrated at effect 0.50 and 2.2x anti-conservative at 0.25. We are 3.1x and 4.0x.
And the power gap is the bigger story: **0.183 vs 0.915 at effect 0.25** — a fivefold difference in
true positives found.

Caveat that matters: these simulations have no donor structure, and I had to build synthetic
replicates by splitting cells at random. That is LeafCutter's best case — its model assumptions
are exactly satisfied. On the real donor-split null it falsely called 7.7% of events against our
0.14%.

**So the honest reading is that our apparent conservatism on real data is not superior
calibration — it is low power showing up as few false positives.** When assumptions hold,
LeafCutter finds 5x more true signal and calibrates better. When donor heterogeneity is present,
it over-calls badly and we do not. Those are two different statements and the paper needs both.

### The miscalibration decomposes into two separable causes

Using the cached per-ATSE scores (`simscores_*.npz`):

**Cause 1 — prob_de is overconfident in exactly the range that drives discoveries.**
Binning ATSEs by prob_de and asking what fraction are truly differential:

| prob_de bin | n | actually differential |
|---|---|---|
| 0.00–0.10 | 34,487 | 0.107 |
| 0.10–0.30 | 1,684 | 0.353 |
| 0.30–0.50 | 1,057 | 0.403 |
| 0.50–0.70 | 935 | 0.446 |
| 0.70–0.90 | 1,075 | 0.574 |
| 0.90–0.99 | 900 | 0.758 |
| 0.99–1.00 | 2,542 | 0.969 |

Calibrated would put each bin's rate at its centre. It is roughly right below 0.5 and
systematically optimistic above it — 0.90–0.99 should be ~0.95 and is 0.758.

**Cause 2 — prob_de saturates at exactly 1.0, and the FDP rule then reports zero.**

| top n | rule's FDP | actual FDP |
|---|---|---|
| 100 | 0.0000 | 0.010 |
| 500 | 0.0000 | 0.024 |
| 2,000 | 0.0000 | 0.018 |
| 5,000 | 0.085 | 0.204 |

2,542 ATSEs have prob_de = 1.0 exactly, so 1 − prob_de = 0 and the rule computes an FDP of
literally zero for the entire top 2,000 — while 1.8% of them are false. This is the S-resolution
issue biting in the worst possible place, compounded by genuine overconfidence.

A Jeffreys-style estimate, (n_exceed + 0.5)/(S + 1), would cap prob_de at 0.999 rather than 1.0
and put a floor under the FDP. That is a one-line change and should go in regardless, but it only
buys a floor of 0.001 where the true rate is ~0.03, so it is a patch and not the fix.

---

## Running overnight

| jobs | what | expect |
|---|---|---|
| 19460660–65 | effective-n fix: 2 sims x {none, (m-1)/m, 1/m} | ~1.5 h |
| 19460568–73 | constrained psi vs baseline, 3 seeds each | running 2h14 |
| 19458963 | phi E-step, recovers q(phi) | running 3h51 |
| 19460622 | sim rescore on a finer threshold grid, caches fits | running 1h30 |

Already landed: LeafCutter-on-simulations (`LC_SIM/leafcutter_sim_calibration.tsv`), null-TV
distributions for 6 cell types (`DS_CALIBRATION/null_tv_*.npz`), the 9-model K sweep.

## In the morning, in this order

1. **`EFFN/effn_*.tsv`** — does down-weighting close the gap? Compare realised FDP at nominal
   0.05 across none / (m-1)/m / 1/m. If neither weighting moves it materially, effective sample
   size is not the dominant cause and the Dirichlet-multinomial would not fix it either. That is
   the single most decision-relevant number.
2. **`SIMPLEX_EXP/`** via `A34_simplex_experiment_report.py` — does constraining psi improve
   seed agreement, and does it cost anything at high-effect events.
3. **`DS_CALIBRATION/phi_posterior_*.npz`** — then rerun the calibration with phi sampled to see
   how much of the overconfidence it explains.
4. **Empirical calibration** using `null_tv_*.npz`: convert observed TV to an empirical p-value
   against the donor-split null distribution, then BH. This is the route that does not depend on
   fixing the model at all, and it is what Reviewer 1 actually asked for.

## What I would not do without seeing 1–3

Commit to the Dirichlet-multinomial. It fixes the joint likelihood and the effective sample size,
but the decomposition above says a meaningful share of the problem is the FDP rule and the
saturation, which the DM would not touch.

## Open thread I have not resolved

The FDP rule assumes the per-ATSE probabilities are independent. They are not — ATSEs share
programs, so their scores are correlated through phi and psi. The top-n table above cannot
separate that from the saturation effect, because both push in the same direction. Distinguishing
them needs the FDP rule evaluated on scores where the correlation is broken (e.g. permuting ATSE
labels within a prob_de stratum) and comparing. Worth doing before claiming the rule itself is
sound.
