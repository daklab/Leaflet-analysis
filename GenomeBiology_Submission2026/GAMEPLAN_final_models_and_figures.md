# Game plan: final model configuration, Figure 2, Figure 3

Written 2026-08-09, while the deciding runs are still in flight. Every decision below is stated as
a **rule applied to a measurement**, not as a preference, so the answer does not depend on who reads
the table. Where a result is not in yet, the rule is written first and the number filled in after —
that is the guard against what happened with `fbln2_dphi`, which was adopted as an acceptance
criterion after seeing the numbers and turned out to select the worst fit in the table.

---

## Part 1 — Final model configuration

### Decided, and not reopening

| choice | why | evidence |
|---|---|---|
| Dirichlet-multinomial likelihood | correct for compositional counts | term-for-term vs LeafCutter `dm_glm.stan` (−114.362812 vs −114.362793) and SpliceVI (−8.906084) |
| within-ATSE simplex for psi | psi is a composition by construction | 100% of program-ATSE pairs sum to 1 vs 86.9% under free psi |
| per-ATSE concentration, floored at 2.0 | sparse events read shot noise as overdispersion | free conc collapses to 0.519; Spearman(conc, reads) = 0.562; pooled MLE is 1.9966 |
| 16,200 steps, batch 8192, lr 0.025, gamma 0.1 | the only lr that does not diverge | lr sweep; `num_steps` sets the schedule, so it cannot be changed independently |
| K in {10, 20, 30} x 3 seeds | robustness, NOT model selection | effective K is sub-linear in nominal K: 13 of 20, 17 of 30 |
| programs survive changing the RANK | cross-K check, passed 2026-08-11 | 14 of K=20's 20 reappear at K=30, median r 0.96; 8-9 of K=10's at both higher ranks |

### DECIDED 2026-08-11: `dir_conc` alone. K=20 seed 0 ships.

Both arms landed (17 of 18 fits; one weighted K=10 seed died on a bad node). The rule was applied:
delta-only does not CLEARLY fail, so it ships.

Be accurate about why. Delta-only is **not** more reproducible. Pairwise cross-seed replication at
r>=0.9 puts the weighted arm ahead at K=20 (15.7 vs 14.3 of 20) and behind at K=30 (20.7 vs 22.0 of
30) -- a wash that flips with K. Delta-only wins on the Fbln2 contrast at K=20 (3/3 seeds recover
the direction, ||dphi||_1 0.97-1.11, no collapsed seed; the weighted arm has one at 0.12) and it
wins the ARGUMENT: the model-based DS test is a posterior test, so a tempered likelihood commits us
to permutation calibration permanently. Claiming it is more reproducible would be false.

Simulation at the atlas's regime, 3 seeds per arm, scored against ground truth:

| arm | phi_r_true | phi_ari | effect size recovered | run-to-run CV |
|---|---|---|---|---|
| control (conc 50, full depth) | 0.978 | 0.998 | 0.973 | 0.012 |
| atlas regime, delta learned | 0.247 | 0.078 | 0.539 | 0.144 |
| delta = K | 0.577 | 0.189 | 0.642 | 0.106 |
| delta = K + m/(m-1) | 0.668 | 0.411 | 0.765 | 0.046 |

**Decision rule, firmed up 2026-08-10.** Ship `delta = K` ALONE unless it CLEARLY fails — not
merely unless the weighted arm wins by a margin. Two reasons, and the second is the stronger:

1. **The posterior IS the product.** The model-based DS test is a posterior test, so a tempered
   likelihood means DS can only ever be calibrated through the permutation null. Honest uncertainty
   is worth more than a marginal phi gain when DS is a headline claim.

2. **The weight points the wrong way given what we now know.** Its justification was "restore what
   the BetaBinomial was accidentally doing" — but that behaviour was WRONG. The BB over-counted,
   scoring an ATSE's m junctions as m independent observations when only m-1 are free. Restoring it
   deliberately means deliberately over-counting. Meanwhile the molecular measurement says the true
   effective sample size is LOWER than the read count, not higher: ~2-3 independent molecules where
   the data shows 48 reads (METHODS_dirichlet_multinomial.md). `conc` already carries that
   correction — A_e = conc IS the effective sample size, and the floor at 2.0 sets it to the
   measured molecular reality. The m/(m-1) weight then inflates that same evidence by 1.5-2x,
   moving AWAY from the measurement.

That the weight empirically improves phi therefore suggests it compensates for some OTHER deficiency
— the phi prior, or the optimisation — by a mechanism that happens to work rather than one that is
right. Ship it only if delta-only cannot produce a usable model at all, and then state the tempering
explicitly, with the composite-likelihood sandwich correction named as the principled fix we did not
implement.

**What ships either way**

    likelihood dirichletmultinomial   psi_parameterization simplex
    conc_per_atse True                conc_min 2.0
    dir_conc K (pinned)               steps 16200  batch 8192  lr 0.025  gamma 0.1
    NO holdout

`--holdout-frac` is off deliberately. The held-out metric compares two SINGLE compositions — the
model's `mean(phi) @ psi` against the pooled training frequencies — and the latter is the maximum
likelihood single composition for that task, so the model cannot win except by chance. Every gain
was negative for that reason. A held-out claim needs `infer_phi` on the held-out cells, which is a
separate piece of work, not a flag.

### What we will NOT claim

- **phi is not fully recovered at atlas sparsity.** The recipe recovers ~58% of the phi-accuracy gap
  and ~74% of the reproducibility gap, not all of it. State the residual.
- **Effective K is 8-13 regardless of nominal K.** The K sweep is robustness, not model selection;
  there is no K-selection criterion here and we should not imply one.
- **Cross-seed agreement must be quoted live-only.** Dead programs match each other at r ~ 0.93
  because they converge to a common average, which inflates a naive "reproduced" figure.
- **The program filter's threshold is a CHOICE now.** On the old collapsed-phi fit the cross-seed
  distribution was bimodal -- 8 programs at r 0.978-0.998 against six below 0.6, nothing between --
  and that clean separation was what made the post hoc filter defensible. On the shipped fits K=20
  runs 0.993 -> 0.811 as a smooth gradient, largest gap 0.879->0.828. `r >= 0.9` is a chosen cutoff
  and must be described as one, not as a natural break in the data.

---

## Part 2 — Figure 2 (simulation) — BUILT 2026-08-11

`figures/FIGURE2_simulation_imputation/plots/FIG2_final.{pdf,png}`, from
`code/build_fig2_final.py`. Eight panels, replacing the old version entirely (every panel of which
validated the BetaBinomial and Binomial models, neither of which ships).

    a  psi recovery vs NMF at its BEST gauge        e  DS power vs LeafCutter
    b  concentration recovery (10/50/200 -> 10/49/183)  f  phi vs truth by regime -- the limitation
    c  psi recovery vs read depth                   g  delta: the calibration-power trade-off
    d  DS calibration vs LeafCutter                 h  what fraction of the atlas any delta reaches

**The headline.** LeafletFA's LRT is better calibrated than LeafCutter at EVERY simulated effect
size (realised FDP / nominal 5.99 -> 0.33 as effect grows, against LeafCutter's 7.94 -> 2.16) and
more powerful from effect 0.25 up. Ours reaches 1.0 around effect 0.30; **LeafCutter never does** --
still 2.16x over nominal at effect 1.0, where the difference is impossible to miss. Do not flatten
this into "both are anti-conservative at small effects".

**Panel g is how delta is chosen**, and it makes the choice a reading rather than a claim: plot
realised-FDP-over-nominal against power with delta as the point label. delta = 0.15 sits at ratio
1.19-1.97 with power 0.46-0.90 at effects >= 0.35. delta = 0.10 buys power at a 2-3x cost, but that
inflation is nearly CONSTANT across effects (2.02/2.25/2.66) and is therefore correctable by
tightening q to ~0.02; delta = 0.05's 5-6x inflation grows with effect size and cannot be corrected
by any constant. delta = 0.20 overshoots into conservative and discards signal for calibration
already had at 0.15.

**Panel h scopes every posterior-test claim in the paper.** On LeafCutter's own measurements, 28% of
atlas (contrast, ATSE) pairs exceed TV 0.10 and 18% exceed 0.15, median 0.045. The test speaks to
roughly the top fifth of the atlas. That belongs in the main text, not left for a reader to derive.

**Captions must carry two caveats.** Both methods are anti-conservative below effect 0.20; and the
simulations sit at median 33 reads per ATSE against the atlas's 16, so every power number here is an
OPTIMISTIC bound on real data.

### What the simulations cannot settle, and what replaces them

`effect` caps the achievable delta -- at effect 0.15 the largest true delta anywhere in the dataset
is 0.147, so delta >= 0.15 is untestable there by construction. Combined with the depth mismatch,
**delta = 0.15 is a LOWER BOUND for real data, not the answer.**

The replacement is `--permute-atse-frac`, now in both A36 and A44: permute a random 10% of ATSEs'
count vectors across cells, keeping each cell's N_e with its own vector. Those events are null BY
CONSTRUCTION while the untouched 90% still pin phi, so dphi keeps real magnitude. Their TV
distribution is the resolution floor at the right scale, on real data, with no truth labels.
delta = its 95th percentile (A64).

A64 runs a GATE first, on simulations where the true nulls are known: permuted TIGHTER than true
nulls means dphi collapsed and delta comes out too small; LOOSER means the permutation destroyed
more than the cell-to-usage association. A39 also looked sound on paper and was anti-conservative
for a reason only visible after it was built.

### Drop from Figure 2

- the BetaBinomial-vs-Binomial overdispersion panel — its point is made better by concentration
  recovery, and keeping it invites "why is neither of these the shipped model?"
- A36's DS calling table entirely. It runs the RETIRED fixed-delta TV rule and is anti-conservative
  (realised FDP 0.051 / 0.148 / 0.188 against nominal 0.05). Cite as motivation for the LRT if at
  all; never as current calibration.

---

## Part 3 — Figure 3 (differential splicing, vs LeafCutter)

Builds on `figures/DS_COMPARISON/PLOTTING_PLAN.md`, which stands. What follows is what CHANGED.

### Re-run on the shipped fit: DONE for seed 0

`LEAFCUTTER_CMP_DELTAONLY/`. A49 (phi and group), A51 (delta 0.05/0.10, and 0.15/0.20 running),
A52, A53, A50. The old numbers stay on disk for the before/after rather than being overwritten.
**A51/A52/A53 are still seed 0 only**; A49 seeds 1-2 are running.

### The four arms, and what each is allowed to claim

One thing changes per row, which is what makes each comparison readable.

| arm | data unit | covariate | test | uses denoised psi? |
|---|---|---|---|---|
| LeafCutter | pseudobulk, 7 vs 8 | cell-type label | DM GLM LRT | -- |
| A49 `--design group` | single cells | cell-type label | DM LRT | no |
| A49 `--design phi` | single cells | learned phi | DM LRT | no |
| A51 posterior | single cells | learned phi | `P(TV > delta)` | **yes** |

Calls per contrast at MATCHED realised FDP: LeafCutter 641, **group 840**, phi 559. At nominal
q<=0.05 LeafCutter calls 1,124 against phi's 197 -- that 5.7x gap is its miscalibration, not
sensitivity. To deliver a realised 0.05 LeafCutter needs q<=1.39e-3 and LeafletFA q<=3.16e-2.

- **group vs LeafCutter is a DATA-DESIGN result, not a model result.** Same test, same covariate;
  only the data unit changes. LeafCutter on equally many pseudobulks would presumably close it.
  Say so in the caption rather than letting the +31% read as a modelling win.
- **phi vs group (559 vs 840) is the price of the label-free claim** -- about a third of the calls.
  State it as a number.
- **The posterior test is the ONLY arm that uses what LeafletFA learns.** The LRT takes phi as a
  design matrix and refits every event's compositions from raw counts, so the denoised psi never
  enters it. Leading with the LRT alone reduces LeafletFA to a covariate generator and hollows out
  the paper's claim. This objection is correct and the figure has to answer it.
- **The posterior arm is not a head-to-head** -- LeafCutter cannot do an effect-size posterior at
  all. Compare it on the SAME question instead (A63): restrict to events where LeafCutter's own
  measured TV exceeds delta. At delta 0.10, posterior recall 0.101 vs LeafCutter 0.171, but
  precision 0.811 vs 0.681. Conservative and precise rather than sensitive.

### delta, and why it is not a free knob

delta is the effect size of interest, on the TV scale: fraction of the ATSE's splicing mass that
moved, [0,1], equal to |dPSI| at m=2. It is NOT the significance threshold -- that is `q <= 0.05`.
**delta says how big, q says how sure.**

Simulations put it at **0.15** (Part 2, panel g). But `effect` caps the achievable delta there and
the sims are 2x deeper than the atlas, so 0.15 is a lower bound. The atlas permuted-ATSE refit
measures the floor on the real data; delta = its 95th percentile.

**The posterior test was never broken.** delta = 0.05 sat below psi's own posterior sd (0.024-0.09
per junction), i.e. it asked the test to resolve an effect smaller than its uncertainty in the
quantity the statistic is built from. The old "conservative at every effect size" numbers (FDP
0.000/0.018/0.016) came from the collapsed-phi model where dphi ~ 0.3 made every TV small, and are
invalid regardless.

Three fixes were tried and measured, so they are not retried: q(psi) is NOT overconfident (spread
across programs on true nulls is 0.8x the posterior sd); `|E[Delta]|` removes only ~1/3 of the
floor; a rotation null at matched ||dphi||_1 is ~10x WORSE.

### `--permute` is a check, not the calibration

Delta-only ships UNTEMPERED, so nominal `P(TV > delta)` is a legitimate posterior probability --
unlike under the m/(m-1) weight, where permutation would have been the only option. Measured
limitation, unchanged: permuting drives `E[dphi] -> 0`, so the null loses its location and the test
reverts to the point null, calling ~24,4xx of 24,475 per contrast. Usable only as the INTERSECTION
rule -- `q_perm <= alpha` AND `P(TV > delta)` high.

**A53 cannot serve as the posterior test's calibration number.** Its null has ||dphi||_1 ~ 0.12
against ~1.0 for a real cell-type contrast, so it measures the floor ~8x too small. Its "0.76%
called on a true null" must not be quoted as calibration.

### Panels

Keep A-G from `PLOTTING_PLAN.md`. Two changes:

- **Panel A** must be annotated with the asymmetry, in the caption not a footnote: LeafCutter's null
  is clean (labels shuffled within a cell type); LeafletFA's permuted-phi null is contaminated by
  real signal and is therefore an UPPER bound.
- **Panel D** (power at matched REALISED FDP, not matched nominal q) is the panel a referee goes to
  first. A miscalibrated test calls more by construction, so comparing counts at nominal 0.05 would
  score LeafCutter's 4.48x inflation as sensitivity.

### Numbers to quote, with caveats attached

- LRT validation: FDP at or below nominal, power 0.84-0.99, 4 effect sizes x 3 seeds.
- Posterior test: FDP ratio 0.00 / 0.35 / 0.31, power 0.253 / 0.717 / 0.850 — **scored against
  contrast-specific truth**; scoring against "differential across any program pair" understates
  power ~1.5x.
- Matched-null calibration: LeafletFA 1.42x vs LeafCutter 4.48x at alpha 0.05; by event size,
  1.15 vs 3.23 (m=2) to 1.93 vs 11.35 (m=9). **On the OLD fit — re-run before quoting.**
- Our lower false-positive rate is partly LOW POWER, not purely better calibration. Report both.

---

## Sequence — where we actually are, 2026-08-11

DONE
1. ~~Both arms land -> A57 -> A54~~ 17/18 fits scored; A57 17/17 on the shipped fit.
2. ~~Apply the Part 1 rule~~ delta-only, K=20 seed 0.
3. ~~Cross-K replication~~ passed: 14 of 20 survive K=20 -> K=30.
4. ~~Figure 2~~ built, eight panels, `FIG2_final.{pdf,png}`.
5. ~~DS suite on the shipped fit~~ seed 0: A49 phi+group, A51, A52, A53, A50 (after fixing two
   silent A50 faults -- the null column and the cluster join key).
6. ~~delta chosen on simulations~~ 0.15, with the trade-off shown rather than asserted.

RUNNING
7. atlas permuted-ATSE refit -> A64 -> **delta measured on the real data**. The critical path.
8. 70/30 retrain on the MATCHED split -> imputation.
9. A51 at delta 0.15/0.20; A49 seeds 1-2.

NEXT
10. Explain the Jaccard: 0.134 for phi, 0.309 for group. The two methods agree on roughly the same
    NUMBER of events and disagree on WHICH. Decompose by event size, depth, effect size. Right now
    it is a bare number in a figure.
11. Make A51's two TV statistics one statistic. The reported `tv` is absolute-of-mean; calls use the
    per-draw mean-of-absolute. 1.4x apart over all events, 1.07x on called events, 4.1% of called
    events have a reported tv below delta -- small, but they should not be different functionals.
12. Imputation eval: the E-step and the downsampled data both exist. What is missing is a FORMAT
    ADAPTER -- A3 reconstructs psi as `sigmoid(loc + scale*eps)`, which is wrong for a simplex fit
    (softmax WITHIN ATSE) and would silently give psi that does not sum to 1.
13. Assemble Figure 3 from `PLOTTING_PLAN.md` panels A-G.
14. Update `METHODS_dirichlet_multinomial.md` and `METHODS_differential_splicing.md`: the
    concentration story, the phi prior, delta's derivation, and every "will not claim" above.
15. Characterise the aging program's junctions/genes; run allen_brain and TMS separately.

## The mistake worth not repeating

`--holdout-frac` picks a RANDOM 30%, but the downsampled evaluation layers exist for ONE specific
split of 39,812 cells. Same size, different set -- the model would have been clean but unscoreable.
`--holdout-ids` now takes the split's own row indices. Caught only because the retrain was checked
against the data it would be scored on; **100% of those test cells are in the atlas training file,
so every imputation number on file was scored on cells the model had seen.**
