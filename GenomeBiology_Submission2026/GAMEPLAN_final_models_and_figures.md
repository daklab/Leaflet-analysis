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
| K in {10, 20, 30} x 3 seeds | robustness, NOT model selection | effective K is 8-13 regardless of nominal K |

### The one open decision: `dir_conc` alone, or `dir_conc` + the m/(m-1) weight

Both arms are training on the atlas now (`runs_final` = both, `runs_final_deltaonly` = delta only).

Simulation at the atlas's regime, 3 seeds per arm, scored against ground truth:

| arm | phi_r_true | phi_ari | effect size recovered | run-to-run CV |
|---|---|---|---|---|
| control (conc 50, full depth) | 0.978 | 0.998 | 0.973 | 0.012 |
| atlas regime, delta learned | 0.247 | 0.078 | 0.539 | 0.144 |
| delta = K | 0.577 | 0.189 | 0.642 | 0.106 |
| delta = K + m/(m-1) | 0.668 | 0.411 | 0.765 | 0.046 |

**Decision rule, fixed now.** Ship `delta = K` ALONE unless the weighted arm beats it on the atlas by
more than the simulation gap predicts — concretely, unless Fbln2 `skip_err` improves by >0.02 AND
across-seed spread of the skip contrast drops by >30%. Rationale: a weighted likelihood is a
tempered pseudo-likelihood, so its credible intervals are too narrow by construction, and a model
whose uncertainty means what it says is worth a modest loss in point accuracy. If the weighted arm
wins by more than that margin, ship it and state the tempering — with the composite-likelihood
sandwich correction named as the principled fix we did not implement.

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

---

## Part 2 — Figure 2 (simulation)

The current `FIG2_simulation.png` validates the **BetaBinomial and Binomial** models, neither of
which is shipped. It has to be rebuilt or it validates a method the paper no longer proposes.

### Row 1 — the model recovers what it claims

| panel | content | status |
|---|---|---|
| a | effect-size recovery vs simulated effect: LeafletFA (DM+simplex) vs NMF at matched rank | DONE — `data/dm_vs_nmf.tsv` |
| b | concentration recovery: true 10 / 50 / 200 -> 9.8 / 48.6 / 184 | ready (`Q6_overdispersion.tsv`) |
| c | psi recovery vs read depth, by effect size | ready (`Q5_recovery_by_coverage.tsv`) |

Panel a numbers (NMF scored at its BEST gauge, so the baseline is not handicapped):

    effect            0.10    0.25    0.50    1.00
    LeafletFA        0.207   0.478   0.724   0.851
    NMF 0-impute    -0.014   0.002   0.041   0.104
    NMF mean-impute -0.020   0.087   0.103   0.239

Panel c is the "imputation is denoising" result: recovery RISES with coverage (0.80 at 0.3 reads to
0.90 at 14 reads). A method scoring 1.0 at 1-2 reads would be fitting noise, so this is the right
direction and should be said so in the caption.

### Row 2 — and here is where it stops working

Built: `plots/FIG2_regime_phi_collapse.{pdf,png}`, from `build_regime_phi_panels.py`.

| panel | content |
|---|---|
| d | joint criterion: effect size recovered vs run-to-run spread |
| e | phi against `obsm['true_phi']` per arm — the quantity the atlas cannot measure |
| f | psi survives / phi collapses |

**Why row 2 is main-text, not supplementary.** The paper's claim is a factor model; phi is the
product. This is the only place phi's accuracy is quantified against truth, and it is what justifies
the shipped recipe. Asserting the recipe without it would be asserting a tuning choice.

### The open addition: where is the cliff

phi ARI is 0.000 at effect 0.10 and 0.997 at 0.50 — a cliff, not a gradient. Fits at effects
**0.15 / 0.2 / 0.35** are running to locate it. If the transition sits above the atlas's real
cell-type effect sizes, then the atlas phi collapse is the DATA not containing the signal, not a
modelling failure — which changes what we claim rather than what we tune. Add as panel g, or fold
into panel e as a second x-axis.

### Drop from Figure 2

- the BetaBinomial-vs-Binomial overdispersion panel — its point is made better by concentration
  recovery, and keeping it invites "why is neither of these the shipped model?"
- A36's DS calling table entirely. It runs the RETIRED fixed-delta TV rule and is anti-conservative
  (realised FDP 0.051 / 0.148 / 0.188 against nominal 0.05). Cite as motivation for the LRT if at
  all; never as current calibration.

---

## Part 3 — Figure 3 (differential splicing, vs LeafCutter)

Builds on `figures/DS_COMPARISON/PLOTTING_PLAN.md`, which stands. What follows is what CHANGED.

### Everything must be re-run on the final fits

A51 / A52 / A53 all ran on the collapsed-phi model (conc median 0.520). For A52 it probably does not
matter — its covariate is a binary indicator and phi only initialises a concentration. For A51 and
A53 it matters completely: the statistic is `Delta_j = dphi . psi`, so a collapsed phi makes every
Delta small. **The "0.53% called on a true null" may be an artefact of the collapse rather than
evidence of calibration**, and must not be quoted until re-run.

### The three arms, and what each is allowed to claim

| arm | design | claim |
|---|---|---|
| A49 `--design group` | binary cell-type indicator | matched head-to-head with LeafCutter. Isolates single-cell vs pseudobulk with the TEST fixed |
| A49 `--design phi` | learned loadings | the model-specific claim: DS across structure found without labels. LeafCutter cannot run this |
| A51 posterior | `P(TV > delta)`, permutation-calibrated | uses the denoised psi the LRT discards |

**Framing that has to be right.** The `group` arm shows that testing across thousands of single
cells beats 7-vs-8 pseudobulk samples. That is a DATA-DESIGN result, not evidence the model is
better — LeafCutter on equally sized pseudobulks would presumably do the same. Say so. The `phi`
arm is the model claim and cannot be a head-to-head, so frame it as *recovering the labelled
contrasts without the labels* — agreement, not superiority.

### The posterior test now runs with `--permute`

If the weighted arm ships, nominal `P(TV > delta)` is not usable: the tempered posterior is too
narrow by construction. The permutation null (shuffle cells between the two arms, keep arm sizes and
the sampled phi vectors) shares the posterior draws with the observed statistic, so the width enters
both identically and cancels.

**Known limitation, measured:** permutation drives `E[dphi] -> 0`, so the null has no location and
the test reverts to the point null "Delta != 0" — which called 24,422 of 24,475 events. Used ALONE
it is uninformative. The defensible rule is the INTERSECTION: call when `q_perm <= alpha` AND
`P(TV > delta)` is high. Permutation supplies detectability, delta supplies effect size.

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

## Sequence

1. 9 fits x 2 arms land (~3 h) -> A57 indexing checks -> A54 both-properties + Fbln2 + phi UMAP
2. Apply the Part 1 decision rule -> one configuration
3. Cliff fits land -> Figure 2 panel g -> assemble Figure 2 (rows 1 and 2)
4. Re-run the whole DS suite on the chosen fits -> Figure 3
5. Update `METHODS_dirichlet_multinomial.md` and `METHODS_differential_splicing.md` with the
   concentration story, the phi prior, the tempering (if shipped), and every "will not claim" above
