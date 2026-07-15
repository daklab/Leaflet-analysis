# Figure 2 — simulated splicing data: methods and validation

Draft methods + validation for the benchmark/calibration figure.

**Code:** `pipelines/09_simulation/00_simulate_splicing.py`
**Notebook:** `figures/FIGURE2_benchmark/simulation_explore.ipynb`
**Data:** `SIMULATION/sim_heart_K5_binom_eff{1.0,0.0}_seed0.h5ad`

---

## Methods

### Rationale

Benchmarking a splicing factor model requires data where the answer is known. Purely synthetic
data is easy to get right and easy to make unrealistically clean; real data is realistic but has
no ground truth. We therefore simulate **counts under LeafletFA's own generative model while
conditioning on real coverage**: the ATSE read depth, dropout pattern and event structure are
taken verbatim from an observed tissue, and only the splicing *signal* is synthetic. This keeps
sparsity and depth realistic while making the latent programs, the per-program splicing levels,
and the differential/non-differential status of every event known exactly.

### Event type: cassette exons only

We restrict the simulation to **cassette (exon-skipping) events** — the simplest and most
interpretable alternative splicing event, and the one for which "percent spliced in" is a single
well-defined number:

```
upstream exon ---J1---[ cassette exon ]---J2--- downstream exon
              \________________J3 (skip)_______/
```

Each cassette has exactly three junctions: two inclusion junctions (J1, J2) and one skip junction
(J3) that spans them. The entire event collapses to one parameter per program — the skip usage
psi — which makes the ground truth directly interpretable.

Cassettes are identified from **real junction coordinates**, not assumed from a sort order: the
skip junction must be the one spanning the other two (minimum start, maximum end), and the two
inclusion introns must abut it and not overlap each other. Of 5,566 three-junction ATSEs present
in mouse heart, **4,268 (77%) satisfy this canonical cassette structure**; the remainder
(alternative 5′/3′ splice sites, mutually exclusive exons) are excluded rather than mislabelled.

### Source data and coverage

Real ATSE coverage was taken from **mouse heart** (Tabula Muris Senis SmartSeq2, 6,986 cells,
8 broad cell types) in the filtered, ATSE-regrouped splicing object. Junctions covered in fewer
than 50 cells were dropped, and a cassette was retained only if **all three** of its junctions
survived, giving **4,268 cassettes / 12,804 junctions**. The observed `cell_by_cluster_matrix`
(ATSE read totals per cell) is used **unmodified** as the denominator of the simulation, so
depth, coverage and zero-inflation are those of real data: 84% of (cell, junction) entries have
zero coverage and contribute no counts, exactly as observed.

### Generative model

For cell *i*, cassette *a*, junction *j* ∈ *a*:

1. **Programs (phi).** Each cell is assigned to one of *K* latent programs, seeded from its real
   broad cell type so that the ground-truth structure is biologically organised rather than
   arbitrary. Its program mixture is then drawn from a Dirichlet peaked on that program
   (concentration 20), giving a genuine mixture on the simplex (mean maximum weight ≈ 0.74)
   rather than a hard assignment.
2. **Splicing levels (psi).** For each cassette, a baseline skip usage is drawn from
   Beta(0.5, 0.5) — a U-shaped prior, so most events are predominantly included or predominantly
   skipped, as observed. A fraction of cassettes (default 30%) are designated **differential**:
   for these, skip usage is drawn independently per program. The remaining cassettes are **null**:
   skip usage is identical across all *K* programs. An `effect` parameter in [0, 1] interpolates
   between the shared baseline and the independent per-program draw, so effect size is tunable and
   `effect = 0` yields a **pure-null dataset**.
3. **Junction probabilities.** P(J3) = psi; P(J1) = P(J2) = (1 − psi)/2, so the three
   probabilities sum to 1 within the event (asserted at runtime).
4. **Per-cell probability.** p_ij = Σ_k phi_ik · psi_kj.
5. **Counts.** Either of two emissions, selected with `--bb-conc`:

   **Binomial** (`--bb-conc inf`, default) — no overdispersion:
   ```
   n_ij ~ Binomial(N_ia, p_ij)          Var(n) = N·p(1−p)
   ```

   **Beta-Binomial** (finite `conc`) — overdispersed:
   ```
   q_ij ~ Beta(p_ij·conc, (1 − p_ij)·conc)
   n_ij ~ Binomial(N_ia, q_ij)
   ```
   The Beta has mean p for *any* conc, and variance p(1−p)/(conc+1), so marginally

   ```
   Var(n) = N·p(1−p) · [ 1 + (N−1)/(conc+1) ]
                        └── overdispersion factor ──┘
   ```

   As conc → ∞ the factor → 1 and the Beta collapses to a point mass at p, recovering the
   Binomial exactly. In the implementation `conc = inf` **short-circuits the Beta draw and calls
   the same `rng.binomial`**, so the two code paths are provably identical rather than merely
   close (verified below).

### Why both emissions are required

LeafletFA can *assume* either likelihood (`input_conc_prior`: `inf` = Binomial, `None` = learn a
Beta-Binomial concentration). The robustness-and-calibration question is what happens when the
model's assumption and the truth disagree:

| truth ↓ / model → | Binomial | Beta-Binomial |
|---|---|---|
| **Binomial** | correct | should be fine (learns conc → large) |
| **Beta-Binomial** | ⚠️ **overconfident** — genuine overdispersion is attributed to signal, inflating false positives | correct |

Filling that 2 × 2 is the point of the figure, and it requires a simulator that can generate
overdispersed truth — hence the Beta-Binomial arm.

### Centred PSI

`junc_ratio` (= junction count / ATSE total) and `junc_ratio_centered` are computed with
`leafletfa.waypoints.calculate_centered_psi` (rho = 0.1) — **the same function used on real data**
in `00_extract_train_splice.py`. This is not a naive column mean-subtraction: each PSI is weighted
by the beta-binomial variance implied by its coverage, so a PSI estimated from 2 reads is
down-weighted relative to one estimated from 200. Reusing the pipeline's own centring means PCA on
simulated data is directly comparable to PCA on real data.

### Ground truth recorded

| field | content |
|---|---|
| `var['is_differential']` | event truly differs between programs (calibration label) |
| `var['junction_role']` | J1 / J2 / J3 |
| `obs['true_program']` | program each cell was assigned to |
| `obsm['true_phi']` | (N, K) program mixtures |
| `uns['true_psi_skip']` | (K, n_cassettes) skip usage per program |
| `uns['atse_delta_psi']` | max − min skip usage across programs (effect size per event) |
| `uns['sim_params']` | all settings and the random seed |

---

## Validation

The simulated data was checked against its own ground truth before any model was run on it.

**1. The truth is recoverable.** Pooling cells by their true program and re-estimating each
cassette's skip usage from the simulated counts recovers the planted values with **Pearson
r = 0.99** (restricted to program/event combinations covered in ≥100 cells). Recovery degrades
gracefully with coverage (r = 0.97 with no coverage requirement), as expected from sampling noise.

**2. Null events show no false signal.** For cassettes designated non-differential, the empirical
spread of skip usage across programs is **Δψ ≈ 0.000** (median, ≥100 cells), versus a clear
separation for differential cassettes. With no coverage requirement the null median rises to 0.058
purely because programs seeded from rare cell types (as few as 23 cells) give noisy per-program
estimates — this is estimator noise, not planted signal, and it disappears once programs are
adequately covered. This is worth stating explicitly: **an unbalanced K will look like
miscalibration if power is not accounted for.**

**3. Coverage is realistic, not idealised.** Because ATSE totals are inherited from real heart
data, 84% of (cell, junction) entries have zero coverage and the PSI distribution is U-shaped
(consistent with the Beta(0.5, 0.5) prior and with real splicing).

**4. Structure is present in the signal dataset and absent in the null.** PCA on the centred
junction usage ratio separates the true programs in the `effect = 1.0` dataset. The `effect = 0.0`
dataset — same cells, same coverage, same phi, but **no** cassette differing between programs —
serves as the false-positive control: any program structure recovered there would be an artefact
of coverage or Binomial noise rather than signal. This is the most important control in the
figure, and is quantified in the notebook with a silhouette score of the true programs on the
leading PCs.

**5. The Beta-Binomial path reduces exactly to the Binomial.** At `conc = inf` the Beta-Binomial
emission produces **byte-identical counts** to the Binomial arm under the same seed (the Beta draw
is short-circuited), confirming the new code path does not perturb the validated behaviour. And
overdispersion behaves as the theory predicts — empirical variance of PSI relative to the Binomial
baseline, at median coverage:

| conc | 1000 | 100 | 20 | 5 | 1 |
|---|---|---|---|---|---|
| variance vs Binomial | 0.97× | 1.01× | **1.21×** | 1.75× | 3.18× |

monotonically increasing as `conc` shrinks, and → 1× as `conc` → ∞. The `conc = 20` dataset used
for the figure carries a **2.5× overdispersion factor at median ATSE coverage (N = 33)**.

---

## Parameters used

| parameter | value | note |
|---|---|---|
| tissue | Heart | 6,986 cells, mouse SmartSeq2 |
| cassettes | 4,268 | 12,804 junctions |
| K | 5 | programs seeded from broad cell type |
| frac differential | 0.30 | 30% of cassettes truly differ |
| effect | 1.0 / **0.0** | signal dataset / **pure-null control** |
| emission | Binomial / Beta-Binomial | `--bb-conc inf` / `--bb-conc 20` |
| psi prior | Beta(0.5, 0.5) | U-shaped |
| phi concentration | 20 | mean max weight ≈ 0.74 |
| min cells per junction | 50 | |
| seed | 0 | |

### The four datasets (the calibration grid)

| file | truth | signal |
|---|---|---|
| `sim_heart_K5_binom_eff1.0_seed0.h5ad` | Binomial | 30% differential |
| `sim_heart_K5_binom_eff0.0_seed0.h5ad` | Binomial | **pure null** (FPR control) |
| `sim_heart_K5_bb20_eff1.0_seed0.h5ad` | Beta-Binomial (conc 20) | 30% differential |
| `sim_heart_K5_bb20_eff0.0_seed0.h5ad` | Beta-Binomial (conc 20) | **pure null** (FPR control) |

Reproduce with:

```bash
S=pipelines/09_simulation/00_simulate_splicing.py
# Binomial truth
python $S --tissue Heart --K 5 --frac-differential 0.3 --effect 1.0 --seed 0
python $S --tissue Heart --K 5 --frac-differential 0.3 --effect 0.0 --seed 0   # null
# Beta-Binomial truth (overdispersed)
python $S --tissue Heart --K 5 --frac-differential 0.3 --effect 1.0 --bb-conc 20 --seed 0
python $S --tissue Heart --K 5 --frac-differential 0.3 --effect 0.0 --bb-conc 20 --seed 0
```

Fitting LeafletFA on each with `input_conc_prior = inf` (Binomial) and `= None` (Beta-Binomial)
fills the 2 × 2 calibration table.

**A note on K.** K = 5 was chosen over K = 8 because programs seeded from heart cell types are
very unbalanced (2,880 / 1,898 / 889 / 864 / … / 23 cells). At K = 8 the smallest programs have
too few cells to estimate PSI reliably, which reads as miscalibration when it is really low power.
K = 5 merges the rare types into a single program and keeps every program adequately powered.

---
---

# Part 2 — The benchmark: what we sweep, and why

*Everything above describes the DATA. This half describes what we do with it: the parameter sweep in
`simulation_explore.ipynb`, why each comparison exists, and what each metric can and cannot tell you.*

The simulation is the only place in the whole project where **ground truth exists**. The atlas cannot
tell you whether ψ is right, whether the FDR is controlled, or whether a configuration is better than
another — it can only tell you whether something is *reproducible* or *decodable*. So every claim about
model behaviour has to be established here first, and the sweep is the instrument that does it.

## 2.1 What "recovery" means — the metrics

Programs are identifiable only **up to a permutation**, so every learned ψ is Hungarian-matched to the
truth before scoring.

| metric | question it answers | chance |
|---|---|---|
| `psi_r` | did we recover the **programs**? Median matched correlation with true ψ | 0 |
| `auroc` | can the model rank a **differential** cassette above a **null** one? | 0.5 |
| `fdp`, `power` | is the **FDR actually controlled**? Realised false-discovery proportion at nominal α | — |
| `agree` | did each cell get its **right program**? Dominant-program agreement | 1/K |
| `ari` | do the inferred φ **cluster cells into the true programs**? Adjusted Rand Index | 0 |
| `psi_contrast` | *diagnostic.* Are the K programs different from each other, or collapsed onto one? | ~0 = collapse |
| `bb_conc` | *diagnostic.* When ψ is wrong the Beta-Binomial collapses the concentration to absorb the residual | true value |
| `elbo` | the only quantity available **without ground truth** — does it track recovery? | — |

`agree` and `ari` are what answer *"does φ recover cell types"*. `auroc` + `fdp` answer *"does it find the
right differentially spliced junctions"*. Reporting only `psi_r` would hide a model that recovers the
programs but cannot use them.

**Two metrics that lie, and must not be used alone:**

- **`psi_contrast` is necessary but NOT sufficient.** It detects *total* collapse, not *partial* failure.
  Measured: a run scoring 0.045 passed a 0.04 health gate while true-ψ recovery was only **r = 0.22** and
  differential-vs-null AUROC was **0.525 — chance.** Always score against ground truth here.
- **Scale-invariant metrics cannot see effect-size attenuation.** Hungarian-matched Pearson r, cosine
  similarity and argmax agreement all stay ≈ 1.0 even if every program is shrunk toward a common
  per-junction mean — i.e. even if the DS effect size β collapses by 4×. Any chunking or posterior-width
  claim built on them is uninformative by construction.

## 2.2 The design: one axis at a time, plus the crosses that actually interact

A full factorial over every axis (K × likelihood × lr × ψ-prior × waypoints × batching × seeds) is
**576+ fits** and buys almost nothing — most cells answer questions nobody asked. Instead: vary one axis
at a time from a fixed **reference**, and fully cross only the pairs where an interaction is mechanistically
expected.

**Reference:** `K=5` (the true K), Beta-Binomial, `lr=0.05`, `gamma=0.5`, early stopping **off**, random
init, full-batch.

**Five seeds per configuration. This is not optional — see §2.4.**

### The crossed blocks, and why each one exists

**A. `lr × likelihood`.** The Binomial is a much *sharper* likelihood than the Beta-Binomial, so an `lr`
tuned on one is not valid for the other. Sweeping them independently would confound the two.

**B. `lr × waypoints` — the mechanistic test.**
The model begins **on a symmetric saddle**: `AutoDiagonalNormal` initialises every cell's φ at the prior
median and all K programs at nearly the same ψ. When φ is uniform, every ψ_k receives an *identical*
gradient, so the programs cannot separate until gradient noise breaks the symmetry — and **the rate of that
escape scales with the learning rate**. That is why `lr = 0.001` sits at chance no matter how many epochs
you run.

But **waypoint initialisation starts you off the saddle entirely**: it seeds each program at a distinct,
data-supported anchor. So the theory makes a falsifiable prediction:

> **Waypoints should rescue low `lr`, and should collapse the seed variance.**

If that holds, the saddle mechanism is confirmed and *"waypoints converge faster"* becomes a quantitative
claim. If it fails, the saddle story is wrong. **Waypoints were previously tested only at the one `lr`
where everything already works — which could not have told us anything.**

**Measured (100 fits, 5 seeds/config):** the prediction holds decisively.

| `lr` | random init (mean ± sd) | waypoint init (mean ± sd) |
|---|---|---|
| 0.005 | +0.452 ± **0.334** | +0.874 ± **0.006** |
| 0.01 | +0.625 ± 0.325 | +0.982 ± 0.001 |
| 0.02 | +0.768 ± 0.241 | +0.995 ± 0.000 |
| 0.05 | +0.862 ± 0.149 | +0.997 ± 0.000 |

Mean seed sd of `psi_r`: **random 0.262 → waypoints 0.002 (a ~130× reduction).** Worst single run:
**random +0.103 → waypoints +0.865.** The learning rate matters *only* because it sets escape velocity from
the symmetric start; waypoints start you off the saddle, so `lr` stops mattering and recovery becomes
near-deterministic. This is the direct answer to the SVI-local-optima objection: **not "we needed waypoints
to work at all", but "programs are recovered either way; waypoints make it reproducible".**

**C. `waypoints × batching`.** Their **combination is the atlas production configuration** (the mouse runs
use waypoint init *and* chunked training). Testing "waypoints, full-batch" and "random, chunked" as
separate one-axis arms never touches the cell we actually ship.

**Measured:**

| init | batching | `psi_r` | ARI |
|---|---|---|---|
| random | full | +0.862 | 0.862 |
| random | chunked | +0.906 | 0.941 |
| waypoints | full | +0.997 | 1.000 |
| **waypoints** | **chunked** | **+0.956** | **0.998** ← atlas production |

The production config is excellent. *(A first version of this table showed waypoints+chunked at 0.724/0.494
— the worst cell — which read as "the atlas config is broken". It was a benchmark bug, not a production one:
`train()` re-reads the waypoint init from `adata.varm` when `waypoints_use=True` and ignores the `psi_init`
argument, so the packaged `fit_chunked` was silently discarding the chunk-to-chunk warm start on every chunk.
`run_training.py` routes the carried ψ through `varm` and was never affected; `fit_chunked` now does the same.
It is a cautionary example of §2.1's rule — a benchmark that re-implements production will drift.)*

It is also where an interaction is
plausible: chunking re-seeds ψ from the previous chunk on every chunk, which may already be doing the
re-anchoring that waypoints exist to provide.

**D. `ψ prior × waypoints`.** `ψ_kj ~ Beta(a_j, b_j)` indexes `a_j`/`b_j` by **junction only — there is no
`k`**, so all K programs at a junction are drawn from the *same learned* Beta. That is a **shrinkage prior
pulling the programs toward each other**, i.e. toward the saddle. Waypoints push them apart. **Waypoints may
be exactly what has been masking the shrinkage**, which would explain why the prior looks harmless in the
paper's configuration. The paper uses `junc_specific_prior=True` and it had never been tested.

**E. `ψ prior × K`.** More programs = more for the shared `Beta(a_j, b_j)` to pull together, so the
shrinkage should get **worse as K grows**. The atlas runs K = 10/20/50; testing the prior only at K = 5
cannot see this.

## 2.3 Full-batch vs chunked — and what `obs_scale` is for

The atlas cannot be fit full-batch (13.8% dense, 1.46e9 nnz → ~305 GB), so it is trained in multi-pass
minibatches. The simulation is small enough for both, which makes it the only place the two can be compared.

The chunked arm calls **`leafletfa.chunked.fit_chunked()` — the same code production runs.** This matters:
the notebook previously *hand-copied* the training loop, and the copy drifted (it dropped the per-cell φ
carry-over and invented a final frozen-ψ E-step that production never runs), so the comparison that
justifies chunking for the entire atlas was measuring a scheme nobody uses. The three things a hand-copy
always gets wrong:

1. **globals** (ψ, π, α) are warm-started chunk → chunk — everyone gets this right;
2. **φ is carried PER CELL** — written back after every chunk, warm-started whenever the cell reappears, so
   it is continuously refined across passes;
3. **the final φ IS that accumulated store** — there is no final E-step.

`obs_scale = N/B` rescales the **local** terms (likelihood + the `assign` prior/entropy) so each chunk is an
unbiased estimate of the **full-data** ELBO, with the global KL at weight 1. Without it, each chunk
optimises the ELBO of a dataset that genuinely has only B cells, and `q(ψ)` ends up with the width it would
have given B cells — **too wide by ≈ √(N/B)**.

This is not cosmetic: **the differential-splicing FDR samples from `q(ψ)`.** `P(|Δψ| > δ)` is computed from
those draws, so a wrong posterior width miscalibrates every reported probability. Measured
(`tests/test_basic.py::test_obs_scale_gives_chunked_training_the_full_data_posterior_width`):

| | mean posterior sd of ψ |
|---|---|
| full-batch | 0.0182 |
| chunked **with** `obs_scale` | 0.0205 (13% wide) |
| B-cells-only, no rescaling | 0.0385 (**2.1× too wide**) |

Theory predicts √(N/B) = √(600/150) = **2.0×**. Measured **2.1×**, and `obs_scale` recovers the full-data
width. **This is the result that licenses sampling the chunked posterior for the FDR**, and it had never
been checked — every previous chunking check used only scale-invariant metrics (§2.1) and was structurally
incapable of detecting it.

## 2.4 Seed variance is a result, not a nuisance

SVI is routinely criticised for landing in different local optima from different starts, and this model is
a genuinely hard case (§2.2B). So reproducibility is measured, not assumed.

**Measured on the reference configuration — the one that works:**

```
seed 0:  psi_r +0.983   ari 0.998   elbo 1.5170e+07
seed 1:  psi_r +0.813   ari 0.774   elbo 2.2043e+07
seed 2:  psi_r +0.631   ari 0.644   elbo 2.4649e+07
```

**Single-seed fits vary enormously even at the good learning rate.** Pooled seed sd of `psi_r` = **0.25**.

This has a hard consequence for the design: with n = 3 seeds the standard error of a configuration's mean
is 0.145, so **two configurations must differ by ≈ 0.41 in `psi_r` before the difference is even
detectable.** At n = 3, *every axis except the learning rate was buried in seed noise* — likelihood, K,
ψ prior and batching all showed gaps of 0.03–0.08, which mean nothing. **Any claim read off those means
would have been noise.** Hence n = 5, and hence the estimand below.

### The right estimand: best-of-N by ELBO, not the mean over seeds

**Nobody deploys a random seed.** The Methods specify fitting three times and keeping the best ELBO, so the
quantity that describes *how the model is actually used* is the **best-ELBO run**, not the average over
seeds. Averaging over seeds answers a question no user asks.

That is only legitimate if the ELBO actually tracks recovery — otherwise selecting on it is unjustified.
So it was measured directly:

```
ELBO-selection regret (best psi_r  -  psi_r of the seed the ELBO picked):
   mean 0.001,  worst 0.009
ELBO vs psi_r across all configurations:  rho = -0.77  (p = 3.8e-08)
   (loss, so LOWER is better)
```

**The ELBO reliably picks the best seed.** This is a load-bearing result: on the atlas there is no ground
truth, so model selection *has* to run on the ELBO (or a held-out metric). Without this number, atlas model
selection would be an assumption. With it, it is licensed.

**Report both**, because they answer different questions:
- **spread across seeds** → *reproducibility* (and it is large — restart selection is necessary, not a
  formality);
- **best-ELBO run** → *performance as deployed*.

## 2.5 Differential splicing: the FDR must be the real procedure

The empirical false-positive rate is Reviewer 1's explicit request (*"the manuscript does not provide a
clear empirical estimate of false positives"*), and it is easy to get wrong.

**It must run the model's actual decision procedure**, not a threshold on the point estimate:

1. draw S posterior samples of ψ → `psi_samples [S, K, J]` (in `model.all_samples["psi"]`; note
   `psi_samples` only exists after `prune_K()`);
2. per cassette *e* and sample *s*, the effect is the cross-program spread of the skip junction:
   `β_se = max_k ψ[s,k,e] − min_k ψ[s,k,e]`;
3. `P_e = (1/S) · #{ s : β_se > δ }` — the **posterior probability** of a real effect;
4. `bayesian_fdr(P, α)` keeps the **most permissive** set whose posterior expected FDP ≤ α
   (max-detections; `FDP(n)` = running mean of `1 − P` over the top *n*).

An earlier version of this notebook simply called `effect > δ` a discovery — no posterior, no
`P(|β| > δ)`, no FDR control — and reported an alarming realised FDP of **0.292**. **The model was fine; the
metric was meaningless.** Verified on a known input, the real procedure gives realised FDP **0.029** at a
nominal 0.05 with power 1.00, and on a pure null calls **0 / 1000**.

**The null simulation (`effect = 0.0`) is the number to lead with**: every discovery there is a false
positive *by construction*, so it is a direct empirical FPR — not an argument from posterior width, which is
not characterised and whose error is not even consistent in sign.

## 2.6 Why early stopping is OFF in every simulation fit

All sweep fits use `patience = 10**6` — early stopping is **disabled**. This is deliberate: early stopping
has silently truncated these fits **twice**, in two different ways, and neither was visible in the loss
curve or in any health check.

1. **`min_delta` is ABSOLUTE.** Against an ELBO of 1e6–1e9, a value of 0.01 (or even 100) is a relative
   tolerance of ~1e-7 — numerically inert. The rule degenerated to *"stop after `patience` epochs with no new
   global best"*, which 1-particle SVI noise trips almost immediately. On the real atlas logs: **9,239 /
   9,239 batch fits ended by early stopping, at a median of 10 epochs** against a nominal 500–1000.
2. **The fix for (1) was itself broken.** `best_loss` starts at `+inf`, and the relative threshold was
   computed against it: `min_delta_rel * inf = inf`, so the test became `loss < inf − inf` → `loss < nan` →
   **always False**. `best_loss` never left infinity, the patience counter incremented every epoch, and the
   fit broke at **exactly `patience`** regardless of how fast the ELBO was still descending. Early stopping
   was not a convergence rule at all — it was a **hard epoch cap**.

**The tell is `n_epochs_ran == patience` exactly.** Log epochs-per-fit against the schedule that was
supposed to run; `psi_contrast` was a perfectly healthy 0.208 on a batch that had been cut off mid-descent.

## 2.7 Reproducing the sweep

```bash
cd figures/FIGURE2_benchmark
sbatch run_sweep.sbatch          # 100 fits = 20 configs x 5 seeds, ~4 h on an L40S
```

`run_sweep.py` is **extracted verbatim from the notebook's own cells** (1, 16, 18, 20), so the script and the
notebook cannot drift apart. It streams progress to `sweep.log` and **appends each fit to
`sweep_results.csv` as it lands** — so a crash at fit 99 does not discard the other 98, and re-running
resumes from whatever is already done. The notebook then loads that CSV from cache, so opening it is
instant.

Delete `sweep_results.csv` to force a clean re-run.
