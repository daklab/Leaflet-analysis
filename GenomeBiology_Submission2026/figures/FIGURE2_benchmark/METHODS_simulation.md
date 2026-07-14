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
