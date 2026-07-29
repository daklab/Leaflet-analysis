# Reproducing Figure 2 — every step, in order

Everything below was regenerated on **2026-07-26** from the released splicing AnnData. All parameters
were read out of the code or out of the `uns['sim_params']` block written into each simulated file.
Earlier outputs are kept under `archive_pre_2026-07-26/` (this directory) and
`SIMULATION/archive_pre_2026-07-26/` — nothing was deleted.

Environment: `/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python`.
Model code: `/gpfs/commons/home/kisaev/LeafletFA` (`leafletfa.minibatch`).

---

## Step 0 — input

```
MODEL_INPUT/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad
```

The real mouse splicing AnnData released with the paper. Only the **heart** tissue is used, and only
its `cell_by_cluster_matrix` — the simulation borrows *real coverage*, never real splicing values.

---

## Step 1 — simulate cassette exons with known truth

`pipelines/09_simulation/00_simulate_splicing.py`

**1a. Effect-size sweep (primary — the main figure).** 4 effect levels × 3 seeds = **12 datasets**.

```bash
for eff in 0.1 0.25 0.5 1.0; do
  for seed in 0 1 2; do
    python 00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3 \
           --effect $eff --bb-conc inf --seed $seed
  done
done
```

**1b. Differential-fraction sweep (secondary — supplement).** 3 fractions × 3 seeds = **9 datasets**.

```bash
for frac in 0.1 0.5 0.9; do
  for seed in 0 1 2; do
    python 00_simulate_splicing.py --tissue Heart --K 5 --frac-differential $frac \
           --effect 1.0 --bb-conc inf --seed $seed
  done
done
```

Each run writes `SIMULATION/sim_heart_K5_binom_eff<e>_frac<f>_seed<s>.h5ad`.

**Verified filtering counts** (identical on every run — the pipeline is deterministic):

| stage | count |
|---|---|
| Heart cells × junctions | 6,986 × 79,768 |
| junctions covered in ≥ 50 cells | 57,726 |
| three-junction ATSEs | 5,566 |
| **canonical cassettes** (identified from coordinates) | **4,268 (76.7%)** |
| final simulated object | **6,986 cells × 12,804 junctions** |

`--effect` controls *how different* the programs are at a differential cassette; `--frac-differential`
controls *how many* cassettes differ. Median true |ΔΨ| by effect level: 0.085, 0.21, 0.42, 0.85.

---

## Step 2 — fit every method on every simulation

`figures/FIGURE2_benchmark/sim_benchmark.py`  (supersedes `benchmark_sweep.py`)

```bash
python sim_benchmark.py <sim.h5ad> <out.csv> cuda 2048
```

Fits, all at the true *K* = 5:

| arm | how |
|---|---|
| LeafletFA × {Beta-Binomial, Binomial} × {junction-specific, global prior} | `LeafletFAMinibatch`, batch 2,048, lr 0.05, γ 0.1, 5,000-step cap with early stopping |
| NMF 0-impute / mean-impute | `NMF(n_components=5, init='nndsvda', max_iter=300, random_state=0)` |

**Why this replaced `benchmark_sweep.py`:** the old script fitted through the legacy `LeafletFA`
class (full-batch, fixed epochs), which is *not* the code path that trains the atlas models. The
simulation was therefore validating a different implementation from the paper's results.

Each fit asserts that the traced local variable is batch-shaped, so "minibatch" can never silently
mean "full batch". Every row records `effect`, `frac`, `seed`, `batch_size`, `steps_used` and
`trainer`.

Shards are concatenated to `benchmark_effect_sweep.csv` (12 sims × 6 arms = 72 rows) and
`benchmark_results.csv` (9 × 6 = 54 rows).

---

## Step 3 — panel inputs that need the full h5ad

`figures/FIGURE2_benchmark/prep_panels.py` → `panel_data.npz`

- **UMAPs** of the true φ, LeafletFA's φ and NMF's *W* (both imputations), coloured by true program.
- **Silhouettes** printed on those panels, computed on the **latent representation**, not the 2-D
  coordinates — UMAP inflates apparent separation.
- **ATSE illustration** (supplement): the cassette with the largest ΔΨ and one with ΔΨ = 0.
- LeafletFA is **fitted here with the shipped trainer**, not read from a saved file, so the panel
  always reflects current code (an earlier version depended on an external `.npz` and the panel
  silently disappeared when that file was archived).

---

## Step 4 — draw the figures

```bash
cd figures/FIGURE2_benchmark && python build_figures.py
```

Panels report the mean over seeds with s.e.m.; all pairwise comparisons are Welch's *t*-test with
`ns` shown when *P* ≥ 0.05. Styling and the sentence-case pass come from `figures/leaflet_theme.py`.

---

## Optional — supporting analyses

**Trainer ablation** (`ablate_trainer.py`): varies learning-rate decay, ELBO particles, early
stopping and batch size one at a time, plus one configuration adopting *all* the legacy
hyperparameters inside the current implementation. Every configuration lands within 0.001 (Ψ
recovery) and 0.016 (effect size), whereas the legacy implementation scored 0.80 ± 0.18 — localising
the difference to the inference implementation rather than to tuning.

**Held-out E-step check** (`LeafletFA/experiments/minibatch/test_estep.py`): confirms the frozen-global
E-step recovers known loadings (0.997), is seed-stable (mean |Δφ| = 0.004), leaks no global into the
variational family, and leaves standard training unchanged.

---

## Things a reader should know

- **`sweep_results.csv` has a `batching` column (`chunked` vs `full`). Do not use it as a minibatch
  comparison.** It predates the plate rewrite, so its "chunked" arm came from the implementation in
  which subsampling was silently inactive. Minibatch comparisons come from
  `LeafletFA/experiments/minibatch/sim_recover10_Kmodel{10,20,30}.csv`.
- **The differential-fraction sweep saturates.** At `--effect 1.0` the median simulated |ΔΨ| is 0.85,
  which every working method resolves, so all LeafletFA arms sit at ~0.996 regardless of *f*. That
  sweep is kept in the supplement; the effect-size sweep is what discriminates.
- **The junction-specific prior only matters where signal is weak.** Indistinguishable at |ΔΨ| = 0.85,
  worth up to +0.10 Ψ recovery at |ΔΨ| = 0.085.

## File map

| file | produced by | used for |
|---|---|---|
| `SIMULATION/sim_heart_K5_binom_eff<e>_frac<f>_seed<s>.h5ad` | `00_simulate_splicing.py` | everything |
| `benchmark_effect_sweep.csv` | `sim_benchmark.py` (12 shards) | main figure |
| `benchmark_results.csv` | `sim_benchmark.py` (9 shards) | supplement |
| `panel_data.npz`, `representative_fit.npz` | `prep_panels.py` | UMAP + ATSE panels |
| `rerun_logs/ablate_{0,3}.csv` | `ablate_trainer.py` | trainer attribution |
| `FIG2_validation.{pdf,png}` | `build_figures.py` | the figure |
