# Imputation / coverage-robustness analysis — Methods

*Figure number TBD (2 or 3). Filename kept as `imputation_analysis` for readability.*

---

## 1. What this analysis actually is

**It is coverage downsampling, not masked-entry imputation.** The distinction matters and has been
described wrongly before (including in older filenames — `MASKED_X_PERCENT` — which are really
*resample* fractions).

- **What the model is given:** the held-out test cells with their **read counts thinned** to a
  fraction of the original depth (`cell_by_junction_matrix`, `cell_by_cluster_matrix` in the
  RESAMPLED files).
- **What it is scored against:** the **original, full-depth** PSI (`junc_ratio_original`), at
  (cell, junction) pairs that had genuine coverage originally (`cell_by_cluster_matrix_original > 0`).

So the question is **"how much sequencing depth does LeafletFA need before its PSI estimates
degrade?"** — not "can it fill in a hidden entry". This is the more useful claim for a single-cell
splicing atlas, because real data is coverage-starved.

> **Answered, 2026-08-03 — see §5b.** Below roughly **10 % of reads retained**, the model's
> prediction is closer to the full-depth truth than the surviving measurement is: 0.090 vs 0.094 at
> 10 %, 0.082 vs 0.139 at 1 %, 0.086 vs 0.166 at 0.1 % (mean |error|, PSI units). `lf` degrades by
> ~25 % across a 900-fold reduction in depth; the raw ratio degrades 4-fold. Above 10 % the
> measurement is better and the model adds nothing — which is the honest form of the claim.

The coverage mask deliberately **includes cells with PSI = 0** (junction reads 0, ATSE reads > 0).
Those are genuine zero-PSI observations the model must get right; excluding them would hide
predictions at the boundary.

## 2. Data

Seven downsampling levels, all built from the **same** test 30% split that pairs with the 70%
training data (`..._minCells75_minJuncs500_2026-06-17`):

| level | file |
|---|---|
| 10, 25, 50, 75, 90, 99, 100 % | `MODEL_INPUT/20260526/resampled/RESAMPLED_{L}_PERCENT_test_30_70_..._2026-06-17.h5mu` |

100% is the no-downsampling control. Evaluation is restricted to the top `n_hvj` high-variance
junctions (default 10,000) so that per-junction correlations are computed where there is signal.

## 3. Applying the model at test time — a projection, not a refit

φ is inferred for the held-out cells with **every global frozen**:

```
fixed_psi        = trained psi          # the dictionary
pi_fixed         = trained pi           # the prior over phi
alpha_pi_fixed   = trained alpha_pi
delta_fixed      = trained dir_conc
input_conc_prior = as_conc_prior(bb_conc)   # the TRAINED likelihood
```

Only φ is inferred. With ψ fixed the per-cell problems are independent and K-dimensional, so this
is well-conditioned — no restarts or waypoints needed.

> **Two leakage bugs that were fixed here, and why they mattered.**
> 1. Earlier versions passed `pi_init` rather than `pi_fixed`. `pi_init` only *initialises* a π that
>    SVI then keeps optimising **on the test cells** — so φ_test was inferred under a different prior
>    than φ_train, confounding any train-vs-test comparison.
> 2. Earlier versions forced `input_conc_prior = inf` (Binomial) even though the models were trained
>    Beta-Binomial. Fitting φ under a different likelihood than the one that produced ψ means the
>    E-step is not the model. The old rationale ("the tiny learned `bb_conc` leaves the likelihood
>    flat") was a symptom of the old `bb_conc ~ Gamma(10,10)` prior (mean 1); that prior is now
>    `Gamma(2, 0.1)` (mean 20), so the trained concentration is sensible and is used directly.

## 4. Metrics

### 4.1 Correlation (ranking quality)
Per junction, Pearson and Spearman between imputed and original PSI **across cells**, restricted to
cells with original coverage and junctions with at least `min_cells` such cells. Reported as the
median across junctions, plus a global (pooled) Pearson.

### 4.2 Calibration — **the metric correlation cannot replace**

Correlation measures whether the model **ranks** cells correctly. It is completely blind to
predictions being systematically **compressed toward the mean** — which is exactly what a
shrinkage-heavy latent-variable model does. Demonstration on synthetic data:

| predictions | Pearson r | calibration slope |
|---|---|---|
| perfect | 1.000 | **1.000** |
| compressed 0.4× toward the mean | **1.000** | **2.500** |
| pure noise | ~0 | 0.007 |

**A model whose predictions are 2.5× under-dispersed still has a perfect correlation of 1.0.** This
is Reviewer 1's objection ("no calibration reported, only correlation"), and it is why the following
are now computed for **every** method (`lf`, `lf_bb`, `mb`, `mb_ct`, `naive`):

| metric | meaning |
|---|---|
| `calib_slope_*` | OLS slope of **truth ~ pred**. 1.0 = calibrated; **>1 = predictions compressed** (under-dispersed); <1 = over-dispersed |
| `calib_intercept_*` | systematic offset, in PSI units |
| `calib_rmse_*` | RMSE in **PSI units** — interpretable, unlike r |
| `calib_mae_*` | MAE in PSI units |
| `calib_bias_*` | `mean(pred − truth)`; systematic over/under-prediction |
| `calib_bin_pred/obs/n_*` | the **calibration curve**: observed mean PSI within each decile of predicted PSI. Perfect calibration lies on the diagonal |

Report the **slope and the calibration curve alongside r**. A high r with slope ≫ 1 means the model
ranks well but shrinks its predictions, and the PSI values should not be read as absolute.

## 5. Baselines

| baseline | what it controls for |
|---|---|
| `naive` | global per-junction mean PSI — the floor |
| `mb` | beta-binomial weighted per-junction mean (coverage-aware shrinkage) |
| `mb_ct` | the same, stratified by `broad_cell_type` — controls for "is this just cell-type average splicing?" |
| `lf`, `lf_bb` | LeafletFA point estimate (`φ·ψ`) and the Beta-Binomial posterior using observed counts |

> **⚠ 2026-08-03: `mb` and `mb_ct` are not usable as written.** Both come from `mean_bayes.py`, whose
> final step is
> ```python
> return np.where(observed_mask, psi, post_mean_per_junction[np.newaxis, :])
> ```
> — the **observed PSI is passed through unchanged**, and only unobserved entries get the posterior
> mean. At 90 % retained that is ~93 % of entries, so `mb` is mostly the identity function and its
> apparent 0.888 is the measurement scoring itself, not a baseline being beaten. It also explains why
> `mb` and `mb_ct` agree to 0.001: they differ only on the ~7 % that are unobserved.
>
> Separately, `MeanBayes(mb_ct_adata, group_by="broad_cell_type")` is fitted on the **test** cells'
> own data, so the cell-type means are in-sample. On a 2,500-cell subsample (median **4** cells per
> junction × cell type) each entry contributes ~25 % of its own group mean: in-sample AUROC 0.692,
> leave-one-out 0.412 — the first leaks up, the second is biased down at n≈4, and neither is fair.
>
> **The `mb_ct` control the section below calls decisive is therefore unresolved**, not passed. A fair
> version fits cell-type means on the TRAINING cells and applies them to test, mirroring how ψ is
> obtained. Until that exists, do not quote "LeafletFA beats a cell-type-stratified mean".

`mb_ct` is the important one: if LeafletFA does not beat a cell-type-stratified mean, the programs
are adding nothing beyond cell-type identity.

---

## 5b. Findings, 2026-08-02/03 — read this before quoting any number

Working record of a long diagnostic pass (`A40`–`A44` in `pipelines/05_leafletfa/`). Several earlier
readings turned out to be artefacts of the metric or the entry set rather than statements about the
model.

### The result: a crossover at ~10 % of reads

Mean |predicted − true| in PSI units, on entries that still have reads, so the raw measurement is a
comparator (`A44`, 2,000 cells, all seven levels, K = 20):

| reads retained | raw ratio | `lf` = φ·ψ | `lf_bb` |
|---|---|---|---|
| 90 % | **0.038** | 0.103 | 0.051 |
| 50 % | **0.052** | 0.100 | 0.059 |
| 25 % | 0.071 | 0.095 | **0.065** |
| **10 %** | 0.094 | **0.090** | **0.072** |
| 1 % | 0.139 | **0.082** | **0.081** |
| 0.1 % | 0.166 | **0.086** | 0.089 |

**Below roughly 10 % of reads the model's prediction is closer to the truth than the measurement
itself** — 41 % better at 1 %, 48 % better at 0.1 %. `lf` barely moves across the whole range
(0.103 → 0.086) while the raw ratio collapses (0.038 → 0.166). That is the answer to the question in
§1, and it is the claim to make.

`lf` accuracy is governed by **ATSE depth, not by downsampling**: 0.065 at ≥100 original reads versus
0.188 at 1 read, while a 900-fold reduction in coverage costs only ~25 %.

### The metric was wrong for the data

The target is **71 % exactly 0 or 1** (61 % exactly zero), and still 58 % binary at ≥30 reads.
Per-junction Pearson between a smooth prediction and a near-Bernoulli outcome is crippled by
construction, which is the whole reason `median_pearson_lf` reads ~0.05:

- stratified by zero-inflation: **0.18** on balanced junctions, **0.054** on >95 %-zero ones
- AUROC on the same entries — the appropriate metric for a binary target — gives `lf` **0.60**
  against a floor of exactly 0.50 (a constant per-junction mean cannot rank cells)
- Spearman does **not** help (0.0502 vs Pearson 0.0501): with 61 % ties at zero it is equally
  degraded. It does expose passthrough, though — `lf_bb` on `all_covered` reads Pearson 0.858 but
  Spearman 0.507.

### Ceilings, and which denominator is honest

`A40` measures two, agreeing to 0.01 by unrelated routes (split-half resampling; a variance
decomposition at the learned c = 1.9654): **≈ 0.74–0.82**. But both are reachable only by a predictor
that sees the entry's own reads, so they bound *any* method without being a target for a blind one.
They establish that the target is not noise-saturated — nothing more. Do not quote "0.05 is 7 % of
the ceiling".

### Hypotheses tested and rejected

Recorded so they are not re-proposed:

- *HVJ selection picks noisy junctions* — backwards. Variance selection **helps**: 0.19 on
  variance-selected junctions vs 0.09 on random ones (`A42`).
- *The ATSE expansion dilutes the set* — no effect. Variance-selected 0.082, expansion partners
  0.087.
- *The `min_cells_per_junc` threshold* — no effect. 0.085 at ≥20 cells, 0.084 at ≥400.
- *Overdispersion is the binding ceiling* — it is not; 0.808 with it against 0.818 for sampling alone.

The actual driver is zero-inflation.

### Entry sets

- `all_covered` mixes observed and unobserved entries, so its curve slides from ~0.86 to ~0.06 purely
  as the passthrough fraction falls.
- `lost_to_downsampling` is a **moving population** — at 90 % retained it is the shallowest entries,
  whose true PSI is 0 or 1, which is why its calibration slope reads 0.49 there and 0.98 lower down.
- `fixed_lost_at_hardest` (added 2026-08-01) holds the same 93,394,617 entries at every level and is
  the only one readable as a curve. On it, calibration is slope 0.966–0.983, bias ±0.001, RMSE
  0.2045–0.2148 — stable across a 450-fold depth range, which is the direct answer to Reviewer 1.
- On any entry set where nothing is observed, `lf_bb` reduces **exactly** to `lf`
  (`(a+0)/(a+b+0) = lf`), verified identical to machine precision at all seven levels.

### Still open

- The fair (train-fitted) cell-type oracle — the §5 control, currently unresolved.
- The per-cell direction has a floor problem: `lf` 0.85 against a 0.83 junction-mean floor
  (`A42`), a +0.02 margin. Between-junction structure is free and must not be quoted as performance.
- `A44` used 2,000 cells and K = 20 only; the fixed-set array was cancelled at 3/9 tasks.

## 6. Implementation notes

- **Memory.** `junc_ratio_original` is stored dense. Reading the whole layer materialised
  N × J_all float32 (≈13 GB at 40k × 80k) only to discard all but `n_hvj` columns. It is now
  fancy-indexed on the column axis at read time, so the full matrix is never held.
- ψ is **sampled** from `sigmoid(psis_loc + psis_scale·ε)` rather than using the point estimate, so
  predictions carry the posterior uncertainty in ψ.
- `ELBO_num_particles=10`, `num_samples=100`, `batch_size=4096` on GPU.

## 7. Reproducing

```bash
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
python pipelines/05_leafletfa/05e_submit_gpu.py \
    $MODEL_DIR \
    $WD/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad \
    $WD/MODEL_INPUT/20260526/resampled \
    $EVAL_OUT
python pipelines/05_leafletfa/08_collect_evaluation.py $EVAL_OUT
```

> **Note on prior results.** The 96-run held-out findings recorded in `WORKFLOW.md` Step 12b
> (K=50 best; Beta-Binomial collapses at lr=0.01) were produced by models trained **before** the
> training fixes (early-stopping hard cap at `patience` epochs, chunk-2 device mismatch, the
> `Gamma(10,10)` `bb_conc` prior, and the CUDA→numpy save crash). They should be treated as void and
> re-derived on the current models rather than carried forward.
