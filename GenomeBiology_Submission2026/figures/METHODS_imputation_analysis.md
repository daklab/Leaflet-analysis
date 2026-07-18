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

`mb_ct` is the important one: if LeafletFA does not beat a cell-type-stratified mean, the programs
are adding nothing beyond cell-type identity.

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
