# Figure outline, Figures 1–5 — status and what remains

*Working record, updated 2026-08-01. The single authority for what each figure contains.
`FIGURE_OUTLINE_2_3.md` was deleted on 2026-08-01 — it referenced `benchmark_sweep.py`,
`sweep_results.csv` and the 2026-07-18 models (all replaced) and described the pre-Pyro-plate
trainer. Recover it from git history if ever needed.*

**Directory layout (2026-08-01).** Each figure has `code/ data/ plots/`; shared modules are in
`figures/_common/`. `FIGURE2_benchmark` → `FIGURE2_simulation_imputation`, `FIGURE3_programs` →
`FIGURE3_leafcutter_ds`, `FIGURE4_parsebio` → `FIGURE4_aging`. Panels are built by
`FIGURE2_simulation_imputation/code/build_sim.py` and `build_imputation.py`, both thin entry points
over the shared `panels.py`.

**"Atlas" means the full dataset (Figure 1) and the released full-data models (`*_full_K30_*`).**
It does not mean the 70/30 held-out imputation work, which is named `impute`/`heldout` throughout.
The held-out models are `2026-07-30_atlas9_default` — the directory keeps its historical name, but
nothing in the figure code calls it atlas.

Status: **✅ built** · **📊 data ready, figure to build** · **⏳ running** · **🔧 to build** ·
**❓ inputs not inspected**

| | figure | status |
|---|---|---|
| **1** | The mouse splicing atlas | 🔧 |
| **2** | **Simulation + imputation** — can it recover known structure, and impute held-out data? | ✅ / 📊 |
| **3** | **Full-data model: calibration vs LeafCutter, differential splicing** | 📊 / ❓ |
| **4** | **Aging case study in mouse** | ❓ |
| **5** | **Human vs mouse brain splicing** | ❓ |

---

## FIGURE 1 — The mouse splicing atlas  🔧

Descriptive: what the data is, before modelling. `figures/FIGURE1_atlas/` is empty.

| panel | content |
|---|---|
| A | study schematic — tissues, protocol, junction → ATSE construction |
| B | atlas composition — cells per tissue / cell type |
| C | ATSE structure — junctions per event, cassette fraction |
| D | coverage — reads per cell, junction detection rate |

**Numbers already established** (reuse, don't recompute): 132,706 cells × 79,768 junctions; 70/30
split = 92,894 / 39,812; junction coverage 13.80%; observed splicing rate 0.2947; 39 annotated cell
types (27 with ≥50 cells on both sides).

---

## FIGURE 2 — Simulation + imputation  ✅ / 📊

**Claim: on known ground truth it recovers the structure, and on real held-out data it imputes better
than the alternatives.** Two halves, one figure.

### 2a — Simulation ✅ BUILT
`FIG2_validation.pdf`, `FIG2_simulation.pdf`. Methods: `METHODS_simulation_2026-07-26.md`.
Reproduction: `REPRODUCE_ALL.md` §A.

| panel | content | status |
|---|---|---|
| A | cassette-exon schematic | 🔧 **hand-drawn, outstanding** |
| B | effect-size recovery vs simulated ΔΨ, vs NMF | ✅ |
| C | cell separation (silhouette) vs ΔΨ | ✅ |
| D | Ψ recovery vs ΔΨ, with the "read this with B and C" caveat | ✅ |

Effect-size recovery 0.41 → 0.96 with ΔΨ; **both NMF variants ~0 at every effect size**. Silhouette
0.55 → 0.87 (verified: 0.545 → 0.868).

**Be precise about the NMF silhouette comparison.** "NMF −0.05 → 0.49" is *mean-impute alone*.
Averaged over both NMF variants it is **0.168 → 0.452**, and **0-impute never drops below 0.389**.
Quoting −0.05 without naming the variant states the worst NMF arm as if it were NMF, which overstates
the gap at low effect size — the honest low-effect comparison is 0.545 (LeafletFA) vs 0.389 (NMF
0-impute), not vs −0.05. Effect-size recovery needs no such caveat: both NMF variants sit in
[−0.036, 0.098], mean 0.017, at every effect level.

### 2b — Imputation on held-out cells ⏳ fixed-entry-set rerun in flight (`19414852`)

| panel | content | status |
|---|---|---|
| E | imputation accuracy vs coverage on **`fixed_lost_at_hardest`** — LeafletFA vs NMF-transfer (0-impute / mean-impute) vs naive. **Pure-prediction methods only** | ⏳ |
| F | imputation at matched coverage, by K (10/20/30) | ⏳ |
| E-supp | the same curve on `all_covered` with every method, plus the old `lost_to_downsampling` curve and its moving-population caveat — the contrast is what justifies the fixed set | 🔧 |

**Why a rerun.** The original entry set changes composition at every coverage level, so its curve is
not a curve; the fixed set holds the same 93.4M entries throughout. Full derivation, verified counts
and the plotting caveat are in `REPRODUCE_ALL.md` §B3b — read that before building these panels.

The one thing not to get wrong: at mild downsampling **93% of the fixed set is still observed**, and
`lf_bb`/`mb`/`mb_ct` pass observed values through. Plot them here and the junction-mean baseline
appears to beat LeafletFA for reasons that have nothing to do with imputation. All methods are in the
CSV; panel E subsets to the ones that predict at every level.

~7 h per batch of 3 tasks, 9 tasks at `%3` → ~21 h.

### Design decisions, each with its own evidence
- **Beta-Binomial over Binomial** — indistinguishable on non-overdispersed data *by construction*;
  significantly better at c = 2 on silhouette, AUROC and effect recovery (*P* < 0.001), and survives
  re-testing under a changed initialisation.
- **Junction-specific Ψ prior** — *not* significant here (*P* ≥ 0.45 on most metrics). Reported as
  such. Kept because it is free and helps under overdispersion.
- **Minibatch vs full-batch** — differ by ≤0.004; significant only because the trainer is extremely
  reproducible (sd 1e-5), so no significance stars are drawn.

### Supplementary to Fig 2 (built)
ATSE illustration · %-differential sweep per metric · prior ablation at atlas-like overdispersion ·
minibatch check · held-out latent structure (`FIG2_supp_heldout_structure`) · program × cell-type map
train vs held-out (`FIG2_supp_program_celltype_K{10,20,30}`) · transfer controls
(`FIG2_supp_transfer_controls_K*`) · seed consistency in simulation ⏳.

**Held-out validation numbers in hand:** cell-type program usage reproduces on unseen cells to **4.4%
relative error**; **100% of 27 cell types keep the same top program** at K = 10 (92.6–96.3% at K = 30);
transfer costs **0.0011** against the sampling floor, while a different seed's map scores 0.749 and
label-shuffled 0.244; held-out φ is not prior-dominated (perplexity 2.37 vs 2.37 training, prior 9.75).

---

## FIGURE 3 — Full-data model: calibration vs LeafCutter, differential splicing  📊 / ❓

**Claim: the released full-atlas model finds differential splicing that agrees with the standard tool,
and its posterior probabilities are calibrated.**

### Inputs

- **Model: `2026-07-26_atlas_FULL`.** 9 fits, K ∈ {10, 20, 30} × 3 seeds, all 132,706 cells,
  **all 9 finite**, `psis_loc`/`psis_scale` present so the DS sampler works. Trained with the current
  code: plate-based minibatch B = 8192 (after `c8cad3d`, the fix for subsampling never engaging),
  Beta-Binomial, junction-specific Ψ prior, random init, and **default** early stopping
  (`patience=8, min_delta_rel=1e-4`) — byte-identical settings to the clean 70/30 set. The only
  package change since is `68a3185`, which is default-off and does not touch the training path.

- **⚠️ NOT `2026-07-27_full_K30_tight` — 9 of its 10 seeds are CORRUPT** (`psis_scale` 100 % NaN).
  It was trained with `patience=25, min_delta_rel=1e-6`, the same tight early stopping that drove 4
  of 9 fits non-finite on the 70/30 attempt. Only seed 9 survives, so medoid selection over 10 seeds
  is not available from it. `2026-07-28_full_K30_{jitter,detinit}` are also unusable (corrupt, and
  no fits at all respectively) — both are abandoned-experiment arms.

- **The under-training worry that motivated the tight retrain is not supported.** Doubling the
  epochs (77 → 136) moved the Ψ contrast from **0.0808 to 0.0814**, i.e. +0.7 %, with `bb_conc`
  unchanged at ~1.95. Step-counted early stopping does give the full atlas fewer epochs than the 70/30
  set at the same step count, but the quantity the paper reports barely moves.

- **Open gap:** `atlas_FULL` has only **3 seeds per K**. If Figure 3 wants medoid selection or a
  consensus over ~10 seeds at K = 30, that needs a retrain at K = 30 with the **default** early
  stopping — cheap and low-risk, and it must be health-gated (`A1` now exits 2 on any non-finite
  array; `atlas_FULL` predates that gate but passes it when checked manually).

- **LeafCutter:** exists from 2026-06-20 — `pipelines/08_leafcutter_ds/` (steps 00–04) and
  `$WD/LEAFCUTTER_DS/` (`by_celltype`, `by_tissue`, `ds_summary_all.tsv`, `leafcutter_counts.tsv.gz`).
  ❓ not inspected against the current models.
- **LeafCutter:** exists from 2026-06-20 — `pipelines/08_leafcutter_ds/` (steps 00–04) and
  `$WD/LEAFCUTTER_DS/` (`by_celltype`, `by_tissue`, `ds_summary_all.tsv`, `leafcutter_counts.tsv.gz`).
  ❓ not inspected against the current models.

| panel | content | status |
|---|---|---|
| A | LeafletFA vs LeafCutter — concordance of differential junctions per cell type | ❓ |
| A′ | **seed reproducibility of the cell-type DS calls** — Jaccard / effect-size slope / top-N overlap across 3 seeds | 🔧 **run on the fresh models (2026-07-30)** |
| B | calibration — nominal posterior probability vs realised FDR | 🔧 |
| C | effect-size agreement — LeafletFA β vs LeafCutter ΔΨ, with the **slope**, not just correlation | 🔧 |
| D | what LeafletFA finds that LeafCutter cannot (or vice versa) | 🔧 |

### Reproducibility of the DS calls — measure the RIGHT analysis

Measured on the old K=30 models, the **factor-level** contrast (program *k* vs the mean of the other
programs) reproduces poorly across seeds: **Jaccard 0.51**, effect-size **slope 0.72**, top-2000
overlap **42%**, on ~235,000 calls per fit.

That is the analysis being demoted to "program characterisation", so it is a caveat for that section
rather than a result for this figure. **The primary analysis is group-vs-group by cell type, and it has
not been measured yet.** It should reproduce considerably better for a structural reason: cell-type
labels are fixed, so the groups are identical across seeds and nothing depends on program matching —
only the per-cell-type splicing estimate varies, and that reproduces at *r* = 0.79 (map) to 0.90
(structure).

Run the same Jaccard / slope / top-N framework (`A23_ds_reproducibility.py`, adapted to
`compute_differential_splicing_groups`) on the freshly retrained models. Do not design this figure
around the 0.51 figure — it is the wrong analysis.

### Cautions carried in from earlier work — these matter for this figure
- **Report the slope, not the correlation**, when comparing effect sizes. Hungarian-matched Pearson *r*,
  cosine and argmax agreement are all **scale-invariant** and would score ~1.0 even under systematic
  attenuation. This exact blind spot made an earlier batch-equivalence check incapable of detecting
  effect-size shrinkage.
- **Rest the comparison on cell-type-level structure**, which reproduces at *r* = 0.90 — not on
  per-cell φ (0.65–0.87). Now an evidence-backed constraint, not caution.
- **The DS machinery needs reconciling before this figure is trustworthy**: the two FDR functions
  disagree with each other and neither implements the max-detections rule; `|β| > min_effect_size` is
  applied *after* the FDP is computed, so the reported FDR is not the FDR of the reported set; the
  Methods' φ̄-weighted effect-size equation does not match the unweighted quantity the code computes;
  `num_samples` is 100 in the sweep but 500 in the Methods.

---

## FIGURE 4 — Aging case study in mouse  ❓

Pipeline `pipelines/06_easysci/` exists (liftover → prep → apply model). Outputs not inspected.

**Shape of the claim:** apply the frozen atlas model to the aging dataset, and show splicing programs
whose usage shifts with age — a biological result the method enables, not another benchmark.

**To check before drafting:** whether `02_apply_model_to_easysci.py` has been run against the current
models or only older ones; and what age metadata is available.

**Method note:** applying a trained model to new cells is the frozen-globals E-step — Ψ, π, α, δ and
`bb_conc` all fixed, only φ inferred. That machinery is built and validated (it is what the held-out
transfer analysis uses). Freeze `bb_conc` to the trained value; do not switch to Binomial.

---

## FIGURE 5 — Human vs mouse brain splicing  ❓

Pipeline `pipelines/07_conservation/` exists (GSE138734 download → metadata prep). Outputs not
inspected.

**Design trap to avoid — established earlier and worth keeping in front of you.** Projecting human
data onto the mouse dictionary cannot demonstrate shared structure; it *assumes* it, and
reconstruction error on the target has no scale on its own. Two baselines make it a measurement:

- **A — a human-native model at the same K.** If the mouse Ψ reconstructs held-out human data nearly
  as well as a human-native Ψ does, that gap **is** the conservation result, with an effect size.
- **B — a scrambled dictionary** (permuted Ψ, or Ψ from a mismatched tissue). The null. If the real
  dictionary doesn't beat it, the "transfer" was K free parameters per cell fitting noise.

**Alternative framing, probably better:** fit each species separately and match programs *post hoc*
(Hungarian on Ψ correlation over orthologous junctions), reporting match quality. That avoids the
circularity entirely. Transfer stays the right tool for held-out cells and new batches, not for the
conservation claim itself.

---

## Cross-cutting: initialisation and reproducibility  ⏳ OPEN

Gates the released model, so it gates Figures 3–5.

**Established:**
- Ψ reproduces across seeds at *r* = 0.97 and π at 0.93; the program × cell-type map at 0.90. **These
  are the objects the paper claims.** Per-cell φ agrees at 0.65–0.87, and only ~30–50% of φ directions
  reproduce.
- **Cause: overdispersion.** Same simulated design at c = ∞/20/2 gives 100%/83%/42% reproducible
  directions, with accuracy against known φ degrading in step. Learned c ≈ 1.96 is **flat across
  K = 10/20/30**, so it is not structure more programs would absorb. At c = 2 the likelihood penalises
  a wrong φ **20× less** than with no overdispersion. Mechanically: reads inform `q` (this cell's
  realised rate), not `p` (the model's prediction), and the two are only loosely tied — so depth
  saturates at ~c+1 ≈ 3 effective reads per junction. Ψ escapes this by pooling ~10⁴ cells; φᵢ has one.
- **Refuted as causes:** longer training (tested twice), the junction-specific prior, program
  collinearity (centred programs already at the −1/(K−1) floor).
- Pyro's default init is **not random** — `init_to_feasible` sets ψ = 0.5 and φ = 1/K exactly, so all K
  programs start identical and only ~0.02 of sampling noise can separate them.
- **Jitter is out.** Fixed information-free offsets helped in simulation at c = 2 (30% → 52%) but made
  the **real atlas worse** (effective K 36% → 29%). Arbitrary symmetry-breaking is not the answer.
- Decomposition: location **−0.010**, symmetry **+0.092**, data content **+0.066**.

**CLOSED (2026-07-30).** Jobs `19358688` (A20 simulation, three init arms × three overdispersion
levels × 3 seeds) and `19358721` (the same on the 70% atlas, K = 30) both landed. **No initialisation
tried beat the default.** Waypoints and jitter were each verified to work mechanically at scale
(waypoint contrast 0.1494 against 0.000 for the default) yet neither improved cross-seed
reproducibility on the atlas, which is the result that decides — simulation had already mispredicted
the atlas once. The variant branches were removed from `A1`, and the 70/30 models were retrained from
the default path as `2026-07-30_atlas9_default` (9/9 finite, health-gated).

Consequence for the paper: seed variation is **reported, not engineered away**. Ψ, π and the
cell-type × junction rate are what reproduce; per-cell φ does not, for the overdispersion reason
established above. The `A19_uncertainty_calibration.py` question — does per-cell posterior width
predict which cells the seeds disagree on — remains open and is the honest way to present it; `A1`
saves `phi_sd` and `phi_top_prob` for exactly this.

---

## Immediate to-do, in order

1. **Build Fig 2E/2F (imputation)** from `04e_allcov_fixedset.csv` once the fixed-entry-set array
   lands (`19414852`, ~21 h). Plot **pure-prediction methods only** on `fixed_lost_at_hardest` —
   at mild downsampling 93% of that set is still observed and the passthrough methods would win
   spuriously. See `REPRODUCE_ALL.md` §B3b.
2. **Regenerate the A7/A9 supplements** — queued as `19414889` → `19414890` behind the imputation
   array. Every program × cell-type and transfer-control panel previously on disk came from the
   *superseded* 2026-07-26 models; those files are deleted rather than left to mislead.
3. **Fig 2A** hand-drawn cassette schematic — the only missing piece of an otherwise complete 2a.
4. **Cell-type group-vs-group DS reproducibility** on the fresh models (`A23` adapted to
   `compute_differential_splicing_groups`) — prerequisite for Fig 3A′.
5. **Inspect the Fig 3 LeafCutter outputs** and the Fig 4 / Fig 5 pipelines; replace ❓ with real status.
6. **Reconcile the DS/FDR machinery** before Fig 3 numbers are quoted.

## Known loose ends
- `phi_uncertainty` logged `median sd nan` on the atlas though the saved array has 0% NaN — the printed
  statistic is `mean` mislabelled as median, and something in it is non-finite. Cosmetic; check before
  quoting.
- ARI moves the wrong way in the overdispersion arm under a changed init (0.505 → 0.182 at c = 20)
  while silhouette moves up on the same fits. Unexplained. ARI backs no headline claim, but unresolved.
