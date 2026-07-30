# Reproducing the validation analyses, in order

Two independent pipelines feed Figure 2:

- **A. Simulation** — controlled recovery against a known ground truth.
- **B. Atlas held-out evaluation** — imputation and latent structure on real 30% held-out cells.

Both were regenerated on **2026-07-26**. Superseded outputs are under `archive_pre_2026-07-26/`
(nothing deleted).

Everything runs with `/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python`; model code is
in `/gpfs/commons/home/kisaev/LeafletFA` (`leafletfa.minibatch`). All inputs, scripts and outputs sit
on GPFS — SLURM compute nodes cannot see node-local `/tmp` or `/scratch`, so a job that writes there
will appear to succeed and leave nothing behind.

```
WD    = /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD
REPO  = /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
PIPE  = $REPO/pipelines/05_leafletfa
SIMPY = $REPO/pipelines/09_simulation
FIG2  = $REPO/figures/FIGURE2_benchmark
```

---

# A. SIMULATION PIPELINE

### A0 · Input
```
$WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/
    splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad
```
The released mouse splicing AnnData. **Heart** tissue only, and only its `cell_by_cluster_matrix` is
used: real coverage, depth and zero-inflation are inherited from the data and only the splicing
outcome is simulated.

### A1 · Simulate cassette exons with known truth
`$SIMPY/00_simulate_splicing.py` → `$WD/SIMULATION/sim_heart_K5_binom_eff<e>_frac<f>_seed<s>.h5ad`

Two quantities govern how much signal a simulation contains, and they behave very differently.
`--effect` sets *how different* the programs are at a differential cassette; `--frac-differential`
sets *how many* cassettes differ. Recovery responds to the first and is largely insensitive to the
second, so the effect sweep is the primary axis and the fraction sweep is supplementary.

```bash
# A1a — effect-size sweep (PRIMARY, main figure): 4 levels x 3 seeds = 12 datasets
for eff in 0.1 0.25 0.5 1.0; do for seed in 0 1 2; do
  python $SIMPY/00_simulate_splicing.py --tissue Heart --K 5 \
      --frac-differential 0.3 --effect $eff --bb-conc inf --seed $seed
done; done

# A1b — differential-fraction sweep (SUPPLEMENT): 3 levels x 3 seeds = 9 datasets
for frac in 0.1 0.5 0.9; do for seed in 0 1 2; do
  python $SIMPY/00_simulate_splicing.py --tissue Heart --K 5 \
      --frac-differential $frac --effect 1.0 --bb-conc inf --seed $seed
done; done

# A1c — overdispersion arm (likelihood ablation): 2 levels x 3 seeds = 6 datasets
for conc in 2 20; do for seed in 0 1 2; do
  python $SIMPY/00_simulate_splicing.py --tissue Heart --K 5 \
      --frac-differential 0.3 --effect 0.25 --bb-conc $conc --seed $seed
done; done
```

A1a and A1b both pass `--bb-conc inf`, i.e. the counts are drawn from a Binomial. A Binomial
likelihood is therefore correctly specified on those datasets and a Beta-Binomial can only match it,
never beat it — so they cannot be used to justify the choice of likelihood. A1c regenerates the same
simulation with **overdispersed** counts at a fixed effect size: `--bb-conc 2` matches the
concentration of ≈1.96 that the atlas models learn (roughly a fivefold variance inflation at the
observed read depths), and `--bb-conc 20` is mild. Both likelihoods are fitted to each dataset, which
is what makes the comparison meaningful.

**The effect size ΔΨ** of a cassette is the *range* of skipping usage across the K programs,
`max_k Psi_k - min_k Psi_k` — the gap between the program that skips that exon most and the one that
skips it least, not an average over program pairs. A cassette with per-program usage
(0.73, 0.65, 0.84, 0.86, 0.85) has ΔΨ = 0.21: skipped 86% of the time under one program, 65% under
another. Non-differential cassettes have ΔΨ = 0 exactly. Being a range, it is set by the two extreme
programs, so four programs at 0.65 with one at 0.86 scores the same as an evenly spread cassette; the
estimator uses the identical definition on fitted profiles (`psi_hat.max(0) - psi_hat.min(0)`), so
truth and estimate are measured on the same footing, but it is a range rather than a typical pairwise
gap.

Per simulation the reported value is the **median ΔΨ over differential cassettes only**. The four
levels correspond to **0.084 / 0.210 / 0.421 / 0.841**, i.e. typical shifts of about 8, 21, 42 and 84
percentage points. Real differential splicing is usually 5–30 points, i.e. the lower two levels; at
`--effect 1.0` a cassette swings from almost fully included to almost fully skipped, which every
working method resolves — which is why A1b is held at that setting and reported only as a supplement.

Filtering is deterministic: 6,986 cells × 79,768 junctions → 57,726 junctions covered in ≥50 cells →
5,566 three-junction ATSEs → **4,268 canonical cassettes (76.7%)** → **6,986 × 12,804**.

### A2 · Fit all methods on every simulation
```bash
python $FIG2/sim_benchmark.py <sim.h5ad> <out.csv> cuda 2048
```
Six arms per dataset at the true K=5: LeafletFA × {Beta-Binomial, Binomial} × {junction-specific,
global prior}, plus NMF {0-impute, mean-impute}. LeafletFA is fitted with `LeafletFAMinibatch`
(batch 2,048, lr 0.05, γ 0.1, 5,000-step cap with early stopping) — **the same trainer that produces
the atlas models**, so the controlled experiment and the real-data results exercise one
implementation. Each fit asserts that the traced local variable is batch-shaped, so subsampling
cannot silently be inactive.

Concatenate shards → `$FIG2/benchmark_effect_sweep.csv` (72 rows),
`$FIG2/benchmark_results.csv` (54 rows) and, for the overdispersion arm,
`$FIG2/rerun_logs/odbench_*.csv` (36 rows). Every row records `effect` and `sim_bb_conc`, so the
sweep level and the truth's overdispersion are always recoverable. Every row records `effect`, `frac`, `seed`, `batch_size`,
`steps_used` and `trainer`.

Metrics, and how to read them: **effect-size recovery** (Spearman of estimated vs true ΔΨ over
differential cassettes) and **silhouette** of the fitted loadings are the primary readouts. **Ψ
recovery** is reported alongside but should not be read alone at small effect sizes — when programs
barely differ, a model that simply predicts each junction's mean correlates ≈0.9 with every true
program profile while having learned no program structure at all (NMF mean-impute does exactly this
at |ΔΨ| = 0.085: Ψ recovery ≈ 0.92 with negative silhouette and ≈0 effect-size recovery).

### A3 · Panel inputs that need the full h5ad
```bash
python $FIG2/prep_panels.py          # -> panel_data.npz, representative_fit.npz
```
UMAPs of true φ / LeafletFA φ / NMF W coloured by true program, the ATSE illustration, and
silhouettes computed on the **latent representation** rather than the 2-D coordinates (UMAP inflates
apparent separation). LeafletFA is fitted inside this script with the shipped trainer, so the panel
always reflects current code.

### A4 · Supporting analyses (optional)
```bash
python $FIG2/ablate_trainer.py <sim.h5ad> <out.csv> cuda            # trainer attribution
python /gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/test_estep.py
```
For any minibatch-versus-full comparison use these runs, or
`LeafletFA/experiments/minibatch/sim_recover10_Kmodel{10,20,30}.csv` — all produced by the current
plate implementation. (`sweep_results.csv` also carries a `batching` column, but it predates the
rewrite and its "chunked" arm was generated when subsampling was inactive, so it does not describe
the current trainer.)

---

# B. ATLAS HELD-OUT PIPELINE (imputation + structure)

### B0 · Inputs
```
train : $WD/.../MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad
test  : $WD/.../MODEL_INPUT/20260526/resampled/RESAMPLED_{10,25,50,75,90,99,100}_PERCENT_test_30_70_...h5mu
```
The 70/30 split is by **cell**. Each resampled file carries both the downsampled counts and the
`*_original` full-coverage layers, which are the evaluation truth.

### B1 · Train the atlas models
```
MODEL_DIR = $WD/.../LEAFLETFA_MODELS/2026-07-26_atlas9
```
`parameter_combinations.json` defines **9 models**: K ∈ {10, 20, 30} × seeds {0, 1, 2}, Beta-Binomial
likelihood, junction-specific Ψ prior, **random initialisation with `waypoints_use=false`** — no
cell-type label enters model fitting anywhere in the pipeline.

```bash
sbatch $MODEL_DIR/atlas9_array.sh      # array 0-8
# each task runs:
python $PIPE/A1_train_atlas.py <task_id> $MODEL_DIR <train.h5ad> \
       5000 8192 0.05 0.1 0
#      ^cap  ^batch ^lr  ^gamma ^learn_ab
```
5,000 is a cap, not a budget: early stopping on a smoothed ELBO ends each run (observed 750–1,350
steps) and restores the best state. Writes `mb_run_<t>/fit.npz` + `result.json`.

### B2 · Adapt each fit for the evaluators
```bash
python $PIPE/A2_adapt_for_eval.py $MODEL_DIR/mb_run_<t> <out>/model
```
Writes `leafletfa_model.pkl.gz` + `run_summary.csv`.

### B3 · Imputation across coverage (04e)
```bash
sbatch --array=0-8 $MODEL_DIR/eval_04e_allcov.sh     # links all 7 resampled files, then:
python $PIPE/A3_eval_imputation.py <out>/model <train.h5ad> <out>/rs_all \
       <out>/04e_allcov.csv 4096 500 20 3000 0 20 pi
#      chunk=4096  epochs=500  patience=20  n_hvj=3000  min_atse=0  min_cells=20  phi_init=pi
```
For each of the seven levels: infer φ from the thinned counts with **every global frozen** (Ψ, π, α,
δ and the Beta-Binomial concentration), reconstruct φΨ, and score it against the `*_original`
full-depth truth. Freezing Ψ alone would not be enough — π and δ together are the Dirichlet prior on
φ, so re-estimating them on held-out cells would infer those loadings under a different prior from
the one training used.

Baselines: junction mean; cell-type mean (uses labels, so an upper bound rather than a fair
comparator); NMF in-sample; and NMF-transfer, whose dictionary is fitted on the training cells and
cached once per model — it does not depend on how the test set was thinned, so it is not refitted per
level.

### B4 · Held-out latent structure (07c)
```bash
sbatch --array=0-8 $MODEL_DIR/eval_07c_new.sh
python $PIPE/A4_eval_structure.py <out>/model <train.h5ad> <test.h5mu> \
       <out>/07c_out.csv 4096 500 20
```
Runs at full coverage only — 07c reads the `*_original` layers and ignores the downsampled ones, so
any `RESAMPLED_*` file may be passed (their `_original` layers are identical). Leiden clustering of
held-out φ against annotated cell types (ARI, purity, silhouette), compared with the same NMF
projections. Cell-type labels are used only for evaluation, never for fitting.

Both B3 and B4 infer φ through the same implementation used for training.

### B5 · Reproducibility across random initialisations
```bash
python $PIPE/A6_seed_consistency.py $MODEL_DIR A6_seed_consistency.csv \
       $MODEL_DIR/A5_repro_K10_phi_cache.npz
python $PIPE/A5_cluster_reproducibility.py $MODEL_DIR <train.h5ad> <test.h5mu> \
       A5_repro_K10.csv 10 12000
```
There is no ground truth on the atlas, so reproducibility is measured *between* fits that differ only
in their random seed. A6 reports it at three levels — program profiles Ψ (median matched *r* = 0.97),
per-cell loadings φ (median 0.65–0.87) and a Leiden clustering of φ (ARI ≈ 0.31) — which fall off for
the mechanical reason that Ψ is estimated from every cell whereas each φ*ᶜ* comes from one cell's
reads. A5 additionally checks that the held-out clusters track annotation rather than sequencing
depth or batch. **A5 caches its E-step φ**; A6 reuses that cache, so run A5 first if both are wanted.

### B6 · Program ↔ cell-type map, training vs held-out cells — every model, every seed
```bash
sbatch $MODEL_DIR/a7_all_models.sh          # array 0-8, three concurrent, ~1 GPU-hour total
# each task runs:
python $PIPE/A7_program_celltype_map.py $MODEL_DIR <task> <train.h5ad> <test.h5mu> \
       <phi_cache.npz> $MODEL_DIR/A7_prog_ct_task<task> 12000
#                             ^model_dir ^task ^train ^test ^phi cache ^out prefix ^n_test
```
Mean φ per annotated cell type, computed separately for training and held-out cells under the **same**
model — so programs need no matching and the comparison asks directly whether unseen cells use the
programs as training cells do.

Run for **all nine models** (*K* ∈ {10, 20, 30} × 3 seeds), which is the point: one model showing
transfer could be luck, nine showing it is a property of the method.

| input | value |
|---|---|
| train | `$WD/.../MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad` (92,894 cells) |
| test | `$WD/.../MODEL_INPUT/20260526/resampled/RESAMPLED_100_PERCENT_test_30_70_..._2026-06-17.h5mu` |
| cells scored | 12,000 held-out cells, sampled with `default_rng(0)` — the **same** rng and count A5 uses, so a cached φ corresponds to exactly these cells |
| coverage | **full only.** No coverage sweep, so this is one E-step per model, not a rate-limiting step |
| labels | `broad_cell_type`, used only to average φ — never during fitting |

Held-out φ comes from the frozen-globals E-step (Ψ, π, α, δ and `bb_conc` all fixed; only φ inferred),
chunked at 4,096 cells. Tasks 0–2 reuse the φ A5 already cached; tasks 3–8 compute and cache their
own. **Each task writes its own cache file** — an array writing one shared `.npz` would race.

Writes `A7_prog_ct_task<t>{_train,_test}.csv` and `A7_prog_ct_task<t>.npz` per model.
Result at *K* = 10: *r* = 0.998 / 0.999 / 0.999 across the three seeds, with **100% of the 27 cell
types keeping the same top program** on held-out cells in every seed.

### B7 · Do the lineage relationships replicate across seeds and K?
```bash
python $PIPE/A8_lineage_consistency.py $MODEL_DIR <train.h5ad> \
       $MODEL_DIR/A8_lineage_consistency.csv
```
Re-tests the lineage structure visible in B6 on **all nine models**. Uses only the training-side map,
which comes straight from `fit.npz` — **no E-step and no GPU**, so it runs in seconds on a login node
and does not contend with B3/B4. Each claim is posed as an ordering between two cell-type profile
similarities rather than as a statement about which program a type loads, so it needs no cross-model
matching and stays comparable across *K*. Excitatory-vs-inhibitory 9/9 models, microglia-with-myeloid
8/9, lymphoid-vs-myeloid 9/9 (Wilcoxon *P* ≤ 0.008).

### B8 · Seed consistency: why fits differ, and how to get one model
```bash
python $PIPE/A10_effective_K.py  $MODEL_DIR $MODEL_DIR/A10_effective_K.csv      # how many dims agree
python $PIPE/A11_seed_stability.py $MD2/A11_seed_stability.csv                  # convergence vs prior
python $SIMPY/A13_overdispersion_vs_reproducibility.py $WD/SIMULATION/A13.csv 3 cuda
python $PIPE/A14_consensus_subspace.py $MODEL_DIR 30 $MODEL_DIR/A14_consensus_K30.npz
```
Independently initialised fits agree on Ψ (*r* ≈ 0.97) and π (0.93) but only on ~30–50% of φ
directions. **A13** shows why, on simulated data where the truth is known: the same design at
overdispersion *c* = ∞/20/2 gives 80%/67%/33% reproducible directions, with accuracy against the known
φ degrading in step (0.990 → 0.788). The learned `bb_conc` ≈ 1.97 caps per-junction information at
~3 reads regardless of depth, which is also why per-cell agreement is flat across a 450-fold depth
range. **A11** refutes the two fixable candidates — doubling the training budget and dropping the
junction-specific prior each moved effective K by ±1 against a within-arm spread of 2.

**A14** is the deliverable: align every fit by Hungarian matching on Ψ (the well-determined latent),
apply that permutation to all program-indexed latents, and average — yielding one complete consensus
model that beats any single fit on leave-one-out at every K.

**A24** reports every latent at once (matched AND matching-free, with the matching margin), and **A25**
reports the biological output — whether independent seeds agree on which junctions are elevated in
which cell types, which needs no program matching at all and is the number to quote. **A23** does the
same for factor-level DS calls.

```bash
python $PIPE/A24_latent_correspondence.py $MODEL_DIR 6,7,8 out.csv --labels <train.h5ad>
python $PIPE/A25_celltype_junction_reproducibility.py $MODEL_DIR 6,7,8 <train.h5ad> out.csv
python $PIPE/A23_ds_reproducibility.py $MODEL_DIR 6,7,8 out.csv 100 0.1 0.9
```

The A15/A16-A22 initialisation experiments were removed after the comparison concluded (nothing beat
Pyro's default on the atlas); they remain in commit `bd923e4` and their results in
`INIT_COMPARISON_REPORT.md`. Rationale and alternatives: `SEED_CONSISTENCY_APPROACHES.md`.

---

# C. FIGURES

```bash
cd $FIG2 && python build_figures.py
```
| output | contents |
|---|---|
| `FIG2_validation` | A recovery vs effect size · B cell separation · C UMAPs · D imputation vs coverage · E calibration |
| `FIG2_supp_draft` | ATSE illustration, %-differential sweep per metric, prior ablation, minibatch/K checks |
| `FIG2_supp_heldout_structure` | ARI / silhouette / purity vs NMF |
| `FIG2_supp_program_celltype` | program x cell-type map, training vs held-out cells, K=10 seed 0 (B6) |
| `FIG2_supp_seed_consistency_sim` | phi consistency across seeds in simulation, by overdispersion and init (B8) |
| `FIG2_supp_program_celltype_K{10,20,30}` | the same map for **every seed** at each K — seeds across columns, training above held-out (B6) |

Means over seeds with s.e.m.; all pairwise comparisons Welch's *t*-test, `ns` when *P* ≥ 0.05.
Styling from `$REPO/figures/leaflet_theme.py`.

---

# Order of dependencies

```
A0 ─> A1 ─> A2 ─┐
     └─> A3 ────┼─> C (panels A, B, C)
                │
B0 ─> B1 ─> B2 ─┼─> B3 ─> C (panels D, E)
                ├─> B4 ─> C (structure supplement)
                ├─> B5 ─> B6 ─> C (reproducibility + program/cell-type supplement)
                └─> B7          (lineage replication across seeds and K; no GPU)
```
A and B are independent and can run concurrently. Within B, B1 → B2 is strict; B3, B4 and B5 are
independent of each other. B6 reuses the E-step φ cache that B5 writes, so B5 must run first.
