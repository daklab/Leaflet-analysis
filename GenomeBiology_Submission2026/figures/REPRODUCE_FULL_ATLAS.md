# Reproducing the full-atlas models (2026-07-26)

Nine LeafletFA models trained on **every** cell of the mouse splicing atlas. This is distinct from
the 70/30 evaluation models (`2026-07-26_atlas9`), which are trained on 70% of cells so that the
remaining 30% can be held out for imputation and structure evaluation. The full-atlas models are the
ones to release and to use for downstream biology; they have no held-out set by construction.

```
WD    = /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD
PIPE  = /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa
MD    = $WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-26_atlas_FULL
PY    = /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
```

---

## Input

```
$WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad
```

**132,706 cells × 79,768 junctions.** Note the near-identical filename of the evaluation input:

| file | cells | used for |
|---|---|---|
| `train_splice_model_ready_2026-06-17.h5ad` | **132,706** | these full-atlas models |
| `train_splice_model_ready_70pct_2026-06-17.h5ad` | 92,894 | the 70/30 evaluation models |

92,894 is exactly 70% of 132,706, which is the quickest way to confirm the right file is in use.

---

## Configuration

`$MD/parameter_combinations.json` — **9 models**, *K* ∈ {10, 20, 30} × seeds {0, 1, 2}:

| setting | value |
|---|---|
| likelihood | **Beta-Binomial** (concentration learned) |
| Ψ prior | **junction-specific** |
| initialisation | **random** — `waypoints_use=false`, no cell-type information enters fitting |
| inference | minibatch SVI with a subsampling plate (`leafletfa.minibatch`) |
| batch size | 8,192 |
| learning rate | 0.05, decaying ×0.1 over the run |
| step cap | 8,000, with early stopping |

Both modelling choices are justified by simulation (see `METHODS_simulation_2026-07-26.md`):
Beta-Binomial and junction-specific priors are indistinguishable from their alternatives on
non-overdispersed data but significantly better at the overdispersion the atlas exhibits
(learned concentration ≈ 2).

**On the step cap.** 8,000 rather than the 5,000 used for the 70% models: per-step cost is
*B* × *J* and therefore independent of the number of cells, so the same step count is fewer passes
over a larger dataset (5,000 steps = 441 epochs at 92,894 cells but only 309 at 132,706). The cap is
raised so early stopping, not the budget, decides when to stop. The 70% models stopped at
750–1,350 steps, so it is not expected to bind.

---

## Running

```bash
sbatch $MD/train_full_atlas.sh          # array 0-8, three concurrent
# each task runs:
$PY $PIPE/A1_train_atlas.py <task_id> $MD <train.h5ad> 8000 8192 0.05 0.1 0
#                                              ^cap ^batch ^lr  ^gamma ^learn_ab
```

**Resources.** 340 GB RAM and one L40S per task. The trainer densifies both count matrices:
132,706 × 79,768 × 4 bytes = 42 GB each, 85 GB together. GPU nodes cap at 370 GB, so 340 GB is
close to the ceiling and only one task fits per node. The model keeps those matrices in host memory
(`data_on_cpu` auto-enables above 4 GB) and moves only the 8,192 × 79,768 minibatch (2.6 GB) to the
GPU each step, so GPU memory is not the constraint.

---

## What is logged

Per model, in `$MD/mb_run_<task_id>/`:

| output | contents |
|---|---|
| `result.json` | K, seed, prior, cap, batch, lr, gamma, final loss, trainer, wall time |
| `fit.npz` | `psi`, `phi`, `pi`, `alpha_pi`, `dir_conc`, `bb_conc`, `a`, `b`, the full loss trace, and **`psis_loc`/`psis_scale`** — the Ψ posterior, so downstream evaluation samples Ψ rather than plugging in a point estimate |
| `loss_curve.png` | ELBO trace with the early-stopping point |
| `phi_umap.png` | UMAP of the cell loadings, coloured by broad cell type (diagnostic only — labels are not used in fitting) |
| stdout, every 50 steps | step, loss, mean Ψ |
| **Weights & Biases** | project `LeafletFA-mouse-atlas-FULL` (set via `LFA_WANDB_PROJECT`): the loss series plus final health scalars |

---

## Checking a run converged

The 70/30 models give the reference values. Read them out of `fit.npz`:

```bash
$PY -c "
import numpy as np, json, glob
for f in sorted(glob.glob('$MD/mb_run_*/fit.npz')):
    d=np.load(f); r=json.load(open(f.replace('fit.npz','result.json')))
    p=np.clip(d['phi'],1e-12,None)
    print(f\"K={r['K']:>2} seed={r['seed']}  psi_mean={d['psi'].mean():.4f}  \"
          f\"contrast={d['psi'].std(0).mean():.4f}  bb_conc={float(d['bb_conc']):.2f}  \"
          f\"perplexity={np.median(np.exp(-(p*np.log(p)).sum(1))):.1f}/{r['K']}\")
"
```

Expected, based on the 70% models:

| quantity | expected | why it matters |
|---|---|---|
| mean Ψ | ≈ 0.30 | should match the observed splicing rate |
| Ψ contrast | ≈ 0.08 | programs genuinely differ; near zero means they collapsed together |
| `bb_conc` | ≈ 1.9–2.0 | the overdispersion the data carries; reproduced tightly across seeds |
| perplexity | 2–4 of *K* | cells commit to a few programs rather than spreading over all |
| early stop | ≈ 750–1,350 steps | hitting the 8,000 cap would mean it never converged |

Agreement across the three seeds at each *K* is the main check: the plate implementation is highly
reproducible (Ψ recovery s.d. 0.001 in simulation), so scattered globals would indicate a problem.

---

## Relationship to the other model sets

| model set | trained on | purpose |
|---|---|---|
| `2026-07-26_atlas9` | 70% of cells | held-out imputation (A3) and structure (A4) evaluation |
| **`2026-07-26_atlas_FULL`** | **all 132,706 cells** | **the released models; downstream biology** |

Only the 70/30 models can be evaluated on held-out cells; only the full models use all the data.
They share code, hyperparameters and seeds, so they are directly comparable.
