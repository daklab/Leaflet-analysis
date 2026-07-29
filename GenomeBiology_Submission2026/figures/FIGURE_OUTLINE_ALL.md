# Figure outline, Figures 1–5 — status and what remains

*Working record, 2026-07-29. Supersedes `FIGURE_OUTLINE_2_3.md`, which is stale: it references
`benchmark_sweep.py`, `sweep_results.csv` and the 2026-07-18 models (all replaced), and repeats the
"waypoints collapse seed variance ~130×" figure from **before** the Pyro-plate rewrite, which no
longer describes the trainer.*

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
0.55 → 0.87 vs NMF −0.05 → 0.49.

### 2b — Imputation on held-out cells 📊 DATA READY (A3 9/9)

| panel | content | status |
|---|---|---|
| E | imputation accuracy vs coverage (10–100%) — LeafletFA vs NMF 0-impute / mean-impute / naive / mean-Bayes | 📊 **build this first** |
| F | imputation at matched coverage, by K | 📊 |

Longest-running analysis in the project (~13 h for 9 models × 7 coverage levels); finished this
morning and nothing blocks the panels.

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
- **Model:** the released full-atlas fit. Pending the initialisation decision, that is
  `2026-07-27_full_K30_tight` (10 converged seeds, K = 30, 102–136 epochs), selected by **medoid**
  rather than best ELBO — those disagreed at K = 20 and K = 30, and ELBO selects for fit rather than
  representativeness.
- **LeafCutter:** exists from 2026-06-20 — `pipelines/08_leafcutter_ds/` (steps 00–04) and
  `$WD/LEAFCUTTER_DS/` (`by_celltype`, `by_tissue`, `ds_summary_all.tsv`, `leafcutter_counts.tsv.gz`).
  ❓ not inspected against the current models.

| panel | content | status |
|---|---|---|
| A | LeafletFA vs LeafCutter — concordance of differential junctions per cell type | ❓ |
| B | calibration — nominal posterior probability vs realised FDR | 🔧 |
| C | effect-size agreement — LeafletFA β vs LeafCutter ΔΨ, with the **slope**, not just correlation | 🔧 |
| D | what LeafletFA finds that LeafCutter cannot (or vice versa) | 🔧 |

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

**Running:** `19358688` — A20, simulation, three arms (default / waypoints-ψ / waypoints-ψ-and-φ) at
three overdispersion levels × 3 seeds, reporting consistency **and** accuracy against known φ.

**Also running:** `19358721` — `2026-07-29_atlas70_K30_waypoints`, the same comparison on the 70%
atlas, K = 30 × 3 seeds. Launched in parallel rather than gated behind A20, because simulation already
mispredicted the atlas once (jitter helped in simulation at c = 2, hurt on the atlas), so the atlas
result is the one that decides. Waypoint init verified working at scale: contrast 0.1494 against 0.000
for the default and 0.065 for jitter.

**Why waypoints:** SVD of centred usage ratios → max-min waypoints → metacell means. **Unstratified**,
so no cell-type label is read and the "without prior cell-type labels" claim holds. **Seed-dependent
by design**, so "three seeds agree" is a real result rather than a restatement of a fixed start.

**Then:** `A19_uncertainty_calibration.py` — does per-cell posterior width predict which cells the
seeds disagree on? If yes, seed variation is reportable as *calibrated uncertainty*. `A1` now saves
`phi_sd` and `phi_top_prob`.

---

## Immediate to-do, in order

1. **Build Fig 2E/2F (imputation)** — data ready since this morning, nothing blocks it.
2. **A20 + `19358721` land** → decide initialisation from the ATLAS result (simulation is advisory
   only, having mispredicted once). Watcher writes `figures/WAYPOINT_RESULT.md`.
3. **Fig 2A** hand-drawn cassette schematic — the only missing piece of an otherwise complete 2a.
4. **Inspect the Fig 3 LeafCutter outputs** and the Fig 4 / Fig 5 pipelines; replace ❓ with real status.
5. **Reconcile the DS/FDR machinery** before Fig 3 numbers are quoted.
6. Retire `FIGURE_OUTLINE_2_3.md`.

## Known loose ends
- `phi_uncertainty` logged `median sd nan` on the atlas though the saved array has 0% NaN — the printed
  statistic is `mean` mislabelled as median, and something in it is non-finite. Cosmetic; check before
  quoting.
- ARI moves the wrong way in the overdispersion arm under a changed init (0.505 → 0.182 at c = 20)
  while silhouette moves up on the same fits. Unexplained. ARI backs no headline claim, but unresolved.
