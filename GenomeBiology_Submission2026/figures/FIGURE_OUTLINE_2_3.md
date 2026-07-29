# Figure outline — Fig 2 (simulation), Fig 3 (real-data validation), Supp (RNA atlas)

*Working outline. Status legend: ✅ data ready · ⏳ compute running · 🔧 needs building.*

The three eval modes answer **different** questions, so they are separate figures:

| | question | truth source |
|---|---|---|
| **Fig 2** simulation | can it recover *known* structure? | synthetic ground truth |
| **Fig 3** real-data | does it generalise to held-out cells & shallow depth, and agree with the standard tool? | real data |
| **Supp** RNA atlas | does the single-cell dictionary transfer to bulk? | external application |

Escalation: **controlled → real-data → biology** (Fig 4 aging, Fig 5 conservation).

---

## FIGURE 2 — Simulation / controlled recovery   ("trust the model")

Sweep axis: **% of positive (DS) events = 10 / 50 / 90 %**, 3 sim seeds each, K=5 (true K),
Binomial-truth sims. Source: `benchmark_results.csv` (`benchmark_sweep.py`).
**5 panels.** PR curve and φ-UMAP moved to supplementary.

| panel | plot | x / series | message | status |
|---|---|---|---|---|
| **A** | schematic | — | how sims are built: exon-skip ATSEs → assign positive/negative → J1/J2/J3 Ψ ~ Beta → counts ~ Binomial | 🔧 hand-drawn (have the old A) |
| **B** | silhouette vs %DS | x=%DS; lines = **Leaflet-FA vs NMF(mean-impute) vs NMF(0-impute)** | latent space separates cells by program; LeafletFA beats NMF at every DS level and under either imputation | ✅ (LeafletFA line = reference config bb+junc-prior only) |
| **D** | Spearman(sim Δ, est Δ) vs %DS | x=%DS; by **likelihood** (BB vs Binom) | recovers effect *size*, not just presence; BB ≥ Binom | ✅ |
| **E** | precision vs junction-specific prior | x=prior {NO,YES}; by likelihood | DS-detection precision; is the junction prior load-bearing? | ✅ |
| **G** | silhouette vs junction-specific prior | x=prior {NO,YES}; by likelihood | cell separation vs prior | ✅ |

Notes:
- **B must use the reference LeafletFA config** (Beta-Binomial, junction prior) × 3 seeds — not all
  12 configs pooled, or the "LeafletFA" line mixes in Binomial/global-prior variants.
- E and G both probe the junction prior (precision + silhouette). Kept both because reviewers will
  press on the prior; drop one if space is tight.

**Supplementary to Fig 2:**
- lr × waypoints headline: waypoints rescue every lr and collapse seed variance ~130× ✅ (`sweep_results.csv`)
- "which axis matters" OFAT panel ✅
- K-pruning: does over-specified K collapse back? ✅ (`pruning.csv`)
- φ-UMAP coloured by true program (the intuitive "cells cluster by program" view) ✅ (`representative_fit.npz`)

---

## FIGURE 3 — Real-data validation   ("works on real data")

Held-out **30 %** cells, models from the 70 % atlas grid (`LEAFLETFA_MODELS/2026-07-18`).
Self-contained **without** bulk data.

### 3.1 Coverage robustness (imputation)   ⏳ needs 04e on the new models
Read counts thinned to **10/25/50/75/90/99/100 %**; φ inferred under frozen globals; `φ·Ψ` scored
against ORIGINAL full-depth PSI. Source: `A3_eval_imputation.py` × `resampled/`.

| panel | plot | message |
|---|---|---|
| **A** | median Pearson vs coverage %, one line per method (lf, lf_bb, **mb_ct**, mb, naive), error bars over 3 seeds | recovers full-depth PSI from shallow data; **the bar to clear is `mb_ct`** (cell-type-stratified mean) — beating it means programs add signal beyond cell identity |
| **B** | calibration: predicted vs observed PSI, binned, at one coverage level, **slope annotated** | Reviewer 1 ask — correlation hides compression toward the mean; slope≈1 = calibrated, >1 = under-dispersed |
| **C** | calibration slope vs coverage | does it stay calibrated as depth drops? |

### 3.2 Held-out latent structure   ⏳ 07c smoke running (`A4_eval_structure.py`)
Frozen-Ψ E-step on the 30 % → **unsupervised** structure of φ (labels NOT used for fitting).

| panel | plot | message |
|---|---|---|
| **D** | silhouette of φ **by cell type** vs ARI(Leiden, cell type) | cell-type labels are GE-derived; the CONTRAST (internal structure vs cell-type alignment) shows φ captures real splicing structure not identical to cell type |
| **E** | Leiden-vs-cell-type contingency / per-cluster purity | which unsupervised splicing clusters map to which cell types |
| **F** | memorization gap: train vs held-out (silhouette/ARI) | small gap = generalises, not memorised |

### 3.3 Concordance with Leafcutter (pseudobulk)   🔧 needs building
Aggregate the single cells → **pseudobulk** → run Leafcutter → compare to LeafletFA's implied PSI on
shared junctions. No bulk data needed.

| panel | plot | message |
|---|---|---|
| **G** | LeafletFA PSI vs Leafcutter PSI, shared junctions | "our quantification agrees with the field-standard tool" |

---

## SUPPLEMENTARY — RNA atlas (bulk transfer)   🔧 last, if time

Apply the trained single-cell dictionary Ψ to **bulk** RNA-seq: frozen-Ψ E-step treats each bulk
sample as one pseudo-cell → φ = program activity per bulk sample. No retraining. **Chains** with
calibration (bulk calibration REQUIRES this projection first); not redundant with it.

| panel | plot | message |
|---|---|---|
| **S1** | reconstructed bulk PSI (φ·Ψ) vs Leafcutter on the same bulk, shared junctions | agrees with Leafcutter on bulk too (validation) |
| **S2** | bulk samples in program space (φ) coloured by tissue/condition/age | **the capability Leafcutter cannot provide**: places decades of existing bulk RNA-seq into the single-cell program space |
| **S3** *(ambition)* | a single-cell aging program (Fig 4) shown elevated in old bulk samples | loop closure: the same program is portable and biologically meaningful in independent bulk data |

Caveats to state: shared-junction intersection must be large enough; a bulk φ is a
composition-weighted mixture of programs; concordance-with-Leafcutter is what keeps the projection
honest. Needs: a bulk cohort with metadata worth recovering + Leafcutter run on it.

---

## What's blocking each figure right now
- **Fig 2:** ready to render (sweep 100/100, benchmark recomputing NMF variants). Panel A schematic to draw.
- **Fig 3:** gated on the atlas finishing (24/48). Then run 04e (coverage) + 07c array + build pseudobulk-Leafcutter.
- **Supp:** needs a bulk dataset chosen + Leafcutter; lowest priority.
