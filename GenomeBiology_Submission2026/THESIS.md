# LeafletFA (Genome Biology 2026) — Paper Thesis & Analysis Compass

> Standing document. The "why" behind every figure and analysis. Read before
> designing a panel or an eval so we don't drift or overclaim.

---

## One-sentence thesis

**How much cell-type–resolved splicing variation is hidden in high-coverage
single-cell data — across *and* within cell types — and can a Beta-Binomial
factor model (LeafletFA) extract that structure and impute it from sparse data
reliably enough to serve as a resource (and training signal) for single-cell
foundation / virtual-cell models?**

---

## The big question

Alternative splicing is a major, under-explored axis of single-cell state. We ask:
- **Across cell types:** how much does junction/ATSE usage (PSI) differ between cell
  types, and which genes/ATSEs carry that variation?
- **Within cell types:** how much residual splicing heterogeneity exists *inside* a
  cell type, beyond the cell-type average?
- **Recoverability:** single-cell splicing reads are sparse and noisy. How much of the
  true (high-coverage) signal can we recover from degraded coverage — i.e., is the
  learned structure robust enough to transfer to low-coverage cells?

## Why this is hard with existing tools

- Tools like **LeafCutter** were built for bulk/pseudobulk differential splicing between
  predefined groups; they don't model per-cell latent structure and struggle with
  single-cell sparsity. (The paper shows the specific failure modes.)
- We need a method that (a) borrows strength across cells via shared latent programs,
  (b) respects count uncertainty (Beta-Binomial), and (c) produces per-cell PSI
  estimates that degrade gracefully with coverage.

## Why a factor model (LeafletFA)

- **psi** (K × J): K splicing *programs* — PSI profiles across junctions.
- **phi** (N × K): each cell's mixture over programs.
- `phi @ psi` = a low-rank, denoised reconstruction of the cell × junction PSI matrix.
- Beta-Binomial likelihood weights each observation by its coverage/precision.

This directly instruments the thesis: **phi/psi capture the cross-cell-type program
structure; the Beta-Binomial posterior lets us impute per-cell PSI from sparse counts.**

---

## What the model can and cannot do (established, keep honest)

Diagnosed empirically (see `memory/project_imputation_eval_diagnosis.md`):

- **Across-cell-type structure → captured well.** `phi@psi` reconstructs between-junction
  and cell-type-level PSI structure; phi predicts cell type well above chance
  (classification eval). This is the model's strength and a core result.
- **Within-junction, cell-to-cell ranking at the most variable junctions → NOT captured
  by `phi@psi` alone — BY DESIGN.** LeafletFA is built to model junctions that **co-vary
  across cells** (shared low-dimensional programs = axes of *coordinated* splicing). A
  single junction varying idiosyncratically across cells within a type is, by construction,
  outside what shared programs represent. So per-junction Pearson of pure `lf = phi@psi`
  ≈ 0 (ceiling ≈ 0.017, *same at K=5 and K=20* → not a rank/K problem) is a direct
  consequence of the modeling assumption, not a bug.
  - **Reportable scientific reading:** the residual variation at highly variable junctions
    is largely *junction-specific / not part of coordinated cell-state programs* — i.e.,
    much single-cell splicing heterogeneity at these junctions is local rather than a
    shared program signal.
  - **Honesty caveat (must address):** some apparent within-junction variation at HVJ is
    **measurement noise** (sparse counts → noisy per-cell PSI), not biology. Use coverage
    stratification / the downsampling analysis to argue how much is real vs shot noise
    before claiming "within-type biological heterogeneity."
- **Therefore the deployable imputation is the Beta-Binomial posterior, `lf_bb`**, which
  folds the observed counts back in per junction:
  `lf_bb = (phi@psi·bb_conc + obs_junc) / (bb_conc + obs_atse)`.
  It is *not* cheating — it uses only observed (degraded) counts, never the held-out truth.
- **Metric lens matters.** *Per-junction* correlation tests within-junction cell ranking
  (hard). *Global* correlation over pooled pairs is dominated by the easy between-junction
  axis (a per-junction-mean baseline scores ~0.83 there). Report both; never headline the
  global number alone.

---

## Claims / deliverables (what the paper delivers)

1. **Quantify splicing variation across vs within cell types** using learned phi/psi:
   rank genes/ATSEs by cross-cell-type PSI variation; characterize residual within-type
   heterogeneity. → "these genes carry high cell-type splicing variation."
2. **Robustness-to-coverage validation (separate experiment).** Using a 70/30 split and
   read-**downsampling** (10→99% dropout), show the model recovers high-coverage PSI from
   degraded input, and that imputation degrades gracefully as counts drop (fewer counts →
   noisier per-cell `phi` → worse imputation). This is a *validation* experiment that earns
   confidence in the imputed values — it is **not** how the released resource is produced.
3. **Released resource (full-data fit, no held-out transfer).** Fit LeafletFA on **all**
   cells; the released junctions/ATSEs × cells AnnData carries **`lf_bb`** imputed PSI from
   that full model. Each cell keeps its observed PSI where covered and gets `phi_i @ psi`
   where not — a **per-cell** estimate resolved to the granularity of the learned programs
   (`phi`), potentially finer than cell type. Offered as an exploratory resource and a
   candidate track/signal for single-cell foundation / virtual-cell models.
4. **Method comparison:** LeafletFA vs baselines that lack cell-specific learned structure
   (naive per-junction mean, cell-type-stratified MeanBayes `mb_ct`, NMF) — the sharp test
   is whether the learned program prior beats *cell-type averaging* when counts are scarce.

## How the analyses map to the thesis

| Analysis | Serves | Metric / output |
|---|---|---|
| Imputation eval (downsampling, GPU 04e) | Recoverability / robustness; resource confidence | `lf_bb` vs `mb`/`mb_ct`/NMF, per-junction + global + MAE across dropout |
| Classification / aging (07) | Across-cell-type & age structure in phi; model selection | balanced acc (broad/tissue cell type), age R², effective_k |
| Cross-cell-type variation ranking (phi/psi) | "high splicing variation" gene/ATSE list | per-ATSE cross-cell-type PSI variance from psi/phi |
| LeafCutter comparison (RNA atlas, bulk) | Why existing tools fall short / validation | DS concordance |

## Framing guardrails for figures

- **Headline `lf_bb` at high dropout** (90–99%) for the imputation claim — that is where
  the model's prior actually does work; low dropout is near-trivial (observation ≈ truth).
- Isolate the model with **count-fair baselines**: `lf_bb` vs `mb`/`mb_ct` all use the same
  observed counts and differ only in how they fill *unobserved* cells (program prior vs
  population vs cell-type mean). NMF must get the same count-blend to sit on the same axis.
- Use **per-junction (cell-ranking)** as the rigorous metric; show **global/MAE** for scale
  and for continuity with prior work — but label the global one as the "easy" axis.
- The released imputed values = **`lf_bb`** from the **full-data fit** (not the held-out
  downsampling model). State plainly what they represent: a **per-cell** PSI estimate
  (observed where covered; `phi_i @ psi` where not), varying across cells through the
  K-dim `phi` — as fine as `phi` resolves cell state (can be sub-cell-type), but not
  junction-specific within-cell deviations outside the learned programs.

## Relevance to foundation / virtual-cell models

Single-cell foundation models increasingly ingest multiple modalities/tracks. A denoised,
coverage-robust per-cell splicing (PSI) layer — with an explicit program decomposition —
is a candidate input signal or evaluation track. The imputed AnnData is offered in that
spirit: a usable resource plus an honest account of its resolution limits.

---

_Related standing notes:_ `WORKFLOW.md` (pipeline mechanics);
`memory/project_imputation_eval_diagnosis.md` (why `lf`≈0, ceiling, E-step);
`memory/project_rna_atlas_leafcutter.md` (bulk LeafCutter comparison TODO).
