# Held-out latent structure of φ — interpretation for the paper

How much cell-type structure the splicing latent φ carries, why it is less than gene expression,
and how to visualise it honestly. Numbers are from the minibatch K=20 model, 70/30 split
(`A4_eval_structure.py`, held-out 30% via the frozen-Ψ E-step; training φ from the fitted model).

## The result

φ recovers the **major lineages** but not fine subtypes — and it recovers broad cell types markedly
better than fine tissue-specific ones:

| φ aligns with… | ARI (held-out) | ARI (train) | silhouette (held-out) | silhouette (train) |
|---|---|---|---|---|
| **broad cell type** (39 classes) | 0.173 | 0.198 | −0.086 | −0.008 |
| tissue cell type (182 classes) | 0.117 | 0.151 | −0.281 | −0.304 |

The held-out↔train gaps are small (ARI −0.025), so this is **generalisation, not memorisation** — the
released model gives essentially the same structure on cells it never saw.

This is exactly what the φ UMAP shows: distinct blobs for the big lineages (neurons, epithelium,
immune/myeloid, muscle), blurring *within* each. That coarse-but-not-fine signature is the point.

## Why less structure than gene expression — four reasons, in order of weight

**1. Splicing is a fundamentally sparser signal (the dominant reason).** Coverage is ~13.5%, and PSI
is a ratio estimated from a handful of reads per junction; gene expression carries orders of magnitude
more information per cell. So splicing resolves coarse axes and washes out fine ones. The data shows
this directly: φ recovers broad lineages (ARI 0.173) far better than fine tissue subtypes (0.117),
and the fine-class silhouette is much worse (−0.281 vs −0.086). This is the expected fingerprint of a
sparse modality, **not a failure of the model**.

**2. The representation is compositional (bounded dynamic range).** φ_c is a point on the K-simplex —
it sums to 1 and each entry is in [0,1] — whereas a gene-expression embedding lives in an unbounded
continuous space. Two cells can differ at most by the simplex diameter, and because cells commit to
only ~3–4 of K programs (perplexity 3.4/10, 5.5/20), most φ sit in the simplex *interior*, not at
vertices. This structurally **compresses distance-based metrics**: it is the main reason
silhouette-by-cell-type is near zero / slightly negative even where clustering agreement (ARI) is
genuinely non-trivial. Read ARI (partition agreement) as the primary structure metric here, and treat
silhouette as compression-limited rather than as evidence of no structure.

**3. Cell-type labels are defined from gene expression, so part of the gap is real biology.** A
splicing program can span several GE-defined types, or one GE-type can mix splicing programs.
Negative silhouette-by-cell-type means GE-defined types are not compact in *splicing* space — which is
the interesting result if splicing carves cells along partly orthogonal axes (activation state,
tissue-of-origin) rather than GE identity. We weight this **below** reasons 1–2: sparsity and
compositionality are simpler and account for most of the gap, so this claim is made cautiously and
supported, not assumed, by the overlay analysis below.

**4. A confound that makes the held-out number pessimistic — but not the UMAP.** The held-out ARI is
computed on φ from the test-time E-step, which is **under-converged** (it does not early-stop within
the epoch budget; see the E-step convergence note). Direct evidence: held-out φ perplexity is 4.49
versus 3.40 for the converged training φ — the held-out representation is *more diffuse* because it is
truncated, which lowers its apparent structure. The **training-φ UMAP is fully converged** (training
loss plateaus), so it is the more faithful picture; trust it over the held-out ARI. Resolving E-step
convergence is expected to raise the held-out numbers toward the training values.

## Recommended visualisation — splicing programs on a gene-expression map

Head-to-head "does the φ UMAP recover GE clusters" is the wrong framing for a modality that is not
gene expression (it cannot, by reasons 1–2, and should not be expected to). The more informative and
honest figure — the approach used previously — is to **use the GE embedding as the canvas and overlay
splicing activity**:

1. Build the cell map from **gene expression with scVI** (the standard, well-powered embedding).
2. Overlay **per-program φ activity** on that map, one panel per program: colour each cell by φ_ck.
3. Read off, per program, whether it (a) marks a coherent GE region (a splicing program that tracks a
   cell type), or (b) spans / subdivides GE structure (a splicing axis GE does not capture — the
   selling point).

This turns "φ has less structure than GE" (a weakness framing) into "here is what splicing adds on top
of GE" (the actual contribution), and it makes reason 3 testable rather than asserted: a program whose
activity forms a coherent island *within* a GE cell type, or bridges two GE types, is orthogonal
splicing biology one can point to.

## What to report in the paper

- ARI (broad and tissue) with the small held-out↔train gap as the generalisation check.
- The φ UMAP (training, converged) for the coarse-lineage structure.
- The scVI-map + φ-overlay panels for the interesting programs (primary structural figure).
- State plainly that silhouette is compression-limited on the simplex; lead with ARI.
- Note the E-step convergence caveat on any held-out φ metric until it is resolved.
