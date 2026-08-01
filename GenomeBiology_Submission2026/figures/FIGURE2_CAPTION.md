# Figure 2 (simulation half) — caption and how to read it

Figure file: `FIG2_simulation.{pdf,png}` · built by `code/panels.py::figure_simulation()`, driven by `code/build_sim.py`
Data: `benchmark_effect_sweep.csv` (12 simulations = 4 effect levels × 3 seeds) and `panel_data.npz`.

---

## Reading the x-axis: what "simulated effect size (median |ΔΨ|)" means

Every panel except E is plotted against the size of the splicing difference that was simulated.

**Per cassette**, ΔΨ is the **range of skipping usage across the five programs** — the difference
between the program that skips that exon most often and the one that skips it least:

```
ΔΨ = max_k Ψ_k − min_k Ψ_k
```

It is a range, **not** an average over program pairs. A real example, a cassette sitting at the
median of the `effect = 0.25` simulation:

```
skip usage per program:   0.73   0.65   0.84   0.86   0.85
                          max 0.86  −  min 0.65  =  ΔΨ = 0.21
```

That exon is skipped 86% of the time under one program and 65% under another — a 21-percentage-point
difference.

**Per simulation**, the axis value is the **median ΔΨ across the differential cassettes only**
(~1,280 of 4,268). Non-differential cassettes have ΔΨ = 0 exactly and are excluded.

| axis value | a typical differential exon shifts by … |
|---|---|
| **0.09** | ~9 percentage points between the most- and least-skipping programs |
| **0.21** | ~21 points (e.g. 65% → 86%) |
| **0.42** | ~42 points |
| **0.85** | ~85 points — a near-total switch (e.g. 8% → 93%) |

So the figure answers: **how small a splicing difference can the method still recover?** Reported
differential splicing in real data is typically 5–30 percentage points, i.e. the **left two levels**.

*Caveat:* because ΔΨ is a range it is set by the two extreme programs — four programs at 0.65 with
one at 0.86 scores the same as an evenly spread cassette. The estimator uses the identical definition
on fitted profiles, so truth and estimate are measured on the same footing.

The four levels are drawn at **evenly spaced positions**: this is a discrete four-level sweep, not a
continuous variable, and intermediate effect sizes were not sampled.

---

## Panels

**A — Effect-size recovery.** Spearman correlation between the true ΔΨ and the estimated ΔΨ across
differential cassettes: did the method recover *how much* splicing differs, not merely that it does?
LeafletFA rises from 0.41 to 0.96 as the simulated difference grows; **both NMF variants stay at ~0
at every effect size**, so NMF recovers no quantitative effect-size information at all. This is the
headline comparison.

**B — Cell separation.** Silhouette coefficient of the fitted cell loadings, labelled by each cell's
true program: are cells from the same program grouped together in the learned space? LeafletFA 0.55 →
0.87; NMF mean-impute begins negative (−0.05) and reaches only 0.49.

**C — Ψ recovery.** Median Pearson correlation between fitted and true program profiles after optimal
one-to-one matching. **Read this panel together with A and B.** NMF mean-impute scores 0.92 at the
smallest effect size and *falls* to 0.72 at the largest — a backwards trend, and the tell: when
programs barely differ, predicting each junction's mean correlates with every program profile while
capturing no program structure whatsoever (the same model has ~0 effect-size recovery in A and
negative silhouette in B). Ψ recovery alone therefore cannot distinguish "recovered the programs"
from "predicted the average" in the low-effect regime.

**D — Junction-specific Ψ prior.** The junction-specific prior is consistently slightly better than
a shared prior, but the difference is **not significant at any effect size** with three seeds. The
largest gap is at ΔΨ = 0.09 (Ψ recovery 0.962 vs 0.919, +0.043, *P* = 0.45), where seed-to-seed
variance is also largest; pooled across all four levels the differences are +0.004 to +0.021 with
*P* ≥ 0.46 on every metric. The one exception is AUROC for ranking differential cassettes, which
favours the junction-specific prior significantly at three of four levels (*P* = 0.019–0.023) but by
only 0.001–0.013 — reliable in direction, negligible in size.

The honest reading is that the prior does not measurably change recovery in this simulation.
Establishing whether the small positive trend is real would require more seeds; each fit takes about
two minutes, so this is inexpensive to settle if a reviewer presses on it.

**E — Recovered structure.** UMAP of the cell × program loadings coloured by each cell's *true*
program: the simulated φ (left), LeafletFA's inferred φ (middle), NMF's W (right); same cells, same
colours, so the panels are directly comparable. Silhouettes printed above each panel are computed on
the **latent representation**, not on the 2-D coordinates — UMAP inflates apparent separation, so
scoring the embedding would measure UMAP rather than the model. LeafletFA reaches 0.87, matching the
true φ's 0.87; NMF reaches 0.39.

---

## Statistics

Points are means over three simulation seeds; error bars are standard errors. Where pairwise tests
are shown they are Welch's *t*-tests with `ns` for *P* ≥ 0.05.

## Supplement

`FIG2_supp_draft` — the ATSE illustration (one differential and one non-differential cassette); UMAPs
of the simulated and recovered structure; the %-differential sweep resolved per metric; the
junction-prior ablation at atlas-like overdispersion (n = 10); and the minibatch check.

`FIG2_supp_program_celltype` — **mean loading of each program within each annotated cell type**, for
training cells (left) and held-out cells (right). Both panels come from the same *K* = 10 model, so
the programs are the same objects and the two maps can be compared directly; the held-out loadings
were inferred by the frozen-globals E-step on cells never seen in training. The maps agree at
*r* = 0.998, and all 27 cell types load most strongly on the same program in both.

This replaces an ARI against cell type as the summary of how the representation relates to cell
identity. Cell-type labels come from gene expression, so an ARI penalises any splicing structure that
does not coincide with cell identity, and it discards the continuous loadings the model actually
produces. Note this is a cell-type-level statement: averaging the cells within each type cancels the
per-cell noise, so it should be read alongside the per-cell reproducibility figures (median *r*
0.65–0.87), not in place of them. Labels are used only to build the map, never during fitting.

On the minibatch panel: full-batch and minibatch recovery differ by at most 0.004 (0.999 vs 0.995 at
K = 10), and both exceed 0.994 everywhere. Those differences are statistically significant
(*P* = 0.001 at K = 10) only because the trainer is extremely reproducible — full-batch recovery has
a standard deviation of 0.00001, so a 0.4% gap lies ~25 standard deviations out. Significance stars
are therefore not shown, and the axis is drawn over a wider range, because the panel's finding is
that minibatch training costs essentially nothing, not that it costs a measurable amount.
