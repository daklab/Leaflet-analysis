# Figure 2 caption — v2 draft 2026-08-16

**Figure 2. LeafletFA recovers simulated splicing programs and the exons that differ between
them, without imputing the input.**
Simulations start from 6,986 heart cells of the atlas: each cell keeps its real per-ATSE read depth
and receives a mixture of K = 5 splicing programs (φ, Dirichlet concentration 20); every one of 4,268
cassette exons receives a per-program skip-junction usage ψ, identical across programs for 70% of
exons and shifted by the stated effect for 30%; counts are drawn per cell and exon as a
Dirichlet-multinomial (concentration 50) over the exon's three junctions.
**(a)** The simulated cells: UMAP of the true φ, coloured by each cell's dominant program.
**(b)** One exon simulated as differential: black bars are the true skip usage of each program;
violins/points are the observed per-cell skip-junction ratio of the cells belonging to that program.
This is what the model has to find from counts alone.
**(c)** One non-differential exon: same usage in every program. This is what it must not call.
**(d)** UMAP of the φ LeafletFA learns from the counts, coloured by the TRUE program: five clean
groups (silhouette on the latent space 0.82 vs 0.87 for the truth).
**(e)** NMF at the same rank, on the mean-imputed usage matrix (its best-scoring gauge): programs
merge (silhouette 0.41).
**(f)** Recovery of which exons differ between programs — Spearman correlation between the true and
estimated between-program Δψ across exons — by simulated effect. LeafletFA 0.21–0.85; NMF 0.0–0.24.
**(g)** Differential-splicing calibration on the same simulations, three tests against the same
truth: realised over nominal FDP at α = 0.05. LeafletFA's DM likelihood-ratio test on single cells,
LeafletFA's posterior effect-size test P(TV > 0.15) (shown from effect 0.25, below which no true
effect can exceed δ), and LeafCutter on 7-vs-8 pseudobulks. Both LeafletFA tests reach calibration
by effect 0.35; LeafCutter is 2.2× above nominal even at effect 1.0.
**(h)** Power at α = 0.05, read together with (g): the LRT matches or exceeds LeafCutter at every
effect; the posterior test is deliberately conservative (it asks for TV > 0.15, not TV > 0).
**(i)** Held-out imputation on the atlas (30% of cells never seen in training; reads removed at
increasing rates): [A3 sweep pending — DM posterior mean vs NMF]. Supporting analysis
`plots/SANITY_imputed_vs_raw_heldout.png`: the pure prior φ·ψ recovers each junction's usage level
(r = 0.87 over all entries) but not the ranking of cells within a junction (median r = 0.11);
the DM posterior mean, which combines the prior with the cell's own remaining reads, holds
within-junction r 0.98/0.95/0.74 at 10/50/90% of reads removed, and reverts to the prior (0.14) on
entries with no reads left.

Caveats for the text: the simulations sit at median 33 reads/ATSE and DM concentration 50 (the
atlas: 16 reads and an effective sample size of ~2 molecules), so power here is an optimistic
bound; the concentration-2 regime and the depth dependence are in the supplement.

**Supplement (from FIG2_final):** concentration recovery; ψ recovery vs read depth; φ recovery by
training regime; the δ calibration–power trade-off; the fraction of the atlas any δ can reach;
DS at concentration 2.
