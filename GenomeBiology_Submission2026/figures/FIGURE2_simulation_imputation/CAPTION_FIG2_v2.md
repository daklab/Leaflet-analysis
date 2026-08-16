# Figure 2 caption — v2 draft 2026-08-16

**Figure 2. LeafletFA recovers simulated splicing programs and the exons that differ between
them, without imputing the input.**
Simulations start from 6,986 heart cells of the atlas: each cell keeps its real per-ATSE read depth
and receives a mixture of K = 5 splicing programs (φ, Dirichlet concentration 20); every one of 4,268
cassette-exon ATSEs receives a per-program skip-junction usage ψ, identical across programs for 70%
of ATSEs and shifted by the stated effect for 30%; counts are drawn per cell and ATSE as a
Dirichlet-multinomial (concentration 50) over the ATSE's three junctions.
**(a)** The simulated cells: UMAP of the true φ, coloured by each cell's dominant program.
**(b)** One ATSE simulated as differential: black bars are the true skip usage of each program;
violins/points are the observed per-cell skip-junction ratio of the cells belonging to that program.
This is what the model has to find from counts alone.
**(c)** One non-differential ATSE: same usage in every program. This is what it must not call.
**(d)** UMAP of the φ LeafletFA learns from the counts, coloured by the TRUE program: five clean
groups (silhouette on the latent space 0.82 vs 0.87 for the truth).
**(e)** NMF at the same rank, on the mean-imputed usage matrix (its best-scoring gauge): programs
merge (silhouette 0.41).
**(f)** Detecting which ATSEs are differential across programs, with no labels: each ATSE is scored
by max_k ψ̂_k − min_k ψ̂_k over the learned programs and ranked against the simulated truth
(AUROC, solid; AUPRC, dashed; 30% of ATSEs are differential). LeafletFA vs NMF at the same rank
(NMF at its best gauge per effect).
**(g)** Differential-splicing calibration on the same simulations, three tests against the same
truth: realised over nominal FDP at α = 0.05. LeafletFA's DM likelihood-ratio test on single cells,
LeafletFA's posterior effect-size test P(TV > 0.15) (shown from effect 0.25, below which no true
effect can exceed δ), and LeafCutter on 7-vs-8 pseudobulks. Both LeafletFA tests reach calibration
by effect 0.35; LeafCutter is 2.2× above nominal even at effect 1.0.
**(h)** Power at α = 0.05, read together with (g): the LRT matches or exceeds LeafCutter at every
effect; the posterior test is deliberately conservative (it asks for TV > 0.15, not TV > 0).
**(i)** Held-out imputation on the atlas (30% of cells never seen in training; reads removed at
increasing rates): [A3 sweep pending — DM posterior mean vs NMF]. Supporting analysis
`plots/SANITY_imputed_vs_raw_heldout.png` (1,500 held-out cells, entries covered at full coverage
with ≥5 reads; DM posterior mean vs NMF rank 20 on the mean-imputed matrix): over all entries the
DM posterior mean gives r 0.99/0.99/0.98/0.94 at 0/10/50/90% of reads removed (NMF 0.87 throughout);
within a junction (ranking cells) 1.00/0.98/0.95/0.74 vs NMF 0.23/0.22/0.21/0.18; on entries whose
reads were ALL removed (the imputation task proper) within-junction r is 0.14 vs NMF 0.06 and
within-cell r 0.85 vs 0.83 — both methods predict a junction's level far better than a cell's rank
within it, which is expected at ~2 effective molecules per cell.

Caveats for the text: the simulations sit at median 33 reads/ATSE and DM concentration 50 (the
atlas: 16 reads and an effective sample size of ~2 molecules), so power here is an optimistic
bound; the concentration-2 regime and the depth dependence are in the supplement.

**Supplement (from FIG2_final):** concentration recovery; ψ recovery vs read depth; φ recovery by
training regime; the δ calibration–power trade-off; the fraction of the atlas any δ can reach;
DS at concentration 2.
