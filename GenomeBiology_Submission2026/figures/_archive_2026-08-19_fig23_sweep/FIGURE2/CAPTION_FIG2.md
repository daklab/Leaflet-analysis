# Figure 2 caption — draft 2026-08-15

**Figure 2. LeafletFA on simulated data with known truth, and on held-out atlas cells.**
Simulations use real Tabula Muris Senis heart cells (6,986 cells, 4,268 cassette ATSEs): every
cell's read total per event is its real value, so sparsity is that of the data; junction counts are
Dirichlet-multinomial draws (concentration 50 unless stated) from K = 5 programs, 30% of ATSEs
differential with an effect parameter controlling program separation.

**(a)** Recovery of the true between-program effect (Spearman) against non-negative matrix
factorisation at the same rank, NMF scored at whichever missing-value convention favours it.
**(b)** Recovered against simulated concentration (10 / 50 / 200 -> 10 / 49 / 183); the overdispersion
parameter is identified.
**(c)** Recovery of program compositions psi against read depth per ATSE, by effect size. Recovery
rises with coverage.
**(d)** Differential-splicing calibration -- realised over nominal FDP at alpha = 0.05 -- for LeafletFA's
likelihood-ratio test on single cells and LeafCutter on 6-replicate pseudobulks of the same
simulated cells, scored against the same contrast-specific truth. LeafletFA reaches nominal at effect
~0.30; LeafCutter does not at any effect tested. Both are anti-conservative below effect 0.20.
**(e)** Power at alpha = 0.05 for the same runs; to be read only alongside (d).
**(f)** Recovery of the true cell loadings phi under the atlas's regime (concentration 2, atlas
depth), for the shipped configuration and the alternatives it was chosen over. Pinning the loading
prior at delta = K recovers 0.58 of the truth against 0.25 when delta is learned; delta = 5K is worse
than either. The recipe is a partial fix, and the residual is stated.
**(g)** Choosing the minimum effect delta for the posterior test: realised-over-nominal FDP against
power with delta as the point label. delta = 0.15 sits at 1.2-2.0x nominal with power 0.46-0.90 for
effects >= 0.35; delta = 0.05 is 5-6x anti-conservative because it lies below the posterior standard
deviation of psi itself. Cells with fewer than ten calls omitted.
**(h)** What fraction of atlas (contrast, ATSE) pairs exceed a given effect, measured by LeafCutter's
pseudobulk estimates: 28% above 0.10, 18% above 0.15. The posterior test speaks to roughly the top
fifth of the atlas.
**(i)** [PENDING -- A3 running] Held-out imputation: model refit on 70% of cells, reads of the held-out
30% downsampled to 10-99%, loadings inferred with all global parameters frozen; imputed usage against
each cell's full-coverage value, versus NMF, per-junction mean, and cell-type mean (the last using
labels the model does not see).
**(j)** [PENDING -- conc-2 arm running] Panels d/e at concentration 2, the effective sample size
measured on the atlas: LeafletFA's LRT is 6.7x / 2.5x / 1.3x nominal at effects 0.25 / 0.5 / 1.0
against 1.1x / 0.8x / 0.35x at concentration 50, i.e. calibration is harder at the data's real
overdispersion; LeafCutter's numbers on the same simulations to follow.

Captions must state: simulations sit at median 33 reads per ATSE against the atlas's 16, so every
power number is an optimistic bound; delta = 0.15 was derived from permuted negative-control ATSEs
(gate: permuted vs true-null TV ratio 0.93-1.02; atlas floor p99 = 0.13), not chosen.
