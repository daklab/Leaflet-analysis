# Figure 3 caption — draft 2026-08-15

**Figure 3. Differential splicing on the atlas: pseudobulk LeafCutter against single-cell LeafletFA.**
Seventy pairwise cell-type contrasts within tissues from Tabula Muris Senis. LeafCutter was run on
pseudobulk replicates (cells aggregated per mouse, 7 vs 8 per contrast), the realistic design for
Smart-seq2 differential splicing; LeafletFA's likelihood-ratio test uses the same Dirichlet-multinomial
GLM on every individual cell.

**(a)** Null p-values from each method's own permutation null (LeafCutter: labels shuffled within a
cell type; LeafletFA: cells of one type split at random), against uniform. Distance above the diagonal
is over-calling. LeafletFA's null is contaminated by residual donor structure and is therefore an
upper bound on its inflation.
**(b)** Realised over nominal false-discovery proportion at alpha = 0.05 as a function of junctions per
ATSE. Pseudobulk LeafCutter degrades from 3.2x at m = 2 to 11.4x at m = 9; single-cell LeafletFA stays
between 1.15x and 1.9x. Identical across three model seeds.
**(c)** ATSEs called per contrast at matched *realised* FDP 0.05 (each method thresholded where its own
null delivers 5%: LeafCutter q <= 1.4e-3, LeafletFA q <= 3.2e-2). Grey bars are counts at nominal
q <= 0.05, where LeafCutter's 5.7-fold apparent advantage is miscalibration rather than sensitivity.
Same test, same covariate; only the unit of analysis changes, so this is a data-design result.
**(d)** Where the two disagree. Called ATSEs binned by the effect size LeafCutter itself measures;
agreement rises monotonically with effect (19% both-called below TV 0.05, 61% above 0.30).
LeafletFA-only calls concentrate at m = 2, LeafCutter-only at m >= 6 where its null is most inflated.
**(e)** The posterior effect-size test, which uses the model's denoised program compositions:
P(TV > 0.15) with Bayesian FDR q <= 0.05, called in at least 2 of 3 independently seeded fits, restricted
to (contrast, ATSE) pairs with at least 25 covered cells in each arm (406,288 eligible). On
well-covered events the posterior test is conservative: half of its 7,041 calls are independently
made by LeafCutter, and LeafCutter calls 25,431 it does not.
**(f)** Both methods on the same question -- events whose LeafCutter-measured effect exceeds 0.15.
The posterior test recovers fewer of them (0.07 vs 0.22) but its calls are more often large-effect
by LeafCutter's own measurement (0.75 vs 0.56).
**(g)** Two events, junctions drawn as arcs over the annotated transcripts, then each junction's usage
in single cells (violin, one point per covered cell, bar = mean) with per-mouse pseudobulk values
overlaid. *Apbb2* (brain endothelial vs oligodendrocyte): a clean group-level switch both methods
call. *Gna13* (heart endothelial vs fibroblast; LeafletFA-only in this and four other tissues'
mesenchymal-vs-immune contrasts): every cell sits at 0 or 1 in both types, so pseudobulk means are
similar (0.42 vs 0.36; LeafCutter q = 0.63) while the cell-level structure the model fits is bimodal.
This is what pseudobulk erases; whether the between-type difference is real is at the resolution
limit of the test and is presented as an illustration, not a discovery.

*To supplement:* per-seed call counts and rank consistency (Spearman 0.90); Jaccard by m; the
ungated posterior counts and the coverage analysis behind the >= 25-cell gate; A63 at delta 0.10.
