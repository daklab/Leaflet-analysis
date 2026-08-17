# Results-text notes: how TV is defined, and how imputation is described (2026-08-17)

## TV, said once and clearly (use in Results the first time the posterior test appears)

For every ATSE and every posterior draw, LeafletFA computes each cell's expected junction usage
rho_cj = sum_k phi_ck psi_kj, averages it over the cells of each group, and takes the difference per
junction, Delta_j. The effect size is the total variation TV_e = 1/2 sum_{j in e} |Delta_j|: the
fraction of the ATSE's splicing mass that moves between the two groups. The Delta_j sum to zero
within an ATSE (mass leaving one junction arrives at the others), so the 1/2 counts each moved unit
once. For two junctions TV = |dPSI|; for a cassette exon it is |dPSI_skip| when the two inclusion
junctions move together.

Worked example (K = 2, three junctions J1 J2 J3): program 1 = (0.40, 0.40, 0.20), program 2 =
(0.25, 0.25, 0.50); group A mean phi = (0.8, 0.2), group B = (0.3, 0.7), so dphi = (0.5, -0.5).
Delta_J1 = Delta_J2 = 0.075, Delta_J3 = -0.15 (they sum to 0). TV = 1/2 (0.075 + 0.075 + 0.15) =
0.15: 15% of the mass moved from the skip junction to the inclusion junctions. Ceiling for this
pair of groups: 1/2 ||dphi||_1 = 0.5; this ATSE reaches 0.15 because the two programs differ only
moderately at it.

Calling: P(TV_e > delta) over 300 draws with delta = 0.15 (derived from permuted negative-control
ATSEs), Bayesian FDR q <= 0.05, >= 2 of 3 seeds, >= 25 covered cells per arm.

## Imputation, said once and clearly

LeafletFA denoises junction usage at single-cell resolution by pooling information across all cells
(through psi) and all ATSEs (through phi): each cell's usage is estimated from its program membership,
shrunk toward its own reads where they exist, and reported with a posterior uncertainty. Because a
single cell's observed usage at one ATSE rests on ~2 effective molecules, per-cell estimates cannot be
scored against per-cell observations -- the observation, not the estimate, sets that floor (a
cell-type mean that uses the labels scores 0.04 on the same entries) -- so per-cell estimates are
validated where the noise averages out: per cell type and per group, which is also where the
differential-splicing test and program analyses use them. What the held-out 30% shows: (i) the
learned programs transfer -- E-step phi of unseen cells reproduces the training cell-type structure
(Pearson 0.997 over type x program); (ii) [A66 cell-type-level within-junction r, pending];
(iii) with the cell's own reads the posterior mean holds within-junction r 0.95 / 0.90 / 0.66 at
10 / 50 / 90% of reads removed (junction mean + same reads 0.86 / 0.80 / 0.51; NMF 0.10 / 0.09 /
0.07); (iv) posterior sd tracks across-seed variation (A68, pending).
