# Figure 3 caption — draft 2026-08-16 (main = 4 panels; the rest in the supplement)

**Figure 3. Differential splicing on the atlas: pseudobulk LeafCutter against single-cell LeafletFA.**
Seventy pairwise cell-type contrasts within tissues from Tabula Muris Senis. LeafCutter was run on
pseudobulk replicates (cells aggregated per mouse, 7 vs 8 per contrast), the realistic design for
Smart-seq2 differential splicing; LeafletFA's likelihood-ratio test uses the same Dirichlet-multinomial
GLM on every individual cell.

**(a)** Realised over nominal false-discovery proportion at α = 0.05 as a function of junctions per
ATSE, each method on its own permutation null. Pseudobulk LeafCutter degrades from 3.2× at m = 2 to
11.4× at m = 9; single-cell LeafletFA stays between 1.15× and 1.9×. Identical across three model seeds.
**(b)** ATSEs called per contrast at matched *realised* FDP 0.05 (LeafCutter q ≤ 1.4e-3, LeafletFA
q ≤ 3.2e-2). Grey bars are counts at nominal q ≤ 0.05, where LeafCutter's 5.7-fold apparent advantage
is miscalibration rather than sensitivity. Same test, same covariate; only the unit of analysis
changes.
**(c)** The posterior effect-size test on the model's program compositions: P(TV > 0.15), Bayesian FDR
q ≤ 0.05, called in ≥ 2 of 3 seeded fits, restricted to (contrast, ATSE) pairs with ≥ 25 covered cells
in each arm (406,288 eligible). Half of its 7,041 calls are also made by LeafCutter. The LeafletFA-only
bar is split: 2,191 calls where the two cell types' pooled observed usage itself differs by TV ≥ 0.15
(effects the 7-vs-8 pseudobulk design lacked power for) and 1,299 (hatched) where it does not — the
programs the two types load differ at the event, so the model reports a program-level difference,
typically the right direction at an amplified magnitude (Gna13 ATSE1: posterior 0.33–0.41, cells 0.09,
atlas-wide endothelial-vs-mesenchymal 0.15). Reported as program-level inference, not merged.
**(d)** Two events from one contrast (brain endothelial vs oligodendrocyte): junction arcs over the
annotated transcripts, then each junction's usage in single cells (violin, one point per covered cell,
bar = mean) with per-mouse pseudobulks overlaid. *Apbb2*, found by both (LeafCutter q = 3e-18;
posterior TV 0.64–0.76; observed 0.98). *Mbd3*, LeafletFA-only: 238 and 414 covered cells shift J1 from
0.19 to 0.37 (observed TV 0.35; posterior 0.25–0.28 in all seeds), but the 7 and 8 per-mouse pseudobulks
overlap and LeafCutter reaches q = 0.20. Examples gated on ≥ 50 covered cells per arm and observed
TV ≥ 0.25.

K = 20 throughout; the posterior test's call set at K = 10 / 30 overlaps K = 20 at Jaccard 0.52 / 0.71
(supplement); the LRT and LeafCutter arms do not depend on K.

*Supplement (`FIG3_supp_calibration_and_K`):* null QQ on each method's own null; Jaccard by
LeafCutter-measured effect; recall/precision on LeafCutter's own large-effect set (0.07 vs 0.22 recall,
0.75 vs 0.56 precision); posterior calls and LeafCutter agreement at K = 10/20/30.
*Supplement (`FIG3_supp_program_sharing`, `FIG3_supp_effect_sizes_by_contrast`):* the posterior TV
ceiling per contrast is 0.34‖Δφ̄‖₁ − 0.04 (r 0.95; bound ½‖Δφ̄‖₁ because ψ sums to 1 within an ATSE);
29 of 70 contrasts have no posterior call at any ATSE; mean program activity per cell type (SP1 = largest
π); per-ATSE posterior TV vs LeafCutter TV vs observed pooled TV in six contrasts.
*Supplement:* per-seed call counts and rank consistency (Spearman 0.90); Jaccard by m; the ungated
counts and the coverage analysis behind the ≥ 25-cell gate; A63 at δ 0.10; A65 by call category; the
Fbln2 program decomposition (SP-labelled).
