# Figure plan: differential splicing, LeafletFA vs LeafCutter

Two tests, one comparison. Written before the results land so the panels are chosen by what would
be informative either way, not by what the numbers turn out to be.

Inputs, all produced overnight 2026-08-05/06:

| file | from | contents |
|---|---|---|
| `LEAFCUTTER_CMP/a49_all_phi.tsv` | A49 `--design phi` | LRT across learned programs, 73 contrasts |
| `LEAFCUTTER_CMP/a49_all_group.tsv` | A49 `--design group` | LRT with cell-type indicator (control) |
| `LEAFCUTTER_CMP/a51/a51_group_ds_posterior.tsv` | A51 | posterior effect-size test, 73 contrasts |
| `LEAFCUTTER_DS_2026-08/by_pair/*/ds_results_cluster_significance.txt` | LeafCutter | loglr, df, p, p.adjust |
| `LEAFCUTTER_DS_2026-08/permutation_null/*/` | LeafCutter | 60 clean null runs |
| `ATLAS_LRT/a45_lrt_atlas20000_seed0.tsv` | A45 | LeafletFA permuted-phi null |
| `DM_FRESH_LRT/a45_lrt_*.tsv` | A45 | simulation validation, known truth |

---

## Panel A — calibration on each method's own null

QQ of null p-values against Uniform, log-log so the tail is visible. One line per method.

Reads as: distance above the diagonal is how much a method over-calls. Include the diagonal and
mark alpha = 0.05 and 0.01.

**Must be annotated with the asymmetry**, or the panel is misleading: LeafCutter's null is clean
(labels shuffled within a cell type), LeafletFA's is contaminated by real signal and is therefore an
upper bound. Put this in the caption, not a footnote.

## Panel B — calibration by event size

Ratio at alpha=0.05 against m, two lines. This is the sharper result: LeafCutter degrades 3.23 -> 11.35
from m=2 to m=9 while LeafletFA is flat near 3. Annotate the sample sizes (7v8 pseudobulk vs
218-2,870 cells), since that is the explanation and it is not a modelling claim.

## Panel C — realised FDP vs nominal, on simulation

The only place truth exists. Three lines: the retired fixed-delta rule (2.9x over), the LRT, and the
posterior effect-size test. Diagonal for reference.

This panel carries the methods argument: the old rule was anti-conservative, both new tests are at
or below nominal.

## Panel D — power at MATCHED realised FDP

Not at matched nominal q. A miscalibrated test calls more by construction, so comparing call counts
at nominal 0.05 would score LeafCutter's 4.48x inflation as sensitivity. Threshold each method where
its OWN null says the realised FDP is 0.05, then compare counts. `A50.q_at_realised_fdp` does this.

Expect this to be the panel a referee goes to first.

## Panel E — agreement on real contrasts

Scatter of LeafletFA `loglr` against LeafCutter `loglr` per ATSE, one facet per contrast or pooled
with density. Report Spearman. Add the Jaccard of the calibrated call sets as an inset number.

Overlap alone is uninterpretable when the two call very different numbers of events, so always show
the counts beside it.

## Panel F — what the programs buy

The `--design phi` arm against the `--design group` arm, same events. Asks whether DS across
label-free programs recovers what labelled contrasts find. This is the panel that speaks to
LeafletFA's actual claim rather than to test mechanics.

If phi recovers most of the group-design calls, the claim is "no labels needed". If it finds a
distinct set, the more interesting claim is that programs capture structure the cell-type
partition misses -- check those events against `broad_cell_type` composition before asserting it.

## Panel G — the two LeafletFA tests against each other

LRT calls vs posterior-test calls on the same contrasts. They test different hypotheses (Delta != 0
against TV > 0.05), so disagreement is expected and informative: events called by the LRT but not
the posterior test are statistically detectable but small.

Use this to state the practical guidance rather than to declare a winner.

---

## Numbers to quote, with their caveats attached

- LRT validation: FDP at or below nominal, power 0.84-0.99, 4 effect sizes x 3 seeds.
- Posterior test validation: FDP ratio 0.00 / 0.35 / 0.31, power 0.253 / 0.717 / 0.850 at effect
  0.25 / 0.5 / 1.0, **scored against contrast-specific truth** (scoring against "differential across
  any program pair" understates power ~1.5x).
- Retired rule: 2.9x over nominal at alpha=0.05. Cite as the motivation, not as a current result.
- Calibration head-to-head: quote the DIRECTION and the m-dependence confidently; do NOT quote the
  exact ratios as matched until both methods are run on one null construction.

## Do not plot

- `% of saturated deviance` (18.2%) as a model-quality measure. The saturated model fits each cell's
  shot noise and 8.35M entries have 1-2 reads. Use the cell-type-pooled reference instead, where the
  model captures 98% of what 38 curated labels buy without using any.
- The atlas ELBO curve as evidence of convergence. Its whole 11.25% drop lands in the first ~50 steps
  and the tail slope flips sign between fits. Use the `psi_contrast` trajectory instead.
