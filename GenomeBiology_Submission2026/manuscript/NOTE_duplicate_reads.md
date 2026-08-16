# Note: PCR-duplicated reads in Smart-seq2, and why they do not break the model

Reference note, 2026-08-15. Not part of Methods; kept so the question "did you look at duplicate
reads?" has a written answer with numbers.

## The concern

Smart-seq2 libraries carry no unique molecular identifiers, so a cell's reads over an ATSE are PCR
copies of a small number of cDNA molecules rather than independent observations. Any likelihood that
treats N reads as N draws is overconfident about that cell's composition by roughly N / (number of
molecules). The previous model (BetaBinomial with a learned scalar concentration ~1.97) and any
plain Binomial did exactly this.

## What we measured, on the atlas, independent of any fit

Restrict to ATSEs whose POOLED composition within one cell type (microglia, 3,000 cells) is
genuinely mixed -- top junction usage between 0.2 and 0.8, so alternative splicing is definitely
occurring at the population level. Then ask how often a single cell puts ALL of its reads on one
junction:

| reads in the cell | fraction all on one junction | expected under independent reads |
|---|---|---|
| 5-8   | 0.632 | 0.054 |
| 10-18 | 0.532 | 0.005 |
| 20-48 | 0.434 | 0.0001 |
| 50-198 | 0.238 | ~0 |

At 50-198 reads, independence predicts all-on-one essentially never. It happens in a quarter of
cells. Across all mixed (cell, ATSE) pairs, 68% are all-on-one at a median of 48 reads.

Solving P(all on one) = p^M + (1-p)^M for the number of independent molecules M at p ~ 0.5:

| reads | P(all on one) | effective molecules M |
|---|---|---|
| ~6   | 0.632 | 1.66 |
| ~14  | 0.532 | 1.91 |
| ~34  | 0.434 | 2.20 |
| ~124 | 0.238 | 3.07 |

Reads rise 19x; effective molecules rise 2.5x. If each cell held a fixed number of molecules the
fraction would not move with depth at all, so molecule count does grow with depth -- just far more
slowly than read count. That is the duplication signature. It also means the effect is not simply
"a few cells are bad": it is the regime the whole atlas sits in.

## What it would do to a naive model

Treating ~48 reads as 48 draws makes the per-cell composition posterior ~20x too narrow (SE scales as
1/sqrt(n): sqrt(48/2) ~ 5x in SE, ~24x in precision). Every downstream quantity built on per-cell
confidence -- differential splicing p-values, credible intervals on psi, the "which program is this
cell in" question -- inherits that overconfidence. This is consistent with what the old model
showed: DS FDP 3-4x anti-conservative, and phi collapsing into simplex corners as the sparsity prior
was left to explain all-or-nothing counts as program commitment.

## How the shipped model handles it

Under the Dirichlet-multinomial with within-ATSE simplex psi, the Dirichlet's total pseudo-count
A_e = sum_j alpha_j = conc_e IS the effective sample size -- the number of independent draws the
model believes a cell contributes to that event. Fixing conc_e >= 2:

- 48 duplicated reads enter the likelihood as ~2 molecules of evidence, whatever the read count.
  The overconfidence is removed at the source rather than corrected downstream.
- The composition estimate stays informative: the composition of a duplicated molecule is still that
  molecule's composition. Duplication removes precision, not signal.
- It is per event and per cell -- not a global fudge -- because the correction is applied where the
  counts are.

The unconstrained fit put conc at a median of 0.52: fewer than one molecule per event, which is not
physically realisable. That was not the model discovering low overdispersion; it was the free
parameter falling into an impossible regime to explain all-or-nothing counts. The floor at 2 sits
inside the measured 1.7-3.1.

## What it cannot fix, stated

If a cell truly carries ~2 molecules over an event, no model recovers more than 2 molecules' worth
of information about that cell. Consequences, all of which shape the paper:
- phi per cell is noisy, and averaging cells does not rescue it (A60: cross-seed agreement is flat
  from n=1 to n=1000, so the disagreement is systematic). Programs are interpreted at cell-type
  level.
- Differential splicing is done between GROUPS of cells; the posterior effect-size test cannot
  resolve effects below the psi posterior sd (~0.02-0.09 per junction), which is why delta = 0.15.
- Imputation is scored on cells the model never saw, at controlled downsampling.
Duplication is a ceiling on per-cell resolution, not a bias in the estimates.

## The obvious alternative, and why not

Deduplicate at the read level (position-based, e.g. samtools markdup). No UMIs means two genuinely
independent fragments landing at the same coordinates are indistinguishable from a PCR pair, so
position dedup over-removes -- and for a ~50 bp junction-spanning read the coordinate space is small
enough that this is common. Modelling the effective sample size treats the whole read distribution
consistently, and the number is measured from the data rather than assumed.

## Two limits to keep attached

- The molecule estimate assumes p ~ 0.5 on the events used to derive it, so it fixes an order of
  magnitude, not a decimal. `conc >= 2` is defensible; `conc = 2.0 exactly` is not.
- The all-or-nothing signature does not by itself separate technical duplication from genuine
  single-cell isoform commitment (a cell really making one isoform). Under duplication a low conc is
  correct and phi should not be asked to explain the excess; under commitment the switching is real.
  The floor stands either way. `A58_commitment_structure.py` tests which, and found the structure
  is donor-level rather than cleanly technical or cell-level -- unresolved, and stated as such.

Sources: `LeafletFA/METHODS_dirichlet_multinomial.md` §"effective sample size";
`pipelines/05_leafletfa/A58_commitment_structure.py`; `A60_phi_aggregation_curve.py`.
