# Program-stability wording — draft for Karin's choice (not applied)

Written 2026-10-27, from LeafletFA validation check 6 (pre-registered as amendment 3 in
`LeafletFA_validation/VALIDATION_PREREG.md`; results in `VALIDATION_REPORT.md`, "Check 6"). Nothing in
`methods_CURRENT.tex` or the Results drafts has been changed by this file.

## What check 6 found

TMS and mouse Allen fits, K 20, seeds 0–2, live programs. For every ordered seed pair, each program is
classified, checking in this order:
1. one-to-one: centered cosine ≥ 0.8;
2. up to splitting: centered ψ explained at R² ≥ 0.8 by ≤ 3 programs of the other fit;
3. not reproduced.

| | reproduced fraction per ordered pair | reading |
|---|---|---|
| TMS (7 live) | 0.57 in all 6 pairs; all one-to-one, no program rescued by splitting | ≥ 0.5 |
| Allen (3–4 live) | 0.75, 0.50, 0.50, **0.25**, **0.33**, 1.00 | below 0.5 in 2 pairs |

- **The pre-registered rule says "reproducible up to resolution" may not be used.**
- **Splits:** the only splits are Allen neuron programs. A mixed excitatory and inhibitory program in one
  seed is split into separate excitatory and inhibitory programs in the others (R² 0.81–0.86).
- **Usage reproduces, content doesn't.** Programs that do not reproduce still have a counterpart used by
  the same cell types: activity-profile r is 0.81 (TMS) and 0.92 (Allen). But they share almost none of
  their top marker junctions (Jaccard 0.19 and 0.02).
- **The supplementary example shows the same thing.** The endothelial program has a counterpart used by
  the same cells in every fit, at centered cosine 0.54–0.65.

## Option 1 (recommended; follows the pre-registered reading): current wording, sharpened

**Methods, Program Reproducibility** (append after the definition of reproduced programs):

> We also asked whether programs that fail this criterion are recovered at a different resolution, as
> splits or merges of programs in the other fits (non-negative least-squares regression of a program's
> centered composition on up to three programs of another fit, $R^2 \geq 0.8$). This rescued no program in
> Tabula Muris Senis and only Allen Brain neuron programs, where one fit's mixed excitatory and inhibitory
> program was split into separate programs in the others. Programs that were not reproduced usually had a
> counterpart used by the same cell types in each fit (median correlation of cell-type activity profiles
> 0.81--0.92) but with a different junction composition (median overlap of the top 50 junctions 2--19\%).
> Cell-type usage of programs is therefore reproducible, while their junction composition is reproducible
> only for the programs that meet the criterion above.

**Results** (one sentence where program reproducibility is reported):

> Which cell types use which program was reproducible across seeds, but for most programs the junction
> composition was not; we therefore make program-level statements only for the programs whose composition
> reproduced (four on the atlas), and otherwise rely on the per-cell representation and on ATSE-level tests.

## Option 2 (not supported by the pre-registered reading): "programs are reproducible up to resolution"

> Programs are reproducible up to resolution: programs that do not match one-to-one across seeds are
> recovered as splits or merges of programs in the other fits.

Why not:
- **Allen:** only 25–33 % of live programs are reproduced in two of the six seed pairs.
- **TMS:** splitting rescues no program at all.
- **What the sentence would claim versus what holds:** it would claim recovery of junction content, but
  what is reproducible is cell-type usage. A reviewer checking the supplementary example would see one fit
  whose closest-composition program is used by different cells.

## Where the chosen text goes

- `manuscript/methods/methods_CURRENT.tex`, "Program Reproducibility" (after the definition, before the
  partially reproducible example).
- `manuscript/RESULTS_DRAFT_Fig4_Fig5_bullets.md`, the program-replication bullet.
- `LeafletFA/METHODS_LEAFLETFA_CURRENT.md` §8 (one sentence, for consistency).
