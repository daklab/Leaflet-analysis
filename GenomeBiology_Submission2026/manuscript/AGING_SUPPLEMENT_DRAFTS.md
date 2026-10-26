# Aging supplement: evidence, mouse-level test, and two drafts for Karin's choice

Written 2026-10-25. The test was pre-registered in `AGING_CLAIM_OPTIONS.md` §6 (commit 93e16cb, before any
computation).
- Code: `figures/FIGURE4_aging/audit/aging_mouse_level_test.py`.
- Outputs: `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/aging_mouse_level/`.

**Until Karin chooses, the program is not called aging-related anywhere outside this decision.** The
placeholders in `methods_CURRENT.tex`, `CAPTION_FIG4.md` and the results draft are marked PENDING.

## 1. Existing evidence that age is confounded with mouse and batch

| Source | Test | Result |
|---|---|---|
| Leaflet-analysis 4b98dae, `figures/FIGURE4_aging/audit/age_confound_audit.tsv` | Crosstab of age with plate, campaign, sex, parsed from TMS cell IDs | **0 of 510 plates contain cells of 2 ages.** 3 m = campaign C1 only; 18 m and 24 m are later formats (C2 / C3). All 24 m mice male. Males 3 m vs 24 m: 0 of 257 plates shared. |
| STAGE5 §27 / §36.3 (SpliceVI audit) | 18 m vs 24 m within the later campaign; age classifier on latents; grouped CV; exact mouse-label permutation (70 assignments) | Splicing latents F1 0.65–0.66 vs null 0.49, p 0.029 (2/70). **Technical covariates alone also separate the ages: F1 0.630, p 0.029.** Detected genes differ (1,458 vs 1,278). Separable, but not shown to be biology. |
| STAGE5 §91 | Processing metadata within the later campaign | Every 18 m mouse is in the secondary-plate format; every 24 m mouse is lane-demultiplexed. Plates are nested in mice, so age is confounded with processing batch as well as sex. |
| STAGE5 §88b | Per-mouse state-matched splicing, relabelling of age across mice | Real labels rank 1 of 70 (p 0.014), sex-conditioned 1 of 15. But random same-size mouse groupings already reach +0.31 of the reproducibility gain (real excess only +0.17–0.23 above that): **most of the signal is mouse-level structure.** |
| STAGE5 §89.2 | Same-age mouse excess: kNN mouse purity among cells of the same age, beyond the mixing expectation | 0.16–0.30 across latents: mice of the same age are strongly distinguishable. |
| STAGE5 §94 | Scope decision | Aging dropped as a SpliceVI evaluation target; all aging results recorded as "age or processing". |

**Mice per age (TMS):**
- 3 m: 8 mice (4 F, 4 M) plus two pooled-cell labels (`3_10_M/3_11_M`, 45 cells; `3_38_F/3_39_F`).
- 18 m: 4 (2 F, 2 M).
- 24 m: 4 (all M).
- The male 3 m vs 24 m comparison is therefore 4 vs 4 mice. A69 counted the pooled label as a fifth mouse.

## 2. Mouse-level test of the LeafletFA program (pre-registered)

Atlas fits K20, seeds 0–2; the program A69 selected in each seed (seed 0 SP4, seed 1 SP3, seed 2 SP8).
Unit = mouse × tissue–cell type (≥ 20 cells).

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| P1 mixed model, males 3 m vs 24 m: age effect [95 % CI] | +0.121 [0.098, 0.145] | +0.120 [0.096, 0.144] | +0.153 [0.122, 0.185] |
| between-mouse SD within age | 0.014 | 0.015 | 0.020 |
| ratio effect / between-mouse SD | 8.5 | 8.2 | 7.5 |
| P2 exact permutation, 5 vs 4 (as pre-registered) | p = 1/126 = 0.008 | 0.008 | 0.008 |
| P2 sensitivity, 4 vs 4 real mice | p = 0.029 (minimum) | 0.029 | 0.029 |
| S1 later campaign, males 18 m vs 24 m (2 vs 4) | +0.040, p 0.067 (minimum) | +0.088, 0.067 | +0.100, 0.067 |
| S2 rank of this program among live programs (ratio) | 6 of 8 | 3 of 6 | 7 of 11 |
| S2 live programs with ratio > 2 | **7 of 8** | **5 of 6** | **10 of 11** |

**Reading (pre-registered rules):**
- **R1, mouse-level group difference: met.** Complete separation of 3 m and 24 m mice, the effect ~8× the
  between-mouse SD, the same direction in all seeds.
- **R2, specific to this program: NOT met.** In every seed most live programs separate the two age groups
  by more than 2× the between-mouse SD, and this program ranks 3rd–7th. The age groups differ globally,
  across the whole program basis. That is the pattern a sequencing-campaign difference produces, and it
  coincides with age by design.
- **R3, later campaign direction: met** (18 m < 24 m in all seeds), but those ages differ in processing
  format (§91).

**Recommendation by the pre-registered rule: option (b).** Option (a) is drafted as asked.

## 3. Draft (a): keep as an age-associated trend with an explicit batch caveat

**Methods** (replaces the subsection "The Aging Splicing Program"):

> \subsection*{Age-associated variation in program activity}
> Tabula Muris Senis provides cells from mice aged 3, 18 and 24 months; the Allen Brain data (all 2-month
> animals from one laboratory) were excluded. In this atlas age cannot be separated from sequencing
> batch: the 3-month animals were sequenced in a different campaign from the 18- and 24-month animals, no
> sequencing plate contains cells from more than one age, the 18- and 24-month animals were processed in
> different library formats, and all 24-month animals are male. We therefore asked only whether age groups
> differ by more than mice of the same age differ from each other, at the level of the mouse. For each
> program, activity was averaged per mouse and tissue--cell type (at least 20 cells) and modelled with a
> linear mixed model (fixed effects: age group and tissue--cell type; random intercept: mouse) on males at
> 3 and 24 months (4 and 4 mice), with an exact permutation of age labels across mice. In each of the three
> seeded fits one program's activity was higher in every 24-month mouse than in every 3-month mouse
> (difference 0.09--0.12 in mean activity; exact $p = 0.029$, the minimum attainable for four versus four
> mice) and rose with age in 94--99\% of tissue--cell types. The same comparison, however, separated the
> age groups by more than twice the between-mouse standard deviation for most programs (5--10 of the 6--11
> programs in use per fit), so the shift is not specific to one program and is consistent with a batch
> difference between sequencing campaigns. The program's composition also reproduced only partially
> across seeds (centred cosine 0.65; Program Reproducibility). We report it as an age-associated trend
> that may reflect sequencing batch, not as an aging program.

**Supplementary Figure SX caption:**

> **Supplementary Figure SX. Program activity differs between age groups in Tabula Muris Senis, but age is
> confounded with sequencing batch.**
> - **(a)** Variance partition of each program's activity (Type-II ANOVA: tissue–cell type, sex, age and
>   their interactions); cell identity dominates every program.
> - **(b)** Activity of the program with the broadest age trend, per cell and per mouse (circles male,
>   triangles female); every 24-month male mouse is above every 3-month male mouse (exact p = 0.029, 4 vs 4
>   mice).
> - **(c)** The same mouse-level age effect, divided by the between-mouse standard deviation, for every
>   program in use: most programs separate the age groups, as expected when age coincides with sequencing
>   campaign.
> - **(d)** Junctions where the program departs most from the other programs (seed-0 program; its
>   composition reproduces only partially across seeds, centred cosine 0.65).
> - 3-month and 18/24-month animals were sequenced in different campaigns, no plate spans two ages, and
>   all 24-month animals are male; these panels cannot separate age from batch.

Panel changes vs the current draft:
- (b) is kept, with 4 vs 4 mice.
- (c) is new: the specificity plot from `aging_mouse_level_specificity.tsv`.
- (d) is the old panel d.
- The old per-cell-type delta heatmap (old c) and the *Kcnip3* example (old e/f) are removed. Kcnip3's age
  trend has the same batch confound, and the example was selected from this program.

**Results** (one sentence, near the program-reproducibility results):

> Program activity also differed between the 3- and 24-month animals of Tabula Muris Senis (Supplementary
> Fig. SX), but because age coincides with sequencing campaign in this atlas and most programs shift
> between the age groups, we do not interpret this as an aging signal.

## 4. Draft (b): remove the aging claim; keep the program only as a partially reproducible example

**Methods:**
- The subsection "The Aging Splicing Program" is deleted.
- Add to the Data section:

> The ages of Tabula Muris Senis animals (3, 18 and 24 months) coincide with sequencing campaign and
> library format (no sequencing plate contains cells from more than one age), so age effects cannot be
> separated from batch in this atlas and are not analysed.

- Add at the end of "Program Reproducibility":

> As an example of a program that does not meet this threshold, a program used mainly by endothelial and
> smooth-muscle cells is found in all three seeded fits but its composition agrees only partly between
> them (centred cosine 0.65; Supplementary Fig. SX); statements about it are therefore restricted to what
> the three fits share.

**Supplementary Figure SX caption:**

> **Supplementary Figure SX. A partially reproducible program.**
> - **(a)** Mean activity per tissue–cell type of the matched program in each of the three seeded fits.
> - **(b)** Centred-ψ cosine between the fits for this program, against the 0.8 threshold and the
>   reproduced programs.
> - **(c)** Its top junctions in each seed, showing which are shared and which are seed-specific.

Panel changes:
- The aging panels (old b, c, e, f) and the age terms of the variance partition are removed.
- New panels a–c are built from the existing fits; no training is needed.

**Results** (one sentence):

> Programs that do not meet the reproducibility threshold can still be found in every fit with partly
> different compositions (Supplementary Fig. SX); we restrict program-level statements to the reproduced
> programs.

## 5. Text that must change under either option (not yet edited; PENDING markers placed)

- `methods_CURRENT.tex`, subsection "The Aging Splicing Program" (L141–142): replaced by (a) or by (b).
- `figures/FIGURE4_aging/CAPTION_FIG4.md`: replaced by the (a) or (b) caption.
- `manuscript/RESULTS_DRAFT_Fig4_Fig5_bullets.md`, aging block: replaced by the (a) or (b) sentence.
- `figures/FIGURE_PLAN_FINAL.md` L72–74 ("Figure 4 — Aging … Program 5 rises with age"): planning
  document, marked PENDING.
- `manuscript/LEAFLETFA_FOR_DUMMIES.md` L53 ("program 5 is an aging program"): explainer, marked PENDING.
- No abstract, introduction or discussion file exists yet in `manuscript/`. When they are written, the
  program must not be described as aging-related unless (a) is chosen, and then only with the batch caveat.
