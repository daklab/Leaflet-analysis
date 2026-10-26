# The aging-program replication claim: where it is made, the conflicting result, and two options

> **DECIDED 2026-10-25 (Karin): option (b)** — soften to partial replication (centred cosine 0.65) and
> move to the Supplement; program stability handled by (b) per-cell/ATSE claims + (a) consensus programs
> for the four reproduced atlas programs. Text drafted on this branch: Methods (Program Reproducibility,
> aging paragraph), `figures/FIGURE4_aging/CAPTION_FIG4.md` (now Supplementary Figure SX),
> `RESULTS_DRAFT_Fig4_Fig5_bullets.md`, `RESULTS_DRAFT_Fig2_Fig3_bullets.md`, `CAPTION_FIG3.md`.
> **Not yet done (figure work, needs Karin):** the new main Figure 4 (programs across cell types with the
> four reproduced programs marked, from the existing SP-by-cell-type heatmaps and panel a); renumbering
> supplementary figures; the age-confound caveat (4b98dae) beside the supplementary panels.

Written 2026-10-25 for Karin's decision. Nothing in Figure 4, its caption or the aging text has been
edited. Line numbers refer to branch `leafletfa-methods-fixes`.

## 1. Where replication is claimed

| Location | Claim |
|---|---|
| `manuscript/methods/methods_CURRENT.tex` L142 | "One program satisfies both criteria in all three seeded fits (activity increasing in 94–99% of cell types; **matched across seeds by ψ at r = 0.95–0.97**)." |
| `methods_CURRENT.tex` L131 | Definition used for that number: Hungarian matching "on the absolute correlation of their compositions ψ"; "reproduced when its matched correlation is at least 0.9 in every other seed". This is raw ψ, not centred. |
| `figures/FIGURE4_aging/CAPTION_FIG4.md` L3 (title) | "**A single splicing program rises with age across the mouse atlas.**" |
| `CAPTION_FIG4.md` L13–15, panel b | "activity rises with age in 94–99% of tissue-cell-types in all three seeded fits (**the program is matched across seeds by ψ at r = 0.95–0.97**)" |
| `CAPTION_FIG4.md` L10, panel a | "the aging program carries the largest age share" (one program, named across fits) |
| `CAPTION_FIG4.md` L16–25, panels c–f | (c) shift per cell type, (d) the program's junctions, (e, f) *Kcnip3* example screened from the program's top-200 junctions. They depend on the program being one reproducible entity. |
| `manuscript/RESULTS_DRAFT_Fig4_Fig5_bullets.md` L17–18 | "psi-matched across seeds r 0.95-0.97" |
| `RESULTS_DRAFT_Fig4_Fig5_bullets.md` L10 | "Program replication across seeds (A61, r >= 0.9 on psi): K20 11-14/20 on every dataset" |

Panels affected: Fig 4 a (the aging row and its label), b, c, d, e, f.

## 2. The conflicting result (A70, 2026-09-27)

- **Script:** `pipelines/05_leafletfa/A70_program_stability_centred.py`, commit 7971974.
- **Output:** `Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/stability/a70_program_stability_centred.tsv`
  and `a70_summary.tsv`.
- **Notes:** `manuscript/RESULTS_NOTES_2026-08-17_full.md` L131–142; consequences per panel in
  `manuscript/PROPOSAL_program_stability_and_consensus.md` (commit 5c1f761).
- **Method:**
  - Each seed's ψ is centred by its π-weighted mean composition before Hungarian matching on cosine.
  - Raw ψ similarity is ~0.98 for *every* program, because all programs share the bulk of each
    junction's usage. The raw r = 0.95–0.97 therefore cannot distinguish a reproducible program from a
    non-reproducible one; the centred cosine measures the program-specific part.
- **Result for the atlas K = 20 aging program** (seed-0 SP4, endothelial / smooth muscle):
  - centred cosine **0.645** (seed 1) and **0.650** (seed 2), min **0.65**;
  - fails the ≥ 0.8 rule.
- **The rule's survivors at atlas K = 20:** 4 of 20 programs — SP5 hippocampal/excitatory (0.80),
  SP7 inhibitory (0.83), SP8 cortical excitatory (0.85), SP15 keratinocyte/basal (0.82).
- **What does replicate:** the aging *behaviour* — p at the floor, activity rising in 94–99% of cell
  types — holds in all three seeds. The program's *composition* agrees only partly.
- **Second caveat:** the proposal file also marks the program "age-confounded", from the commit 4b98dae
  confound audit (plate / lane / campaign / sex per panel).

## 3. Option (a): cut Fig 4b–f

Remove:
- panels b–f and their caption text;
- `methods_CURRENT.tex` L142 from "One program satisfies…" onward;
- the aging bullets in the results draft.

Panel a stays as a basis description, with:
- the cell-type and sex columns;
- program-specific rows only for the stable SP5, SP7, SP8 and SP15, per the proposal's table.

Consequences:
- **Fig 4's headline changes.** The title "A single splicing program rises with age" goes. Fig 4 becomes
  "Splicing programs across cell types" (panel a plus the current supplementary panels).
- **Aging leaves the paper,** unless it is restated at the junction level, without a program, in the
  supplement.
- **Gap a reviewer may ask about:** the previous submission had an aging result. The cover letter or
  Discussion would need to say why it was removed (program instability).
- **Least risk** of a reviewer finding an overclaim.

## 4. Option (b): soften to partial replication, move to the supplement

Move b–f to a supplementary figure. Reword:
- **Title:** "An aging-associated splicing program (partially reproducible across seeds)".
- **Panel b caption:** "the aging trend (p = 0.016 per mouse, males; rising in 94–99% of cell types)
  holds in all three seeded fits; the program's specific composition agrees only partly across seeds
  (centred-ψ cosine 0.65; the stability threshold used for program-specific claims is 0.8)".
- **Methods L142:** replace "matched across seeds by ψ at r = 0.95–0.97" with the centred measure. State
  the 0.8 rule and that the aging program does not meet it.
- **Methods L131:** add the centred-ψ definition, and say that raw ψ correlation (~0.98 for every program)
  is not used to call a program reproducible.
- **Panels c–f:** state that they describe the seed-0 program. *Kcnip3* is a junction-level observation
  (PSI 0.35 → 0.03 with age, expression flat) that stands without the program.
- **Results draft L10:** "11–14/20 programs replicate" must be redefined on centred ψ: atlas K20 is 4/20.

Consequences:
- **The aging result is kept, honestly qualified;** aging becomes an application in the supplement.
  This matches the method-plus-resource framing.
- **Main Fig 4** keeps panel a (with stable programs marked) plus the current supplementary programs
  panels.
- **The age-confound caveat** from 4b98dae should be stated beside it.
- **More text to get right,** and a reviewer may still question a non-reproducible program being shown at
  all; the centred number must be in the caption, not only in Methods.

## 5. Either way

- **Replace the replication definition (Methods L131):** raw-ψ correlation ≥ 0.9 flags almost every
  program and should not be the stated definition. Use the centred ψ.
- **Keep the strongest true claims:** the 4 stable atlas programs and the conserved mouse–human pair
  (centred min cosine 0.97 / 0.98, which passes) can still be stated as reproducible.
- **The LeafletFA validation check 3** (seed stability of the new SpliceVI-split fits) now uses the
  centred metric as primary (pre-registration amendment 2), so its numbers will be comparable with A70.

## 6. PRE-REGISTERED (2026-10-25, before any computation): mouse-level test of the aging program

**Design fact (not a result):** in TMS, age is perfectly confounded with sequencing campaign and
processing.
- 3 m cells are campaign C1; 18 m and 24 m cells are a later campaign with different processing formats
  (every 18 m mouse secondary-plate format, every 24 m mouse lane-demultiplexed; STAGE5 §27, §91;
  `figures/FIGURE4_aging/audit/age_confound_audit.tsv`).
- No plate spans two ages, and all 24 m mice are male.
- So no test here can attribute a difference to age rather than batch. The test can only say whether the
  age groups differ by more than mice of the same age differ from each other, and whether that
  difference is specific to this program.

**Data:**
- Atlas fits `runs_final_deltaonly` K20, seeds 0–2 (the paper's fits); TMS cells only.
- The aging program per seed is A69's selection: seed 0 program index 6 (SP4), seed 1 index 2 (SP3),
  seed 2 index 9 (SP8).
- Unit: mouse × tissue–cell type, mean φ of the program over the unit's cells (≥ 20 cells).

**P1 (mixed model):** males, 3 m (5 mice) vs 24 m (4 mice).
- Model: `unit_mean ~ age24 + C(tissue_cell_type) + (1 | mouse)` (statsmodels MixedLM, REML).
- Report the age coefficient, its 95 % CI, the between-mouse SD within age (random-intercept SD), and
  the ratio |age coefficient| / between-mouse SD.

**P2 (exact mouse-level permutation):**
- Unit means are residualised on tissue–cell type (type effects fitted on all male units), then
  averaged per mouse, giving one value per mouse.
- Observed statistic: the 24 m − 3 m difference of mouse means.
- Exact two-sided p over all C(9, 4) = 126 assignments of the "24 m" label to 4 of the 9 male mice.
  The minimum attainable p is 2/126 = 0.016.

**S1 (within the later campaign, descriptive):** 18 m vs 24 m.
- Males only: 2 vs 4 mice, 15 assignments, minimum p 0.13.
- All 8 mice with sex as a covariate in the mixed model.
- Direction only; it cannot reach significance, and processing still differs between these ages (§91).

**S2 (specificity):** the P1 statistic (|age coefficient| / between-mouse SD) for every live program
(argmax ≥ 1 % of TMS cells); the aging program's rank; and the share of live programs with a ratio > 2.

**Reading, fixed now:**
- **R1 — a mouse-level group difference exists:** P2 p ≤ 0.016 (its minimum) AND ratio > 2 AND the
  same direction, in all 3 seeds.
- **R2 — specific to this program:** the aging program ranks first among live programs by the ratio in
  ≥ 2 of 3 seeds AND fewer than half of the live programs have a ratio > 2. If most programs separate the
  age groups, the difference is a global group (campaign) shift, not an aging program.
- **R3 — later-campaign direction:** 18 m < 24 m in all 3 seeds (descriptive).
- **Recommendation rule:**
  - Option (a), a caveated aging-associated trend, only if R1 and R2 hold and R3 agrees.
  - Otherwise option (b): remove the aging claim and keep the program only as an example of a partially
    reproducible program.
  - In both drafts the batch confound is stated explicitly.

## 7. Result of the §6 test (2026-10-25) — see AGING_SUPPLEMENT_DRAFTS.md for the full tables and both drafts
R1 met (complete separation of 3 m and 24 m mice in all 3 seeds; effect ~8× the between-mouse SD); **R2 not
met** (5–10 of 6–11 live programs separate the age groups at > 2× the between-mouse SD; this program ranks
3rd–7th); R3 met but processing-confounded. Recommendation by the rule: **option (b)**. Deviation: A69's
"fifth" 3-month male is a pooled-cell label (`3_10_M/3_11_M`); with 4 vs 4 real mice the exact p is 0.029
(its minimum), same conclusion.

## 8. Final decision (Karin, 2026-10-26)
Option (b) of AGING_SUPPLEMENT_DRAFTS.md: **the aging claim is removed.** Age and sequencing batch are the
same thing in TMS: no plate contains two ages; 3-month mice and 18/24-month mice were sequenced in different
campaigns and library formats; all 24-month mice are male. Technical covariates alone predict 18 vs 24 months
as well as splicing does (F1 0.63 vs 0.65, STAGE5 §36.3), and nearly every program shifts between the age
groups (§7). The program is kept only as an example of a partially reproducible program (centered cosine
0.65). A Data sentence states that TMS age coincides with sequencing batch and is not analyzed.

