# Program stability: what it cuts in Figs 3-5, and a consensus-program proposal (2026-09-28)

Rule: program-specific claims only for programs matched at cosine >= 0.8 in both other seeds on
mean-centred psi (A70) AND used by some cell type (max cell-type mean phi >= 0.05).
Atlas K=20 passes: SP5 (hippocampal/excitatory), SP7 (inhibitory), SP8 (cortical excitatory),
SP15 (keratinocyte/basal). Everything else fails (SP4 aging 0.65, SP1 myeloid 0.52, SP9 mesenchymal 0.67,
SP16 astro/oligo 0.76, ...).

## What is and isn't program-specific, by figure
| figure / panel | uses programs as... | status under the rule |
|---|---|---|
| Fig 2 all panels (sims; transfer scatter; SP heatmaps train vs held-out) | the whole basis / simulated truth | unaffected |
| Fig 3 a-c (calibration, matched calls, posterior split bar) | whole basis (TV is a basis-level quantity) | unaffected |
| Fig 3 d examples (Apbb2, Mbd3) | posterior TV over the basis; no program named | unaffected |
| Fig 3 text: Gna13 "program-level, amplified" explanation | names the endothelial (SP4) and mesenchymal (SP9) programs | CAVEAT or re-word basis-level (both unstable) |
| Fig 3 supp program_sharing (ceiling vs ||dphi||) | basis-level ceiling; heatmap names all SPs | ceiling unaffected; heatmap: mark stable SPs, or collapse to basis |
| Fig 3 supp Fbln2 decomposition | per-program psi/phi bars, "shared SP" | CAVEAT: programs individually unstable |
| Fig 4 main a (ANOVA) | per-SP rows | keep only cell-type/sex columns for SP5/7/8/15, or cut |
| Fig 4 main b-f | SP4 (aging) | CUT (unstable and age-confounded) |
| Fig 4 supp a (pi/PVE), b (SP clustermap) | all SPs | show as basis description; highlight stable 4 |
| Fig 4 supp c (AB vs TMS broad types) | all SPs | same |
| Fig 4 supp d (top junctions per SP) | per-program content | restrict to SP5/7/8/15 |
| Fig 4 supp e/f (Clstn1) | junction usage by cell type; no program | unaffected |
| SP-by-cell-type heatmaps (TMS-only, AB-only, human) | all SPs | mark stable SPs per grid |
| Fig 5 conservation pair: mouse AB-only SP13 <-> human SP9 | two named programs | PASSES: centred min cos 0.971 (mouse) and 0.982 (human) |

## Consensus programs -- proposal (not started; no GPU jobs submitted)
Goal: replace "programs of one seed" with programs defined by agreement across fits.
1. Pool mean-centred psi from N fits (seeds x K in {20, 30}); cluster programs by cosine (average
   linkage, cut at 0.8). Keep clusters with a member in >= 80% of fits and some cell-type activity >= 0.05.
2. Consensus psi = member mean (renormalised within ATSE); re-infer phi for all cells by the E-step
   with consensus psi frozen (leafletfa.estep), then use consensus programs for every program-specific panel.
Cost:
- Option A, existing fits only (3 seeds x K20/K30 per grid): no training; clustering CPU minutes;
  one E-step per grid (~20-30 min GPU for 132k cells). Weakness: 3 seeds is a thin basis for "80% of fits".
- Option B, 10 seeds at K=20 on the atlas: 7 new fits x ~5 h GPU (300 GB) = ~35 GPU-h, plus the E-step.
- Same for TMS-only / AB-only / human: ~3-5 h GPU per extra fit each.
Recommendation: A first (free) to see how many consensus programs exist; B only if A leaves the key
biology (e.g. myeloid, mesenchymal, astro/oligo) without a stable program.
