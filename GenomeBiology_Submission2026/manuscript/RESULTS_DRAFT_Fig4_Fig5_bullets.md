# Results skeleton — Figures 4 and 5, bullet form (2026-08-19; drafts, numbers quotable)

## Fig. 4 — Splicing programs across cell types

**Programs are cell-identity structure (Fig. 4 heatmaps; FIGURE4_aging/plots/SP_by_celltype_*)**
- Mean SP activity per tissue-cell-type, K=20, 3 seeds side by side, for the atlas fit and for models
  trained on TMS alone and mouse Allen Brain alone (same junctions, no cross-dataset leakage):
  SP1 myeloid, SP4 endothelial, SP6 lymphoid, SP9 mesenchymal, SP15 keratinocyte/basal, SP16
  astrocyte/oligodendrocyte, SP17 hepatocyte, SP18 islet, SP20 intestinal (atlas numbering).
- Program replication across seeds (REVISED 2026-10-25; Methods, Program Reproducibility): matched
  on CENTERED psi (cosine >= 0.8 in both other seeds) and used by a cell type. Atlas K20: 4 of 20
  reproduce (SP5 hippocampal/excitatory, SP7 inhibitory, SP8 cortical excitatory, SP15
  keratinocyte/basal) -> consensus programs; program-level statements are restricted to these.
  The OLD figure (A61, raw psi r >= 0.9: 11-14/20) is withdrawn -- raw psi is ~0.98 similar for every
  program. The per-cell representation is stable across seeds (kNN range <= 0.025, LeafletFA
  validation 2026-10-25), and lower or higher K does not yield more reproducible programs (2-3 at
  K10/20/30 on TMS).

**A partially reproducible program (Supplementary Fig. SX; decision 2026-10-26: the aging claim is removed)**
- In Tabula Muris Senis, age coincides with sequencing campaign and library format (no plate contains two
  ages; all 24-month mice are male), so age is not analyzed (Methods, Data; audit 4b98dae; STAGE5 §36.3, §91).
- A program used mainly by endothelial and smooth-muscle cells (atlas seed-0 SP4) has a counterpart used by
  the same cell types in every fit (seed-1 SP6, seed-2 SP15), but the compositions agree only partly (centered
  cosine 0.645 and 0.537; threshold 0.8); the seed-2 program closest in composition (SP8, 0.650) is used by
  epithelial cells. Result sentence: "Programs that do not meet the reproducibility threshold can
  still be found in every fit with partly different compositions (Supplementary Fig. SX); we restrict
  program-level statements to the reproduced programs."
- Withdrawn with the aging claim: the age test, the per-cell-type age shift, the age variance terms and the
  Kcnip3 worked example (all carry the batch confound; Kcnip3 was selected from this program's age screen).

## Fig. 5 — Conservation of splicing programs between mouse and human brain

**The map (pipelines/10_cross_species/00_; CROSS_SPECIES_2026/)**
- Scripted regeneration of the 2025 mapping on the 2026 junction sets: one-to-one orthologs ->
  pyliftover mm10->hg38 -> same-gene same-strand match, both ends within 100 bp -> one-to-one by
  distance. 20,583 junction pairs (49% exact), 10,571 event pairs, 5,804 events with >= 2 matched
  junctions (used for composition comparisons). 2025's splice-site sequence validation available as
  a stricter filter.

**Program conservation (pipelines/10_cross_species/01_; CROSS_SPECIES_2026/program_match_mouse_human.tsv)**
- Mouse-AB-only vs human-AB models (same recipe, K=20, 3 seeds each; 9 seed pairs; Hungarian on psi
  deviations over the 16,544 matched junctions in >= 2-junction events).
- Three honesty guards, each changed the answer and each is stated:
  (i) raw psi correlations (0.3-0.7) sit on a junction-mean baseline of 0.64 -> use each program's
  DEVIATION from its species' pi-weighted mean composition;
  (ii) DEAD programs (max cell-type activity ~0) sit near the uniform prior in both species and
  "correlate" at 0.57-0.67 -> matching is restricted to LIVE programs (max cell-type mean activity
  >= 0.05; a global-pi cutoff would instead kill rare-cell-type programs like the mouse
  non-neuronal one);
  (iii) the CEILING is within-species: mouse-vs-mouse across seeds on the same junctions gives
  median 0.76 (IQR 0.54-0.85) -- cross-species r must be read against that, not against 1.
- RESULT: one conserved program pair, extremely stable -- the NON-NEURONAL / glial-vascular program:
  mouse SP13 (activity 0.92-0.97 in endothelial, microglia, VLMC, astrocytes, pericytes) matches
  human SP9 (0.76-0.84 in astrocytes and oligodendrocytes; human endothelial/microglia are nearly
  absent, 30-147 cells) at r = 0.581-0.589 in ALL NINE seed pairs (best match per seed pair; the
  program carries different numbers in different seeds: mouse SP13/SP14, human SP8/SP9) (~77% of the within-species
  ceiling). All neuronal programs match weakly (0.06-0.27): mouse inhibitory <-> human LAMP5/VIP/
  PAX6 at 0.26 is the best of them.
- Cell-type comparability across species (for the transfer arm and heatmaps): mouse AB = isocortex
  (motor/visual/somatosensory/temporal/association) + hippocampus with EXCITATORY / INHIBITORY /
  ASTROCYTE / OLIGODENDROCYTE / MICROGLIA / ENDOTHELIAL / PERICYTE / VLMC / OPC labels; human AB =
  cortex areas (S1, M1, A1C, V1C, CgG, MTG) with IT / L4 IT / L5-6 subclasses (excitatory), PVALB /
  SST / VIP / LAMP5 / PAX6 (inhibitory), Astrocyte / Oligodendrocyte / OPC / Microglia (147) /
  Endothelial (30). Directly comparable groups: excitatory (mouse EXCITATORY <-> human IT+L* =
  ~30k cells each), inhibitory (mouse INHIBITORY <-> human PVALB+SST+VIP+LAMP5+PAX6), astrocytes,
  oligodendrocytes+OPC; NOT comparable: vascular/immune (rich in mouse, near-absent in human AB).
  2025 harmonisation table for TMS-wide work: mouse_human_detailed_mapping.csv (archive).
- TO DO: Fig 5 draft FIGURE (mapping stats; deviation-r heatmap live x live with the ceiling;
  SP13<->SP9 side-by-side cell-type activity; a conserved-event example with genomic context in
  both genomes); transfer arm (freeze mouse psi on matched junctions, leafletfa.estep on human
  cells); per-event conserved/diverged classes; interpretation of why neuronal programs diverge
  (finer human subclasses may split what mouse merges -- test by pooling human subclasses).
