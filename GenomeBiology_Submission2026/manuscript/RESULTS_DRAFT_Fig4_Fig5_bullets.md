# Results skeleton — Figures 4 and 5, bullet form (2026-08-19; drafts, numbers quotable)

## Fig. 4 — Splicing programs across cell types, and the aging program

**Programs are cell-identity structure (Fig. 4 heatmaps; FIGURE4_aging/plots/SP_by_celltype_*)**
- Mean SP activity per tissue-cell-type, K=20, 3 seeds side by side, for the atlas fit and for models
  trained on TMS alone and mouse Allen Brain alone (same junctions, no cross-dataset leakage):
  SP1 myeloid, SP4 endothelial, SP6 lymphoid, SP9 mesenchymal, SP15 keratinocyte/basal, SP16
  astrocyte/oligodendrocyte, SP17 hepatocyte, SP18 islet, SP20 intestinal (atlas numbering).
- Program replication across seeds (A61, r >= 0.9 on psi): K20 11-14/20 on every dataset (atlas,
  70/30, TMS-only, AB-only, human); across ranks K10->K20 6-9/10, K20->K30 11-15/20. Effective K is
  a property of the data, not one fit.

**The aging program (Fig. 4 aging draft: FIGURE4_aging/plots/FIG4_aging_draft)**
- Design guards: TMS only (allen_brain = all 2m, one lab -> excluded); 24m is MALE-only, so the
  primary test is per-MOUSE, males, 3m (5 mice) vs 24m (4 mice), Mann-Whitney; p floors at 0.016.
- One program passes floor-p AND breadth in all 3 seeds: activity UP with age in 94/94/99% of
  cell types (per-type Spearman median 0.37/0.46/0.64); psi-matched across seeds r 0.95-0.97.
  Atlas seed-0 name: SP4 [check SP label per fit]. 3m mean 0.13 -> 24m 0.22 (seed 0).
- Variance partition (Type-II ANOVA, panel a): cell type dominates every program; the aging program
  carries the largest age and cell-type x age shares [numbers in data_variance_partition.tsv].
- Aging shift by cell type (panel c): delta mean activity (24m - 3m, males) positive across nearly
  all sufficiently covered tissue-cell-types [breadth from a69 delta_bytype].
- Program content (panel d): top junctions by |psi_aging - pi-weighted rest|; genes incl. Sos1,
  Frmd4b, Ppfibp2, Lipe, Plekhg5, Rps6ka3, Xiap, Fbln2, Cald1, Itga7 (seed 0 top-25).
  2025's annotation classes (GenAge overlap, RBP targets) can be layered on; RBP mediation excluded.
- Worked example (FIG4_example_Kcnip3, screened by A69b over the aging program's top-200 junctions x
  all TMS tissue-cell-types with >= 25 covered cells at each of 3/18/24m): Kcnip3 ATSE 2 in marrow
  hematopoietic progenitors -- J1 usage falls 0.35 -> 0.12 -> 0.03 across 3/18/24m (50/48/52 covered
  cells; per-cell Spearman -0.26) while Kcnip3 expression is flat (Spearman 0.03). Genomic context +
  age violins + GE violins. Runner-ups in ATLAS_DM/aging/a69b_example_candidates.tsv: Sash1 (GAT MSC,
  0.44 -> 0.74 UP), Upp1 (marrow HSC, 0.11 -> 0.40 UP), Usp2 (bladder), Kcnq1 (lung SMC).
- Panel notes addressed (2026-08-19): ANOVA heatmap annotates shares >= 0.05 (values are each term's
  share of the model's explained SS; cell type IS tissue_cell_type, with tissue_cell_type:age and
  sex:age interactions); violins carry median + IQR; per-mouse dots now both sexes (circles male,
  triangles female -- the TEST stays males-only because 24m has no females); delta heatmap in the
  2025 circle style (tile = 24m-3m, circle = median activity).
- TO DO: 2025-style perplexity-by-age/sex and SP-correlation panels as supp if wanted; DS-based
  program content via the posterior test; age-prediction ridge only if kept.

## Fig. 5 — Conservation of splicing programs between mouse and human brain

**The map (pipelines/10_cross_species/00_; CROSS_SPECIES_2026/)**
- Scripted regeneration of the 2025 mapping on the 2026 junction sets: one-to-one orthologs ->
  pyliftover mm10->hg38 -> same-gene same-strand match, both ends within 100 bp -> one-to-one by
  distance. 20,583 junction pairs (49% exact), 10,571 event pairs, 5,804 events with >= 2 matched
  junctions (used for composition comparisons). 2025's splice-site sequence validation available as
  a stricter filter.

**Program conservation (pipelines/10_cross_species/01_; program_match_mouse_human.tsv)**
- Mouse-AB-only vs human-AB models (same recipe, K=20, 3 seeds each; 9 seed pairs; Hungarian on psi
  over the 16,544 matched junctions in >= 2-junction events).
- KEY honesty point (mirrors the imputation lesson): raw psi correlations (0.34-0.67) sit on a
  junction-mean baseline of 0.64 (mean mouse psi vs mean human psi), so conservation is measured on
  each program's DEVIATION from its species' mean composition.
- Result: the small-pi, cell-type-specific programs are the conserved ones -- deviation r 0.57-0.67
  across all 9 seed pairs for mouse SP14-SP20 (oligodendrocyte/astrocyte-type programs) -- while the
  broad high-pi programs' deviations correlate weakly (0.13-0.34). [Interpretation to sharpen: broad
  programs may be atlas-composition-specific; specific programs carry conserved splicing.]
- TO DO: (i) name the conserved pairs by their cell types (join the SP-by-cell-type heatmaps across
  species); (ii) transfer learning as in 2025 (freeze mouse psi on matched junctions, learn human
  phi; or use leafletfa.estep) -> do human cells land on the right programs; (iii) per-event
  conserved vs diverged classification (deviation agreement per ATSE) with examples; (iv) sequence-
  validation filter sensitivity; (v) SP-by-cell-type comparison with the harmonised cell-type map
  (2025 human_mouse_detailed_mapping.csv).
