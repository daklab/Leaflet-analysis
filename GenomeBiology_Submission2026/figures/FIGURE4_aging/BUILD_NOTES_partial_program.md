# Build notes — Supplementary Fig. SX, a partially reproducible program (replaces the aging figure)

Decision (Karin, 2026-10-26): the aging claim is removed. The former aging program is kept only as an
example of a program that does not meet the reproducibility threshold. Caption: `CAPTION_FIG4.md`.
Everything is built from existing fits; no training. Follow `figures/FIGURE_THEME.md` and import
`figures/_common/leaflet_theme.py`.

## Why age is not analyzed (cite these)

- **Data sentence (Methods, TMS item):** in Tabula Muris Senis, age coincides with sequencing campaign and
  library format, so age is not analyzed.
- **Audit, Leaflet-analysis commit 4b98dae** (`audit/age_confound_audit.py`, `audit/age_confound_audit.tsv`):
  - 0 of 510 plates contain cells of two ages;
  - 3-month cells are one campaign, 18- and 24-month cells later formats;
  - all 24-month mice are male.
- **STAGE5 §36.3** (SpliceVI-utils `manuscript/age_analysis/audit_2026_09/STAGE5_sequential_plan.md`): 18 vs
  24 months within the later campaign.
  - Technical covariates alone separate the ages as well as splicing latents: F1 0.63 vs 0.65, both
    p = 0.029 over 70 mouse-label assignments.
- **STAGE5 §91:** every 18-month mouse in one library format, every 24-month mouse lane-demultiplexed; plates
  nested in mice.
- **Supporting:**
  - STAGE5 §27 (the confound), §88b (most of the age signal is mouse-level structure), §89.2 (same-age
    mice strongly distinguishable).
  - The pre-registered mouse-level test of the LeafletFA program (`manuscript/AGING_SUPPLEMENT_DRAFTS.md`
    §2; `audit/aging_mouse_level_test.py`): most programs separate the age groups, so the shift is not
    program-specific.

## The program

- Atlas K = 20, `ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{0,1,2}_b8192/fit.npz`.
- **Seed 0:** program index 6 (SP4, endothelial / smooth muscle).
- **Matches by Hungarian assignment on centered ψ** (each fit's ψ minus its π-weighted mean composition;
  cosine):
  - seed 1: index 5 (SP6), cosine 0.645;
  - seed 2: index 9 (SP8), cosine 0.650.
- Note: these are NOT A69's seed-1 "aging" pick (SP3). Use the centered matches.
- **Built 2026-10-27:** `code/build_supp_partial_program.py` → `supplemental/partial_program/SX_partial_program.{pdf,png}` + `_data.tsv`.
  The build found that the seed-2 composition match SP8 is used by epithelial cells (profile r 0.28); the
  seed-2 program used by the same cells is SP15 (profile r 0.82, centered cosine 0.537). Panels a and c show
  both. Panel c uses the top 8 junctions per program (top 15 was unreadable).

## Panels to build

| panel | content | source | notes |
|---|---|---|---|
| a | Mean activity (φ) per tissue–cell type of the matched program in each fit, side by side (3 columns: seed 0 SP4 / seed 1 SP6 / seed 2 SP8); rows = the 20 tissue–cell types with the highest mean across fits | fit.npz `phi` + atlas obs (`train_splice_model_ready_2026-06-17.h5ad`: `tissue_cell_type`), rows via `pipelines/05_leafletfa/_phi_rows.py` | sequential `mako`, shared 0–max scale; types with ≥ 20 cells; state n cells per type in the caption or a side bar |
| b | Centered-ψ cosine (minimum over the two other seeds) for every program in use in seed 0, sorted; this program highlighted; dashed line at 0.8; the four reproduced programs labeled (SP5, SP7, SP8, SP15) | `ATLAS_DM/stability/a70_program_stability_centred.tsv` (grid `atlas`, K 20) | "in use" = max cell-type mean φ ≥ 0.05 (A70's rule); grey for programs not in use if shown at all |
| c | Top 15 junctions per fit by departure from the π-weighted mean of the other programs (\|ψ_k − ψ̄_−k\|), gene-labeled; marker for junctions in the top 15 of all three fits vs one or two | fit.npz `psi_mean`, `pi`; var `gene_name`, `junction_id` | one shared junction list (union) with three columns of \|Δψ\| makes shared vs seed-specific visible; diverging `BrBG` centered on 0 if signed |

Width: `figsize("double", h≈3.0)`. Save with `save(fig, "supplemental/partial_program/SX_partial_program")`.

## Retired with the aging claim (do not delete; just stop building)

- Panels: old Fig 4 b (age violins), c (per-cell-type age shift), the age terms of the a-panel variance
  partition, e/f (*Kcnip3* worked example).
- Scripts: `code/build_fig4_main.py`, `code/build_fig4_example.py`, `pipelines/05_leafletfa/A69_aging_program.py`,
  `A69b_aging_example_screen.py`.
- The non-aging supplementary panels in `code/build_fig4_supp_nonaging.py` (π/PVE, SP clustermap, AB vs
  TMS broad types, top junctions per SP, *Clstn1*) are unaffected.
