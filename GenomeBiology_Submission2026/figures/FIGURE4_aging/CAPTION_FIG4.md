# Supplementary Figure SX caption — a partially reproducible program (replaces the former Figure 4 aging figure)

Decision (Karin, 2026-10-26): the aging claim is removed (option (b) of `manuscript/AGING_SUPPLEMENT_DRAFTS.md`).
In Tabula Muris Senis, age coincides with sequencing campaign and library format, so age is not analyzed.
The former aging program is kept only as an example of a program that does not meet the reproducibility
threshold. Build notes and the panels to build: `BUILD_NOTES_partial_program.md` (this folder).

**Supplementary Figure SX. A partially reproducible program.**
A program used mainly by endothelial and smooth-muscle cells has a counterpart used by the same cell types in
each of the three seeded fits of the atlas model (K = 20), but the junction compositions agree only partly.
**(a)** Mean activity per tissue–cell type (number of cells in parentheses) of the program in seed 0 (SP4), its
counterpart in seed 1 (SP6, the closest match both in composition and in cell-type usage), and two programs of
seed 2: SP15, used by the same cell types, and SP8, the closest in composition, which is used by epithelial
cells instead. Twenty tissue–cell types with the highest mean activity shown.
**(b)** Centered-ψ cosine to the closest program in the two other fits (minimum of the two) for every seed-0
program in use; this program in teal, the four programs that reproduce (≥ 0.8) in blue.
**(c)** Departure of each shown program's composition from the π-weighted mean of the other programs of its fit
(Δψ), for the union of each program's top eight junctions. Seed 0 and seed 1 share their strongest junctions
(*Sos1*, *Frmd4b*, *Lipe*, *Plekhg5*), as does seed-2 SP15 for most of them; seed-2 SP8 does not.
Statements about this program are restricted to what the fits share (Methods, Program Reproducibility).
Source data: `supplemental/partial_program/SX_partial_program_data.tsv`; code:
`FIGURE4_aging/code/build_supp_partial_program.py`.
