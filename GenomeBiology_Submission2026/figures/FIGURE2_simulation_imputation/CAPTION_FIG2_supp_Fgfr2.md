# Supplementary Figure (FIG2_supp_Fgfr2_IIIb_IIIc): Fgfr2 exon IIIb / IIIc, covered and uncovered cells

Built by `code/build_fgfr2_example.py` (full-atlas K = 20 fits, seeds 0–2); per-type numbers in
`data/fgfr2_IIIb_covered_uncovered.tsv`.

**Fgfr2 IIIb / IIIc: LeafletFA recovers the lineage direction; for cells without reads it reports a program mixture.**
**(a)** *Fgfr2* ATSE 3 (chr7, minus strand): the shared upstream exon is spliced to exon IIIb (epithelial
isoform) or exon IIIc (mesenchymal isoform). Arcs mark the two junctions; transcripts below.
**(b)** IIIb usage per cell in 14 epithelial and 13 mesenchymal tissue cell types (types with ≥ 5 cells
carrying reads at the event), each split into covered cells (≥ 1 read at the event) and uncovered cells.
Left: raw usage of covered cells (points and violin). Middle: LeafletFA posterior mean for the same
covered cells. Right (hatched): LeafletFA for uncovered cells, where no reads enter and the posterior
equals the prior φ·ψ. LeafletFA values are the mean over three seeds per cell; horizontal bars are
type means; ticks are the type mean in each seed. Numbers below the axis are cells per type (top:
covered; bottom: uncovered). Covered cells: 1,573 epithelial, 194 mesenchymal. Uncovered cells:
14,537 epithelial, 10,721 mesenchymal.
**(c)** Mean φ of the same cell types in each seed (columns clustered), programs ordered by π
(SP1 = largest), each labeled with its IIIb usage ψ at this event.

LeafletFA recovers the lineage direction: on the seed mean, every mesenchymal type sits below every
epithelial type, both for covered cells (0.05–0.46 vs 0.80–0.99) and for uncovered cells (0.35–0.63
vs 0.64–0.95). It does not recover mesenchymal IIIc usage for uncovered cells. There it reports the
mixture of programs the cell type uses, and some of those programs are shared with epithelial cells
and use IIIb at this event (for example SP1 and SP4 in seed 0, SP1 in seed 1; ψ ≈ 1.0). So mesenchymal
values sit at 0.3–0.8 (seed means 0.35–0.63; single seeds 0.21–0.81) rather than near 0. Covered
mesenchymal cells track their reads more closely: 0.05–0.15 against raw 0 in 12 of 13 types (Aorta
fibroblast 0.46 against raw 0.38). The shared programs differ by seed (c): in seed 2 the program that
mesenchymal types load most uses IIIc (ψ_IIIb 0.14), and uncovered mesenchymal values are lower
(0.21–0.52); in seeds 0 and 1 they reach 0.81, and lung smooth muscle (0.81) exceeds mammary basal
cells (0.61 and 0.53) within those single seeds.
