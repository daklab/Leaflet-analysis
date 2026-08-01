# Figure legends — splicing events per gene

Four supplementary figures, one per dataset, produced by `supp_exons_vs_atses.py`. All share
the same six panels, so the legends are parallel and can be read side by side.

**Scope common to all four.** Each figure describes the genes present in that dataset's
model-ready splicing object, i.e. genes carrying at least one ATSE. Genes without an ATSE are
absent from these objects by construction (`09b_filter_aligned.py` retains only genes with
surviving junctions), so the figures address how many splicing events a gene carries, not
which genes carry any.

**Definitions used throughout.** Exon count is the maximum number of exons across a gene's
annotated transcripts. Gene read depth is the total raw exonic count for that gene, summed
across cells, taken from the gene-expression object paired with the same splicing object.
ρ is Spearman's rank correlation; the value in parentheses is the partial rank correlation
controlling for the other variable.

---

## Supplementary Figure S_a. Mouse Tabula Muris Senis (single cells)

Splicing events per gene in Tabula Muris Senis, the whole-cell (SMART-seq2, FACS) component
of the mouse resource: 78,135 cells, 7,869 genes with an ATSE, median 6 junctions and 2 ATSEs
per gene.
**(a)** Median ATSEs per gene by exon count; bars show the median, whiskers the interquartile
range, numbers the genes per bin. Events per gene rise from 1 (2 exons) to 5 (≥65 exons).
**(b, c)** Junctions (b) and ATSEs (c) detected per gene against exon count. ρ = 0.36 and 0.47
respectively, essentially unchanged after controlling for depth (0.34, 0.46).
**(d, e)** The same outcomes against gene read depth. Depth shows no positive association here
(ρ = −0.18 for both; −0.14 after controlling for exon count), and the negative sign is not
interpretable across datasets (see note).
**(f)** Median ATSEs per gene for each combination of exon count and depth quintile; tile
labels give the median. Exon count separates the rows and depth is close to flat within a row,
the only systematic departure being a slight *fall* at the highest depth quintile (2 → 1 in the
5–8 exon bin, 3 → 2 in the 17–32 bin), consistent with the negative correlations in d and e.
The ≥65 exon row (*n* = 35) is too small to read.

## Supplementary Figure S_b. Mouse Allen Brain (single nuclei)

Splicing events per gene in the Allen Brain single-nucleus (SMART-seq v4) component of the
same mouse object: 54,571 nuclei, 9,064 genes with an ATSE, median 6 junctions and 2 ATSEs per
gene. Junction detection was recomputed within the nuclei subset, so counts reflect this
modality alone.
**(a)** As above; events per gene rise from 1 to 5 across exon bins.
**(b, c)** ρ = 0.38 (junctions) and 0.50 (ATSEs) against exon count, unchanged by controlling
for depth (0.38, 0.50).
**(d, e)** Depth is uncorrelated with detection (ρ = 0.03 and 0.02).
**(f)** At the highest depth quintile, median ATSEs per gene rise from 1 to 9 across exon bins,
while within the most exon-rich bin the range across depth quintiles is 2 to 9.

## Supplementary Figure S_c. Human Allen Brain cortex (single nuclei)

Splicing events per gene in human cortex: 43,442 nuclei, 10,003 genes with an ATSE, median 7
junctions and 2 ATSEs per gene.
**(a)** Events per gene rise from 1 (2 exons) to 6 (≥65 exons), with the interquartile range
widening from 1–1 to 3–10.
**(b, c)** ρ = 0.41 (junctions) and 0.53 (ATSEs) against exon count; 0.33 and 0.47 after
controlling for depth.
**(d, e)** Unlike the mouse datasets, depth is positively associated with detection here
(ρ = 0.41 and 0.38; 0.33 and 0.26 after controlling for exon count).
**(f)** Both variables contribute. Within the 17–32 exon bin the median rises from 2 to 5
across depth quintiles, while at the highest depth quintile it rises from 1 to 10 across exon
bins. Genes with 2–4 exons remain at a median of 1 ATSE at every depth.

## Supplementary Figure S_d. Human RNA atlas, bulk (GSE138734)

Splicing events per gene across 2,228 bulk libraries spanning 160 primary cell types: 13,847
genes with an ATSE, median 7 junctions and 2 ATSEs per gene.
**(a)** Events per gene rise from 1 to 7 across exon bins, the widest range of the four
datasets, with the interquartile range widening from 1–1 to 2–11. This is the only dataset with a populated single-exon bin (*n* = 4): four loci that
GENCODE v45 annotates with one exon and one transcript nonetheless carry a two-junction ATSE,
i.e. splicing that is unannotated at these loci. Three are lncRNAs or pseudogenes and one is
annotated `gene_type "artifact"`; with *n* = 4 the bar is illustrative only.
**(b, c)** ρ = 0.50 (junctions) and 0.60 (ATSEs) against exon count, the strongest of the four;
0.45 and 0.57 after controlling for depth.
**(d, e)** Depth is positively associated marginally (ρ = 0.25, 0.26) but the association
largely disappears after controlling for exon count (0.08, 0.05), indicating that deeply
sequenced genes are predominantly exon-rich genes.
**(f)** At the highest depth quintile the median rises from 1 to 9 across exon bins.

---

## Notes for the caption or methods

**Depth is not comparable across the four figures.** The marginal depth associations range
from −0.18 (mouse cells) to +0.41 (human nuclei). This ordering does not follow gene count or
depth range, and the four gene-expression objects derive from different quantification
pipelines, so gene read depth is not the same measured quantity in each. Report depth within a
dataset; do not rank datasets by it.

**Exon count is comparable and consistent.** Partial ρ for exon count against ATSEs detected is
0.46–0.57 in all four datasets, and is essentially unchanged by controlling for depth.

**Exon counts scale with annotation depth.** GENCODE v45 (human) is more deeply annotated than
vM19 (mouse), so absolute exon counts are not strictly comparable between species; the
correlations are.

**Low-count bins are a floor, not a measurement.** A gene appears to have few junctions or a
two-junction ATSE when its rarer junctions fall below detection, so complexity is
underestimated at low coverage.

**The accompanying CSV contains rows the figures exclude.** Each `supp_exons_vs_atses_*.csv`
is written from the full expressed-gene table, including a small number of genes with
`has_atse = False` (36 atlas, 11 mouse nuclei, 10 mouse cells, 0 human). These are not
junction-free genes: `09b_filter_aligned.py` aligns the gene-expression and splicing objects on
gene **symbol**, whereas this analysis joins on Ensembl **gene_id**, and roughly 125 symbols map
to more than one gene_id in GENCODE v45. The surplus gene-expression rows therefore share a
symbol with a junction-bearing gene but carry a different gene_id, chiefly the pseudogene and
lncRNA identifiers added in v45. Every panel is drawn from the `has_atse = True` subset, so
these rows affect no plotted quantity. Filter on `has_atse` before reusing the CSV.
