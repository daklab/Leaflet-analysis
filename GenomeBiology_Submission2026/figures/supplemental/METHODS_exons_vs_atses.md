# Exon count, sequencing depth, and detected splicing events

Draft methods + results for the supplementary figure
(`supp_exons_vs_atses.py` → `supp_exons_vs_atses_{rna_atlas,mouse_smartseq2,human_allen_brain}.{png,pdf,csv}`).

---

## Methods

### Datasets

Three datasets were processed through the same junction-extraction, ATSE-mapping and
filtering pipeline, differing only in the source annotation and the depth-appropriate filter
thresholds.

**Human bulk RNA-seq atlas (GSE138734).** A multi-tissue bulk RNA-seq atlas spanning **160
primary cell types** across **23 organ systems**. We restricted the atlas to primary
cell-type samples, excluding bulk mixed-tissue samples, cancer cell lines and immortalised
cell lines. Importantly, this resource contains **both polyA-selected and total RNA
(Ribo-Zero Gold) libraries**: every RNA sample was prepared under multiple protocols, so the
2,228 sequencing runs analysed here correspond to **316 GSM libraries derived from ~158
distinct RNA samples**. The 2,228 units are therefore sequencing *runs*, not independent
biological replicates, and the two libraries per cell type are protocol replicates of the
same RNA rather than biological ones. Each run was treated as an individual pseudo-sample for
junction quantification. Reads were aligned with STAR to GRCh38 (GENCODE v45).

**Mouse single-cell / single-nucleus SmartSeq2.** Full-length SmartSeq2 data combining
**Tabula Muris Senis** (FACS-sorted **single cells**, multi-organ) with **Allen Brain mouse
cortex** (**single nuclei**). The combined dataset therefore contains both single-cell and
single-nucleus libraries. Reads were aligned to mm10 (GENCODE vM19).

**Human single-nucleus brain.** Allen Brain human cortex (`lein-human-cortex`), consisting
**entirely of brain-derived single nuclei** (SmartSeq2). Reads were aligned to GRCh38
(GENCODE v45).

### Junction extraction, ATSE mapping and filtering

Splice junctions were extracted from aligned reads with `regtools` and pooled across
cells/runs. Junctions were retained if supported by ≥100 reads in ≥10 cells (bulk atlas), and
were required to use canonical splice sites. Alternative transcript splicing events (ATSEs)
were then called *de novo* with ATSEmapper: junctions were assembled into a splice graph and
grouped into connected components that **share a splice site**, with each component of ≥2
junctions defining one ATSE. ATSE calling used the same GENCODE annotation as the alignment
for each dataset.

Model-ready objects were produced by applying, in order, (i) a junction filter (junctions
detected in fewer than *N* cells removed), (ii) a cell filter (cells with <500 surviving
junctions removed), and (iii) **re-grouping of ATSEs** on the surviving junctions using the
same splice-graph procedure, so that ATSEs whose connecting junctions were removed do not
persist as incoherent groups. Junctions left as singletons after re-grouping (i.e. sharing no
splice site with any competing junction, and therefore constitutive) were dropped.

Thresholds were set per dataset according to depth (bulk libraries are far deeper per unit
than single cells), but the procedure was otherwise identical:

| Dataset | Units | Junction filter | Junctions | ATSEs | Genes with ATSEs |
|---|---|---|---|---|---|
| Human bulk atlas | 2,228 runs | ≥50 units | 128,138 | 35,614 | 13,886 |
| Mouse SC/SN SmartSeq2 | 132,706 cells/nuclei | ≥75 cells | 79,768 | 24,475 | 11,236 |
| Human SN brain | 43,442 nuclei | ≥25 nuclei | 97,144 | 27,082 | 10,327 |

### Gene-level exon count, depth, and detected splicing events

For each gene we defined:

- **Exon count** = the **maximum number of exons across all annotated transcripts** of that
  gene. We deliberately did *not* use a single "representative" or canonical isoform: the
  quantity of interest is whether a gene *can* be spliced at all, which is determined by
  whether **any** transcript contains an intron. Isoform-based rules fail this test (a gene
  with a single-exon principal isoform but a multi-exon alternative isoform is genuinely
  spliced), and are not portable across annotations — GENCODE vM19 lacks the `MANE_Select`
  and `Ensembl_canonical` tags present in v45, so any tag-based rule silently changes
  definition between species. The max-exon rule is tag-free and identical across all three
  datasets.
- **Sequencing depth** = total raw counts for that gene, summed over the cells/runs shared
  with the corresponding splicing object (raw count matrices only; normalised matrices were
  never used for this axis).
- **Detected junctions / ATSEs** = the number of junctions and distinct ATSEs assigned to that
  gene in the final filtered splicing object.

Analyses were restricted to **expressed** genes (detected in ≥10% of runs for the bulk atlas;
≥5% of cells for the single-cell/nucleus datasets), since otherwise the comparison would
largely measure "not expressed ⇒ nothing detected". Gene-expression matrices used here are the
**full** quantifications, not the gene-subset objects used for model training (which retain
only genes carrying junctions and would render the analysis circular).

Because exon count and depth are correlated (longer, multi-exon genes tend to be more highly
expressed), we report both marginal Spearman correlations and **partial Spearman
correlations**, in which each variable is correlated with the outcome after regressing out the
rank of the other.

---

## Results

**Exon count, not sequencing depth, determines how many splicing events we detect.**

Across all three datasets, the number of exons in a gene was the dominant predictor of the
number of ATSEs detected in it (partial Spearman ρ = **+0.57**, **+0.49** and **+0.47** for the
human bulk atlas, mouse SmartSeq2 and human single-nucleus brain respectively, controlling for
depth). The apparent contribution of sequencing depth largely disappeared once exon count was
accounted for: in the bulk atlas the marginal association of depth with ATSE number (ρ = +0.26)
fell to ρ = +0.06 after controlling for exon count, and in mouse it was abolished entirely
(ρ = −0.08). Depth retained a modest independent effect only in the human single-nucleus data
(partial ρ = +0.26), consistent with dropout making detection genuinely depth-limited in
sparse single-nucleus libraries; notably, the residual depth effect tracks the breadth of the
quantified gene universe rather than the species (the mouse expression matrix is pre-restricted
to well-expressed genes, compressing the depth range).

**Genes without detected ATSEs are overwhelmingly genes that cannot be spliced.** ATSEs were
called in 13,886, 11,236 and 10,327 genes in the three datasets — a minority of all quantified
genes. This is expected rather than a failure of detection: an ATSE requires ≥2 junctions
competing at a shared splice site, so a gene must contain at least one intron to contribute
one. Among **expressed single-exon genes**, essentially none carried an ATSE (**4 / 7,176,
0.1%** in the bulk atlas; **0 / 1,070, 0.0%** in the human single-nucleus data), whereas
57–80% of expressed multi-exon genes did. Detection rose monotonically with exon count, from
~0% for single-exon genes to >80% for genes with ≥17 exons.

Together these results indicate that the gene coverage of our splicing atlas is set by
transcript architecture rather than by sequencing depth, and that genes lacking ATSEs are
predominantly intronless or constitutively spliced loci for which no alternative splicing
choice exists to quantify.

---

## Caveats to state

1. **Gene universes differ between datasets** and are not directly comparable: the bulk atlas
   quantifies all ~62k GENCODE genes (including mitochondrial, single-exon and non-coding
   genes), whereas the mouse expression matrix is pre-restricted to ~19k well-expressed genes
   (and thus contains almost no single-exon genes) and the human single-nucleus matrix ~47k.
   This explains the differing ATSE detection rates (57% / 80% / 66% of expressed multi-exon
   genes) and the differing residual depth effects.
2. **Max-exon counts scale with annotation depth.** GENCODE v45 is more deeply annotated than
   vM19, so absolute exon counts are not strictly comparable across species; the correlations
   and the single-exon result are.
3. **The bulk atlas has no biological replicates** — its 2,228 units are sequencing runs from
   316 libraries of ~158 RNA samples, and polyA and total-RNA (Ribo-Zero) libraries of the same
   RNA are present. Total-RNA libraries retain unspliced pre-mRNA, which is expected to shift
   PSI for intron-retention-like events; this affects roughly half the libraries.
