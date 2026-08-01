# Run log — supplemental data-quality figures

Everything needed to reproduce the six figures in `data_quality/output/` from the model-ready
AnnData objects. Two independent scripts; no SLURM, no GPU. Wall-clock for a cold rebuild of
all six is ~35 min, dominated by two CSR scans of the 30 GB mouse object.

Built **2026-07-26**. Repo revision at build time **`109b366`** (branch `main`).
Both scripts were uncommitted working-tree files at build time — commit them before citing this
log as a provenance record (see *Open items*).

Environment (`conda activate LeafletSC`):

| | |
|---|---|
| python | 3.10.13 (`~/miniconda3/envs/LeafletSC/bin/python`) |
| numpy / pandas / scipy | 1.26.4 / 2.2.3 / 1.14.1 |
| matplotlib / seaborn | 3.8.4 / 0.13.2 |
| anndata / h5py | 0.11.3 / 3.14.0 |

---

## 1. Input data files

Five distinct files across three species/assays. Sizes are as measured on 2026-07-26; the
coverage script records size and mtime in its cache manifest and refuses a stale cache.

### Mouse — Tabula Muris Senis + Allen Brain, one pooled object

```
splice  /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/
        MODEL_INPUT/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad
        29.85 GB   mtime 2026-06-17 16:34
        132,706 cells x 79,768 junctions   24,475 ATSEs   11,236 genes
        layers: cell_by_junction_matrix (CSR int64, nnz 603,601,993)
                cell_by_cluster_matrix  (CSR int64, nnz 1,460,451,625)  <- never read, see §5
        X == cell_by_junction_matrix (byte-identical duplicate)

ge      .../MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad
        12.64 GB   mtime 2026-06-17 16:34
        132,706 cells x 11,233 genes    .X = RAW integer counts (max 229,292)

gtf     .../TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf     0.68 GB
```

The `obs['dataset']` column splits this object into the two modalities:
`tabula_muris_senis` = 78,135 single **cells** (SMART-seq2, FACS);
`allen_brain` = 54,571 single **nuclei** (SMART-seq v4).

> **The 2026-06-17 files supersede 2026-06-13.** The older pair predates the `_regroup_atses()`
> fix and reports 83,593 junctions / 25,750 ATSEs. Any figure showing those numbers was built
> from the wrong input.

### Human — Allen Brain cortex

```
splice  .../HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/
        splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad    7.64 GB
        43,442 nuclei x 97,144 junctions   27,082 ATSEs
ge      .../MODEL_INPUT/20260617/ge_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad
        10.14 GB    43,442 nuclei x 10,327 genes    .X = RAW counts (max 174,878)
gtf     /gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/
        gencode.v45.primary_assembly.annotation.gtf      1.46 GB
```

The human GE object also carries `counts` / `raw_counts` layers. **Ignore them** — `.X` already
is the raw matrix, and both scripts read `.X` for every dataset so the four figures use the same
quantity. (The normalised-`.X` object is a *different*, unfiltered file,
`lein-human-cortex/GeneExpression/adata_2025-01-07_HVG_only_False.h5ad`; it is not used here.)

### Human — RNA atlas, bulk (GSE138734)

```
splice  .../RNA_ATLAS/GSE138734/MODEL_INPUT/20260709/
        splice_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad    5.33 GB
        2,228 libraries x 128,138 junctions   35,614 ATSEs
ge      .../MODEL_INPUT/20260709/ge_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad
        0.42 GB     2,228 x 13,993 genes   (featureCounts, gencode v45, -s 2 -p --countReadPairs)
gtf     gencode.v45.primary_assembly.annotation.gtf   (same file as human Allen Brain)
```

2,228 primary-cell-type libraries; bulk mixed tissues (673) and cell lines are excluded upstream
via `junctions_celltype_only.txt`. Aligned to the same GRCh38 v45 index as the human brain
atlas, so coordinates are directly comparable — no liftover.

**Cell-order alignment.** `obs['cell_id']` is element-wise identical and in the same order
between the splice and GE object of every pair. Both scripts assert this; positional alignment
is safe and no ID cleaner is applied (`clean_ge = False` everywhere).

---

## 2. Commands

```bash
conda activate LeafletSC
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/supplemental/data_quality
```

### Figure 1 — dataset coverage (2 figures, 8 panels each)

```bash
python supp_dataset_coverage.py --dataset mouse_smartseq2      # ~14 min (cold cache)
python supp_dataset_coverage.py --dataset human_allen_brain    # ~ 5 min (cold cache)
```

Two stages. Stage 1 scans the h5ad files and writes three parquet tables plus a JSON manifest
into `cache/`. Stage 2 reads only the parquets. **Re-plotting never rescans the 30 GB object**,
so panel edits are seconds, not minutes. Add `--force-recompute` to redo stage 1.

### Figure 2 — splicing events per gene (4 figures, 6 panels each)

```bash
python supp_exons_vs_atses.py --dataset mouse_tms              # ~4 min
python supp_exons_vs_atses.py --dataset mouse_allen_brain      # ~4 min
python supp_exons_vs_atses.py --dataset human_allen_brain      # ~3 min
python supp_exons_vs_atses.py --dataset rna_atlas              # ~2 min
```

This script is single-stage (no cache); each run re-reads its inputs. `mouse_tms` and
`mouse_allen_brain` read the *same* pooled mouse object and subset it with
`subset=("dataset", <value>)`, recomputing junction detection within the subset so counts
reflect that modality alone.

> `DEFAULTS` also contains a `mouse_smartseq2` entry that runs the pooled mouse object without
> subsetting. **It is deliberately not shipped.** Tabula Muris Senis brain is 1.7 % neurons
> (248) against Allen's 97.3 % (53,106), so a pooled mouse figure compares TMS microglia with
> Allen neurons. The entry is kept only for diagnostics.

Every path is overridable: `--splice`, `--ge`, `--gtf`, `--outdir`.

---

## 3. Outputs

All under `data_quality/output/`. `save()` from `leaflet_theme` writes a vector PDF (submission)
and a 600 dpi PNG (review) for every figure.

| file | built | what |
|---|---|---|
| `supp_dataset_coverage_mouse_smartseq2.{pdf,png}` | 13:56 | 8 panels, mouse, cells vs nuclei |
| `supp_dataset_coverage_mouse_smartseq2_summary.csv` | 13:56 | 458 rows, long format |
| `METHODS_dataset_coverage_mouse_smartseq2.md` | 13:56 | generated methods draft |
| `supp_dataset_coverage_human_allen_brain.{pdf,png}` | 13:57 | 8 panels, human, single modality |
| `supp_dataset_coverage_human_allen_brain_summary.csv` | 13:57 | 220 rows |
| `METHODS_dataset_coverage_human_allen_brain.md` | 13:57 | generated methods draft |
| `supp_exons_vs_atses_rna_atlas.{pdf,png,csv}` | 21:24 | 6 panels + per-gene table |
| `supp_exons_vs_atses_mouse_tms.{pdf,png,csv}` | 21:27 | 6 panels + per-gene table |
| `supp_exons_vs_atses_mouse_allen_brain.{pdf,png,csv}` | 21:31 | 6 panels + per-gene table |
| `supp_exons_vs_atses_human_allen_brain.{pdf,png,csv}` | 21:23 | 6 panels + per-gene table |

Companion prose, hand-written and checked against the shipped CSVs:
`FIGURE_LEGENDS_exons_vs_atses.md`, `METHODS_exons_vs_atses.md`.

The coverage summary CSV is long format — one row per
`(dataset, level, group_by, group, metric, stat)` — so mouse and human rows stack with
`pd.concat` and no column alignment. `level` ∈ {cell, junction, atse, gene};
`stat` ∈ {total, median, q25, q75, p10, p90, mean, min, max, log2_ratio}.

---

## 4. Verification checkpoints

A rerun that does not reproduce these has picked up the wrong input.

**Coverage, mouse** — 132,706 cells / 79,768 junctions / 24,475 ATSEs / 11,233 genes;
median junctions detected per cell 3,349 (IQR 1,454–7,423; min exactly 500, the filter floor;
max 13,378); by modality, median 1,667 in cells vs 8,034 in nuclei.

**Coverage, human** — 43,442 nuclei / 97,144 junctions / 27,082 ATSEs / 10,327 genes;
median junctions per nucleus 2,999; median ATSEs per nucleus 2,123; median junction-read
fraction 0.033; median relative junction position 0.42.

**Coverage, panel g (the headline result)** — log2(nuclei / cells):

| stratum | *n* | junctions detected | junction-read fraction |
|---|---|---|---|
| pooled, unmatched | 132,706 | **+2.27** | −1.02 |
| ASTROCYTE | 790 | +0.34 | −0.97 |
| ENDOTHELIAL CELL | 262 | +0.02 | −0.96 |
| MICROGLIA | 3,065 | −0.04 | −0.96 |
| Neuron* | 486 | +1.08 | −0.90 |
| OLIGODENDROCYTE | 683 | −0.27 | −1.04 |
| PERICYTE | 132 | −0.48 | −1.04 |

The pooled 4.8× breadth advantage of nuclei is composition plus depth and collapses under
matching; the ~2× lower spliced-read fraction is real and near-identical in every stratum. Both
must appear in the panel — publishing either alone is wrong.

**Events per gene** — genes analysed (rows with an ATSE, which is what every panel plots):

| dataset | rows in CSV | with ATSE | med junc/gene | med ATSE/gene | ρ exons~ATSEs (partial) |
|---|---|---|---|---|---|
| mouse_tms | 7,879 | 7,869 | 6 | 2 | +0.47 (+0.46) |
| mouse_allen_brain | 9,075 | 9,064 | 6 | 2 | +0.50 (+0.50) |
| human_allen_brain | 10,003 | 10,003 | 7 | 2 | +0.53 (+0.47) |
| rna_atlas | 13,883 | 13,847 | 7 | 2 | +0.60 (+0.57) |

---

## 5. Implementation notes that are load-bearing

Each of these was measured, and each fails **silently** if ignored.

1. **Never read `cell_by_cluster_matrix`.** Its value is a broadcast of the summed junction
   reads within that ATSE for that cell. Summing along axis 1 double-counts by *k* per event —
   mouse comes out ~2.4× too high and looks entirely plausible. Verified, then skipped: 17.5 GB
   of I/O avoided.
2. **`X` duplicates `cell_by_junction_matrix` byte for byte.** Scanning both doubles runtime for
   nothing.
3. **`event_id` is not contiguous along var order** — 24,583 runs for 24,475 mouse events
   (27,364 vs 27,082 human). Counting change-points within a CSR row gives a wrong
   ATSEs-per-cell with no error. Sort-unique on `(row, event)` keys is required.
4. **`np.add.reduceat` returns `data[offset]`, not 0, for empty rows.** The 500-junction floor
   means it never bites on these datasets — which is exactly why it will bite on the next one.
   Guarded unconditionally in `_h5ad_scan.py`.
5. **`num_junctions` ≠ `n_junctions_in_atse`** — they disagree for 28.2 % of junctions. The
   former is the pre-regrouping ATSE size. Use the latter; the script asserts it equals the
   observed `groupby('event_id').size()`.
6. **Never name a DataFrame column `ge`, `min`, `max`, `size`, `sum` or `T`** — `df.ge` is
   `DataFrame.ge`. This bit once during development (`ne` → renamed `n_end`).
7. **Do not use `read_h5ad(backed="r")` then `.X.sum(axis=1)`** — backed mode materialises the
   whole 30 GB. Raw `h5py` for matrices, anndata for `obs`/`var` only.
8. **Strip whitespace from every object/category obs column generically.** Duplicated
   categories are not confined to `subtissue` (`'Cortex'`/`'Cortex '`, `'Endomucin'`/
   `'Endomucin '`).
9. **The `minCells` reference line is nominal, not exact** — 152 mouse junctions sit below 75
   cells (observed min 31) because the junction filter preceded the cell filter. The line is
   drawn labelled "nominal" with the observed minimum printed; an unremarked reference line
   that the data crosses is worse than no line.
10. **Cache invalidation is the real risk, not compute.** A 74 s scan is cheap; a figure
    silently built from the superseded 06-13 file is not. The manifest mtime/size check is not
    optional.

Free quantities, no matrix read: junctions detected per cell and genes detected per cell both
come from `np.diff(indptr)` (0.1 s). Column aggregates use `np.bincount`; row sums use
`np.add.reduceat`. One chunked pass over the junction layer at `chunk_rows = 20000` costs ~74 s
and ~2 GB RSS for mouse.

---

## 6. Known issues in the shipped figures

**RNA atlas, single-exon ATSEs (4 genes).** GENCODE v45 annotates these with one exon and one
transcript, yet each carries a two-junction ATSE, so they populate the leftmost bin of panel a:

| gene_id | symbol | type | verdict |
|---|---|---|---|
| ENSG00000237424.1 | *FOXD2-AS1* | lncRNA | novel upstream exon, intergenic |
| ENSG00000260077.1 | — | lncRNA | novel upstream exon, intergenic |
| ENSG00000277977.1 | — | unprocessed_pseudogene | **junctions belong to *ZNF880*** |
| ENSG00000280013.1 | — | artifact | `gene_type "artifact"` |

In all four the two junctions share one splice site pinned to the 5′-most boundary of the
annotated exon and reach two different distal sites in unannotated flanking sequence — an
unannotated first exon with two competing donors. For ENSG00000277977 both distal sites fall
inside *ZNF880* (protein-coding, same strand), so the ATSE is attributed to a 97 bp pseudogene
when it belongs to *ZNF880*. Half the bin is therefore annotation trouble rather than biology.
**To be cleaned in the RNA atlas object; the bin should be suppressed or the genes dropped
before submission.** The other three datasets have an empty single-exon bin.

The shared splice site sits on the annotated gene boundary ±1 (exactly at it for two genes, off
by one for the other two) — the intron-boundary vs exon-boundary convention mismatch that is
also the hard part of the pending reference-ATSE catalog task (WORKFLOW step D6).

**Zero-ATSE rows in the CSVs.** Each `supp_exons_vs_atses_*.csv` is written from the full
expressed-gene table and includes rows with `has_atse = False` — 36 atlas, 11 mouse nuclei,
10 mouse cells, 0 human. These are *not* junction-free genes: `09b_filter_aligned.py` aligns the
GE and splice objects on gene **symbol**, whereas this analysis joins on Ensembl **gene_id**,
and ~125 symbols map to more than one gene_id in GENCODE v45 (chiefly the `ENSG000002910xx`
pseudogene and lncRNA identifiers new in v45). All 36 atlas symbols are present in the splice
object under a different gene_id. Every panel is drawn from the `has_atse = True` subset, so no
plotted quantity is affected. **Filter on `has_atse` before reusing the CSV.**

**Junction-read fraction is not a library fraction.** The denominator is the aligned GE object,
which holds only ATSE-bearing genes (11,233 mouse / 10,327 human), not the transcriptome. The
axis, legend, CSV `unit` and methods text all read *"fraction of reads in ATSE-bearing genes"*.
The quantity is biased upward relative to a true library fraction; do not relabel it.

**`dataset` is confounded in the mouse object.** It is aliased with age (all Allen cells are
2m), with organ (Allen contributes 6 brain tissues only) and with library chemistry (SMART-seq
v4 vs SMART-seq2). The matched strata in panel g control the first three but not chemistry, so
residual differences are an **upper bound** on a modality effect. This sentence must survive
into the caption.

**Gene read depth is not comparable across the four event-per-gene figures.** Marginal ρ against
ATSEs detected ranges −0.18 (mouse cells) to +0.38 (human nuclei). The ordering does not follow
gene count or depth range, and the four GE objects come from different quantification pipelines,
so gene read depth is not the same measured quantity in each. Report depth within a dataset; do
not rank datasets by it. Exon count, by contrast, is stable (partial ρ 0.46–0.57 in all four).

---

## 7. Open items

- [ ] **Commit both scripts.** At build time `supp_dataset_coverage.py`, `_shared/`, `README.md`
      and `FIGURE_LEGENDS_exons_vs_atses.md` were untracked and `supp_exons_vs_atses.py` was
      modified-and-moved. The `109b366` revision recorded in the cache manifests therefore does
      **not** contain the code that produced these figures.
- [ ] Clean the four artifact/misattributed single-exon ATSEs from the RNA atlas object, then
      rebuild `supp_exons_vs_atses_rna_atlas`.
- [ ] Consider suppressing panel-a bins below ~20 genes. Only the atlas single-exon bin (*n* = 4)
      is affected; the next-smallest bin anywhere is 118.
- [ ] Human RNA-atlas equivalent of the coverage figure is not built; `supp_dataset_coverage.py`
      has no `rna_atlas` entry yet.
