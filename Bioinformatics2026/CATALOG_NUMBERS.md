# Catalog numbers: where 142,315 / 89,831 / 28,015 and 132,706 / 79,768 / 24,475 come from

2026-11-01. Read-only trace (h5py on obs/var only; logs). Section 1 was re-read directly from the two h5ad files; sections 2–4
come from the logs named and from matching obs IDs across files.
`B` = `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT`.

## Bottom line
- **The two sets are two different builds of the catalog.** Set 2 is not a filtered version of set 1.
  - **Set 1** (SpliceVI Methods "AS and GE Quantification"): the **October 2025 build** (`B/102025/`).
  - **Set 2** (LeafletFA): the **May–June 2026 build** (`B/20260526/`, files dated 2026-06-17).
- **Both papers' models were trained on the 2026 build.** Every TMS_ONLY / AB_ONLY train/test file is 2026, and so are
  the LeafletFA fits and the validation.
- **The SpliceVI text currently mixes the builds.** Its catalog paragraph is 2025; its split paragraph
  (54,694 / 23,441 / 38,199 / 16,372) is 2026.
- **The 9,609-cell gap is not the omitted Allen cells.** It is TMS −10,411 and Allen +802 between builds. None of the
  9,400 omitted Allen cells is in any 2026 file.

## 1. Each number and its file

| Number | Build | File | How read |
|---|---|---|---|
| 142,315 cells (TMS 88,546 / Allen 53,769) | 2025 | `B/102025/model_ready_aligned_splicing_data_20251009_024406.h5ad` (same in `model_ready_gene_expression_data_20251009_024406.h5ad`) | n_obs; `obs.dataset` |
| 89,831 junctions | 2025 | same | n_var |
| 28,015 ATSEs | 2025 | same | unique `var.event_id` |
| 10,572 genes | 2025 | same | unique `var.gene_name` (= unique `gene_id` = GE n_var) |
| **132,706 cells (TMS 78,135 / Allen 54,571)** | 2026 | `B/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad`; identical in `ge_adata_…`, `model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu`, `train_splice_model_ready_2026-06-17.h5ad` | n_obs; `obs.dataset` |
| **79,768 junctions** | 2026 | same | n_var |
| **24,475 ATSEs** | 2026 | same | unique `var.event_id` |
| **11,233 genes** | 2026 | same | unique `var.gene_name` = GE n_var |

- **Genes:** the 2026 splicing var has **11,236 unique Ensembl gene IDs** but **11,233 unique gene names**, because 3 names
  map to two IDs each.
  - Quote 11,233 as "genes".
  - `supplemental/data_quality/RUN_LOG.md` line 33 says 11,236; line 164 and the summary CSV say 11,233.
- **The 2026 numbers are also in the filter log** `B/20260526/filter_aligned_rerun_17388363.out`:
  - "Final splice: (132706, 79768)"
  - "ATSEs retained: 24,475"
  - "GE genes: 13,119 → 11,233"
- **2025 is labeled the old build** in `B/version_comparison_102025_vs_20260526.txt` (142,315 × 89,831).

## 2. Filter chain of the 2026 build (the one to cite)

Code: `GenomeBiology_Submission2026/pipelines/04_align/align_splice_ge.py`, then `09b_filter_aligned.py`, run with
`--min-cells-per-junction 75 --min-junctions-per-cell 500` (the script default is 25).

| Step | Cells (TMS / Allen) | Junctions | ATSEs | Genes | Log |
|---|---|---|---|---|---|
| Frozen input list (2026-05-11) | 169,509 files (106,205 / 63,304) | – | – | – | `pipelines/01_junctions/README.md` |
| Splice AnnData | 169,501 (2 zero-read cells dropped) | 142,896 | – | – | `align_16554847.out` |
| Cells in both splice and GE | 165,968 (106,197 / 59,771) | – | – | – | same |
| Genes common to splice and GE | 165,941 | 111,903 | – | 13,119 | same |
| Cell QC (≥ 5 junctions; ≤ median + 5 MAD for junctions, reads, GE counts) | **150,906** (96,331 / 54,575) | 111,903 | 31,993 | 13,119 | same; `alignment_summary_2026-05-27.txt` |
| Junctions in ≥ 75 cells | 150,906 | 88,833 | – | – | `filter_aligned_rerun_17388363.out` |
| ATSEs re-derived after filtering (9,065 singleton junctions dropped) | 150,906 | **79,768** | **24,475** | – | same |
| Cells with ≥ 500 detected junctions | **132,706** (78,135 / 54,571) | 79,768 | 24,475 | **11,233** | same |

**Superseded run, not to be cited:** `filter_aligned_17183325.out`, files ending `_2026-06-13`. ATSEs were not
re-derived, giving 83,593 junctions / 25,750 ATSEs / 132,875 cells.

**21-month TMS cells (traced 2026-11-04):** the 728 FACS cells (all Mammary_Gland, mice 21_48_F, 21_54_F, 21_55_F) are removed on purpose at metadata preparation: `exclude_ages: ["21m"]` (`GenomeBiology_Submission2026/config/mouse_smartseq2.yaml:61`, applied at `03_gene_expression/00_prepare_metadata.py:114-119`). The junction folders hold only 3, 18 and 24 months, and the frozen input list has no 21-month file. The reason for the exclusion is not recorded.

**2025 build, partly traceable:**
- **Starting point:** 165,950 aligned cells × 113,995 junctions (`B/102025/dataset_summary_20251003_004515.txt`).
- **End point:** 142,315 kept cell IDs (`filtered_cell_ids_20251003_152725.txt`).
- **The filter script survives only in git:** `Mouse_Splicing_Foundation/GeneExpression/04_check_outliers.py`, commit
  `861f9ef`.
- **No log** for the junction step (113,995 → 89,831) or for the 28,015 ATSE definition.

## 3. The 9,609-cell gap

| | 2025 build | 2026 build | Change |
|---|---|---|---|
| TMS | 88,546 | 78,135 | **−10,411** |
| Allen | 53,769 | 54,571 | **+802** |
| Total | 142,315 | 132,706 | **−9,609** |

- **The 9,400 omitted Allen cells** (`supplemental/data_quality/allen_omitted_cells/`) have **0** overlap with the 2026
  aligned or filtered sets.
  - They were lost upstream of both 2026 analyses: a glob missed per-cell folders (branch `allen-omitted-cells`, `58265c3`).
  - So 9,400 and 9,609 are unrelated numbers.
- **Allen cells, matched across builds by sample name:**
  - 51,918 in both builds
  - 1,851 only in 2025
  - 2,653 only in 2026
- **Not determined:**
  - whether the 9,400 omitted cells were in the 2025 build (their SRR IDs are not mapped to sample names);
  - a per-cell TMS overlap (the two builds use different TMS cell-ID formats).

## 4. Training splits (2026; both papers)

| File (`B/20260526/…_2026-06-17.h5mu`) | Cells |
|---|---|
| `train_70_30_TMS_ONLY_…` | 54,694 |
| `test_30_70_TMS_ONLY_…` | 23,441 |
| `train_70_30_AB_ONLY_…` | 38,199 |
| `test_30_70_AB_ONLY_…` | 16,372 |
| pooled `train_70_30_…` / `test_30_70_…` (GB atlas fits) | 92,894 / 39,812 |

- **Coverage:** the per-atlas splits together contain all 132,706 cells, with no overlap.
- **Independent splits:** the per-atlas splits are independent re-splits, not subsets of the pooled split. Only 38,301
  of 54,694 TMS-only training cells are in the pooled training set.
- **Junctions and ATSEs:** every file has 79,768 junctions and 24,475 ATSEs.

## 5. Proposed sentence for both papers

> Splicing events were defined with ATSEmapper from Smart-seq2 data of Tabula Muris Senis and the Allen mouse cortex and
> hippocampus atlas. After cell and junction filtering, the catalog contains 79,768 junctions in 24,475 alternative
> transcript structure events (ATSEs) across 11,233 genes, measured in 132,706 cells (78,135 Tabula Muris Senis cells
> and 54,571 Allen nuclei). Each atlas was split 70/30 into training and test cells (Tabula Muris Senis 54,694 / 23,441;
> Allen 38,199 / 16,372).

- **SpliceVI change:** replace 88,546 / 53,769 / 142,315 / 89,831 / 28,015 / 10,572 with the numbers above.
- **LeafletFA:** this sentence goes in Data; the filter chain (section 2) goes in the Supplementary Methods.
- **Ledger:** these numbers go into the ledger with the source md5 of the 2026 files.
