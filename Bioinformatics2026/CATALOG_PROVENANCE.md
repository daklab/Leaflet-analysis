# Provenance of the June 2026 mouse splicing catalog (both papers)

2026-11-01. Read-only tracing plus counting from obs/var and existing logs; nothing was re-processed. Extends
`CATALOG_NUMBERS.md` (two builds) and `DEFINITIONS.md` (rules with file:line). Every statement has a source; untraceable
items are in **Gaps** (section 7). Flow table as TSV: `paper/supplementary/table_catalog_flow.tsv`.

**Abbreviations**
- WD = `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION`
- B = `WD/MODEL_INPUT/20260526`
- P = `WD/ATSE_mapper/junction_processing_20260511`
- RAW = `/gpfs/commons/projects/knowles_singlecell_splicing`
- GB = Leaflet-analysis `GenomeBiology_Submission2026/`

**Logs** (md5 in parentheses)
- L = `P/logs/atsemap_15915608.out` (73219e21d7d7819bea2e3ef8c640bdb1)
- M = `P/logs/junc_merge_15915229.out` (244fda9935568cd410dd2e198b851c2f)
- A = `B/align_16554847.out` (7b0bdf56aed9e977645f70f98184a99c)
- F = `B/filter_aligned_rerun_17388363.out` (36bd9ac070738d08dd95df1d6f59bdeb)

## 1. Raw data

| Dataset | Source | Start | Alignment | Cells in catalog | Mice / donors | Notes |
|---|---|---|---|---|---|---|
| **TMS Smart-seq2** (FACS) | `s3://czb-tabula-muris-senis/` (`Archive/LeafletFA_Submission2025/TabulaSenis/data-installation/001_AWS_setup.sh:5-19`) | consortium **BAMs** `*.gencode.vM19.ERCC.Aligned.out.sorted.bam` (`SS2_Snakemake/MONTH*/Snakefile:33`; BAM-only pull `001_AWS_setup.py:30-41`) | not re-aligned; the consortium's aligner and settings are not in our files (Gap) | 78,135 cells | 18 mice (10 at 3 m, 4 at 18 m, 4 at 24 m; obs `mouse.id`) | 23 tissues, 120 `cell_ontology_class`; ages 3 m 32,618 / 18 m 24,009 / 24 m 21,508; 21 m never listed (no 21 m folder in the input list) and excluded in config (`mouse_smartseq2.yaml` `metadata.exclude_ages`) |
| **Allen mouse CTX + HPF Smart-seq v4** | GEO **GSE185862**, SRA PRJNA772116 (`RAW/allen-brain/mouse_isocortex_hippocampal_2021/README`, `SRA/SraRunTable.txt`) | **FASTQ** (`fastq-dump --gzip --split-files`, `mouse_brain_dev_2021/dataprocessing/01_download_data.sh:37`) | **STAR 2.7.10b**, `--twopassMode Basic --outSAMstrandField intronMotif --outSAMattributes XS --outSAMtype BAM SortedByCoordinate` (`BAM/SRR16389942/Log.out:1,5`); index GRCm38 + GENCODE vM19 (`star/genomeParameters.txt:1`; `genes.gtf` "GRCm38, version M19 (Ensembl 94)") | 54,571 | 519 donors (obs) | 22 regions (`subtissue`); 381 cluster labels; age set to "2m" for all cells by config (`sources.allen_brain.age`), not from data |
| Human Allen brain (supplement and portal only) | `gs://lein-human-cortex/.../SSv4/human_gru/processed/*` (`BRAIN_NeMO/lein-human-cortex/download_data.sh`) | FASTQ | STAR 2.7.10b, GRCh38 + GENCODE v45 | 43,442 nuclei | 3 donors | 6 regions; separate catalog (97,144 junctions, 27,082 ATSEs, 10,327 genes) |
| Bulk RNA atlas GSE138734 (supplement and portal only) | `RNA_ATLAS/GSE138734/fastqs/manifest.csv` | FASTQ | STAR 2.7.10b, GENCODE v45, plus `--outFilterIntronMotifs RemoveNoncanonicalUnannotated` (`07_conservation/Snakefile:56-67`) | 2,228 cell-type samples (316 GSMs) | — | both polyA and total-RNA GSMs listed (Gap) |

**Annotation used for the mouse catalog: GENCODE vM19** (GRCm38).
- **Sources:** `GB/config/mouse_smartseq2.yaml:9` (`gtf_file .../gencode.vM19/genes/genes.gtf`) and `:11` (FASTA); the run log
  L:10 creates the annotation database from that GTF.
- **The PacBio long-read GTF (Schertzer 2025) was not used for the June 2026 mouse catalog.**
  - Long-read GTFs appear only in the human Smart-seq2 and PBMC configs.
  - The SpliceVI Methods sentence citing it, and our earlier Methods draft, are wrong for this build (fixed in our draft; to fix in SpliceVI).

**Cell metadata and labels**
- **TMS:**
  - `RAW/TabulaSenis/data/AWS/metadata/tabula-muris-senis-full-metadata.csv`, FACS rows (`00_prepare_metadata.py:48`).
  - Columns `cell_ontology_class`, `tissue`, `subtissue`, `mouse.id`, `sex` copied (`:29-54`).
  - `tissue_cell_type` = tissue + `_` + cell_ontology label.
- **Allen:**
  - `cell_ontology_class` = `cell_type_alias_label` from `GSE185862_metadata_ssv4.csv.gz`, which equals `cluster_label` for all
    rows; `subtissue` = `region_label` (`00_prepare_metadata.py:60-80`).
  - Linked by SRR through `expected_R1_fastq` → `sra_s3_links.txt` → `SraRunTable.Run`.
  - Matched to expression by `exp_component_name` (`align_splice_ge.py:83-102`).
- **Allen "cell groups (cluster names)" in check 10 and SpliceVI §111** (Karin, 2026-11-01: called cell groups, not subclasses):
  the cluster label without its number.
  - This matches the official `subclass_label` for **78.7 %** of the 73,347 metadata rows: **70 groups vs 42 official
    subclasses** (verified here; `RAW/.../METADATA/metadata.csv` md5 4bba668ad677a8d1905239e04ea46a9f). Examples:
    `204_L5/6 IT CTX` vs `L5 IT CTX`; `59_Vip Igfbp6` vs `Vip`.
  - `subclass_label` is not carried into the model-ready obs.
  - Each of the 382 cluster labels maps to exactly one official subclass, and only one of the 70 cell groups ("Vip") spans
    two subclasses, so the mismatch is mostly naming (e.g. cell group `L5/6 IT CTX` is in subclass `L5 IT CTX`), not
    mixing.
  - **Decided (Karin, 2026-11-01):** check 10 keeps the 70 cell groups as pre-registered; the official subclass is an extra
    column in the catalog export and the portal (`LeafletFA scripts/validation/c10_allen_subclass.py`,
    `tsv/c10_Allen_group_subclass.tsv`).
- **Dataset label:** by age; age 2 m → `allen_brain` (`align_splice_ge.py:125-127`).

## 2. Splicing data

- **regtools** (all four datasets): `regtools junctions extract -a 6 -m 50 -M 500000 -s XS -b barcodes.txt` on BAMs whose
  CB tag is the cell id (TMS `SS2_Snakemake/Snakefile:31,60`; Allen `mouse_brain_dev_2021/.../Snakefile:194`). Binary
  `/gpfs/commons/home/kisaev/regtools/build/regtools`, repository commit 2e3a291 (reports "Version 0.0.1").
- **One joint ATSEmapper run over both atlases.**
  - All 169,503 readable cell files go into one combined list (`01_junctions/01_split.sh:63-78`).
  - Chunks are merged by summing cells and reads per junction (`General_Utils/split_process_merge_slurm_junctions.py:73-87`).
  - One `final_junctions.pkl` (md5 020cbb13…) is loaded by `02_atsemapper/04_run_atsemapper.py:45-57` (L, one run).
  - **The splice graph and every pooled threshold (≥ 10 cells, ≥ 100 reads, 1 % usage) were defined by all 169,503 cells,
    before cell QC or matching to expression.**
- **Re-grouping (`04_align/09b_filter_aligned.py`):**
  - The ≥ 75-cell count and the junction scores come from the 150,906 cells after QC (`:225-237`, `:298-302`).
  - Grouping is graph connectivity within a gene with usage threshold 0.0 (`:249`), so the scores do not affect it (F: 0
    removed by usage, 0 splits).
  - **The ≥ 500-junction cell filter runs after re-grouping** (`:301-341`).

## 3. Expression data

- **Mouse: the atlases' published matrices, not our own counting.**
  - TMS: `TabulaSenis/data/AWS/processed_for_scanpy/tabulamurissenisfacsofficialrawobj.h5ad`.
  - Allen: portal `expression_matrix_exons.csv` (exons only).
  - Sources: `mouse_smartseq2.yaml` `ge_h5ad`, `ge_exons`; `03_gene_expression/01_prepare_counts.py:148,156`.
- **Gene filters:**
  - GENCODE vM19 protein-coding and lncRNA biotypes (`01_prepare_counts.py:90-105`);
  - single-exon genes dropped (`:222`);
  - inner join on gene symbol (`:201`); genes common to TMS and Allen (`:234`); names deduplicated (`:241-252`);
  - then intersected with genes that have catalog junctions (`align_splice_ge.py:561-569`; `09b:366-390`): 13,119 → **11,233**.
  - **No highly variable gene selection** in the model-ready data; HVG selection appears only in the scVI baseline
    (`02_run_scVI.py:44-56`).
- **Length normalization:**
  - length_norm = 10⁴ · (X_g / L_g) / Σ_g (X_g / L_g), with L_g = mean over the gene's GTF transcripts of summed exon
    length (`01_prepare_counts.py:69,81,110-123`).
  - The MuData layer = length_norm × median(L_g over the retained genes) (`SpliceVI-utils/preprocess/core/createtesttrainmudata.py:322-325`).
  - Library size = row sum (`:329`); median 1,832 for the 11,233 genes (`stage5_thinning_test.py:5-7`).
- **Matching expression to splicing cells:** by cell id (TMS) and `exp_component_name` through the metadata (Allen)
  (`align_splice_ge.py:83-103`).
  - 169,501 splice cells → **165,968** in both (106,197 TMS / 59,771 Allen; A:20,61-63).
  - TMS loses 8; Allen loses 3,533, all runs with no row in the combined metadata (cause in section 7; the align log gives no per-cell reason).

## 4. Cells

| Step | TMS | Allen | Total | Source |
|---|---|---|---|---|
| Cell files listed (frozen 2026-05-11 list) | 106,205 | 63,304 | 169,509 | `01_junctions/frozen_inputs/mouse_smartseq2_junction_files_2026-05-11.txt.gz` (md5 9136f2c521af8fbcb9e5c205b6445cb9) |
| Readable (6 TMS 3 m files missing) | 106,199 | 63,304 | 169,503 | `P/logs/junc_15876961_*.err`; M |
| Zero-read cells dropped | — | — | 169,501 | A:6-7 |
| In both splicing and expression (and metadata) | 106,197 | 59,771 | 165,968 | A:20, 61-63; Allen loss: 3,533 runs without a metadata row (R2-named links, section 7) |
| Common genes; zero-read cells dropped | — | — | 165,941 | A:65-72 |
| Cell QC (≥ 5 junctions; ≤ median + 5 MAD on junctions, junction reads, expression counts) | 96,331 | 54,575 | 150,906 | A:74-92, 120-122 |
| ≥ 500 detected junctions (after the junction filter and re-grouping) | **78,135** | **54,571** | **132,706** | F:181-190; obs |

- **TMS per tissue** (listed / after QC / final): a 23-row table, in the TSV
  (`paper/supplementary/table_catalog_flow_tms_tissues.tsv`). Brain_Myeloid 12,943 / 12,372 / 11,203; Marrow 14,009 /
  13,104 / 10,202; Heart 9,438 / 8,547 / 6,986; …
- **The 21-month TMS cells:** 728 FACS cells (all Mammary_Gland, 3 mice) removed by `exclude_ages: ["21m"]` (`mouse_smartseq2.yaml:61`) before any listed step. **Reason:** too few cells for statistical analysis (Karin's decision; `GenomeBiology_Submission2026/config/mouse_smartseq2.yaml:60`, `GenomeBiology_Submission2026/WORKFLOW.md:254`).
- **The 9,400 omitted Allen cells:**
  - **Cause:** a non-recursive glob (`01_split.sh:64-66` with `junction_pattern: "*_junctions_with_barcodes.bed"`,
    `mouse_smartseq2.yaml:16-17`) matched the 63,304 flat files and missed the cells stored only as per-cell folders
    (72,704 aligned − 63,304).
  - **Where they drop out:** before the input list, so in no model.
  - **Documented** on branch `allen-omitted-cells` (58265c3, merged d6f6113), `01_junctions/README.md:14-28`,
    `figures/supplemental/data_quality/allen_omitted_cells/`.
- **70/30 splits:**
  - Script: `SpliceVI-utils/preprocess/core/createtesttrainmudata.py`; `sklearn.train_test_split`, **seed 42**
    (`random_state`, `:192-195`, `:466-475`), per cell.
  - **Stratified by broad cell type × age** (`:444-470`; strata with < 2 cells pooled as "OTHER"); **not grouped by mouse**
    (test mice also appear in training).
  - Allen is all "2m", so effectively by broad type.
  - Per-atlas re-splits: TMS 54,694 / 23,441 (job 19457949), Allen 38,199 / 16,372 (job 21438648); pooled 92,894 / 39,812
    (job 17395352). Logs in Smriti's `script_outputs/logs/pipeline/step1_create_mudata/`.

## 5. Junction, ATSE and gene flow (both atlases pooled; per-atlas where countable)

| # | Step | Rule (source) | Cells | Junctions | ATSEs | Genes |
|---|---|---|---|---|---|---|
| 1 | Parse | intron 50–500,000 bp; standard chromosomes (`junction_parser.py:57-62`) | 169,503 | 23,673,431 unique | — | — |
| 2 | Pooled junction QC | ≥ 10 cells and ≥ 100 reads, both atlases summed (M; `mouse_smartseq2.yaml:37-38`) | 169,503 | 828,510 | — | — |
| 3 | Canonical motif | GT-AG, GC-AG, AT-AC (L; `04_run_atsemapper.py:69`) | | 762,806 | — | — |
| 4 | Multi-gene removed | 13,309 junctions annotated to > 1 gene (L; `genome_utils.py:366-370`) | | 749,497 | — | — |
| 5 | No gene | junctions with no matching transcript end left out of the graph (L) | | 469,294 | — | 23,711 |
| 6 | Splice-site usage | either share < 0.01 removed: 189,308 (L) | | 279,986 | — | — |
| 7 | Singletons | 137,090 dropped (L); ATSE file `MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz` (md5 dbdc000f…) | | **142,896** | **39,729** | 17,598 |
| 8 | Match to expression and genes | common cells and gene names (A:65-72, 92) | 165,941 | 111,903 | 31,993 | 13,119 |
| 9 | Cell QC | median + 5 MAD (A:74-92) | 150,906 | 111,903 | 31,993 | 13,119 |
| 10 | Junction in ≥ 75 cells | counted over the 150,906 cells (F:10-14) | 150,906 | 88,833 | — | — |
| 11 | Re-grouping | connected components, usage 0.0; **9,065 singletons** (F:35,119-121) | 150,906 | **79,768** | **24,475** | 11,236 IDs |
| 12 | ≥ 500 junctions per cell | (F:181-190) | **132,706** | 79,768 | 24,475 | **11,233** |

- **Steps 4–7:** the junction counts after steps 4 and 5 are from L. The count after step 6 (279,986) is computed as
  469,294 − 189,308, and it is consistent with step 7: 279,986 − 137,090 singletons = 142,896.
- **The 9,065 singletons of step 11** are all junctions that lost at least one original ATSE partner to the ≥ 75-cell filter
  (all 9,065 checked against the step-9 ATSE assignments; reproduces 88,833 exactly):
  - 5,240 lost every partner;
  - 3,825 kept partners but were disconnected when a bridging junction failed.
- **Per-atlas, final catalog:**
  - Junctions with ≥ 1 read: TMS 77,562; Allen 77,204; both 74,998.
  - ≥ 75 cells within one atlas: TMS 56,696; Allen 65,998; 2,314 junctions pass only when the atlases are pooled.
- **Final catalog by annotation status:** both ends 60,603; 5′ only 9,645; 3′ only 9,414; unannotated 106. The filter is "assigned to a gene", not "at least one annotated end" (corrected 2026-11-04).
- **Final catalog by motif:** GT-AG 78,827; GC-AG 926; AT-AC 15.

## 6. The two builds, and which analyses use which

- **Builds compared** (`CATALOG_NUMBERS.md` for the totals):
  - **Inputs:** the same 169,509 files (`junction_files_combined_20250929.txt` equals the 2026 list after sorting).
  - **Code:** 2025 used `LeafletFA-utils`; 2026 used the ATSEmapper package.
  - **2025 downstream filters** (`Mouse_Splicing_Foundation/GeneExpression/04_check_outliers.py`, git 861f9ef):
    - per-batch robust cell QC, rRNA < 10 %, cell types ≥ 50 cells;
    - junction reads and counts within the 1st–99th percentile, and > 1,000 junction reads;
    - lncRNA removal;
    - **ATSEs with > 10 junctions removed**;
    - no minimum-cell junction filter, no re-grouping.
  - **2026:** median + 5 MAD cell QC, ≥ 75-cell junction filter, re-grouping, ≥ 500 junctions per cell; no ATSE size cap
    (largest 43 junctions).
- **Every model fit in both papers reads the June 2026 build** (section 8 below).
- **Still on 2025:**
  - text in `SpliceVI_SuppMethods_Oct2026.tex:85`;
  - stale script defaults in SpliceVI-utils (`SpliceVI/train_splicevi.sh:32`, `eval_splicevi.sh:30-31,45-46`, and three
    others), overridden in the paper runs.

## 7. Gaps and decisions for Karin

1. ~~Allen labels~~ **Resolved (Karin, 2026-11-01):** keep the 70 cell groups (cluster names) for check 10; official subclass added
   as an extra column.
2. ~~AB masked-imputation runs~~ **False alarm (checked 2026-11-01):** the two SpliceVI AB logs
   (`pipeline_20261001_012113_s5-ab-{devp,std}.log`) list the TMS masked test file only in their config echo; both have
   `SKIP_STEP4=true` and "STEP 4: SKIPPED (no evaluation)", so no masked-imputation evaluation ran. Confirmed independently
   by the SpliceVI session (SpliceVI-utils STAGE5 §180, commit 9279810): no Allen masked-imputation number exists.
3. **Long-read GTF:** the SpliceVI Methods cite it for this build; the build used GENCODE vM19. Was `lr_GRCm38.db`
   (Oct 2025) used by the 2025 build?
4. **Allen cells vs nuclei:** no field says; the portal README says cells, SpliceVI Methods says nuclei.
5. **Allen donor ages:** set to 2 m by config, not from data.
6. **TMS consortium alignment** (aligner, version, settings) is not in our files; cite the consortium.
7. **Origin of the TMS expression h5ad** (`...facsofficialrawobj.h5ad`): no download command found.
8. ~~ATSEmapper code version~~ **Snapshot (2026-11-01):** ATSEmapper branch `run-2026-05-11-snapshot`, commit 3807a85, tag
   `atsemapper-run-2026-05-11` = c072959 plus the then-uncommitted `junction_parser.py` change. Evidence: the cpython-310
   cache of `junction_parser.py` was compiled 2026-05-11 00:57 from a 7,509-byte source (the snapshot's file; c072959's is
   7,450 bytes); `event_detection.py`, `main.py` and `genome_utils.py` were last compiled (cpython-312, 2026-05-06) from
   sources of exactly c072959's sizes. Not checkable: edits to those three between 2026-05-06 and the run.
9. **Missing 2026 side files:** `multi_gene_junctions`, `singleton_junctions`, `filtered_low_usage_junctions` were written to
   the job's working directory and were not found; counts come from L.
10. **Unexplained losses:**
    - 6 unreadable TMS files;
    - 2 + 27 zero-read cells (not attributed to an atlas);
    - ~~3,533 Allen cells~~ **cause found (2026-11-01):** all 3,533 have no row in the combined metadata, so they cannot be
      linked to expression. The metadata join (`03_gene_expression/00_prepare_metadata.py:60-80`) matches
      `sra_s3_links.txt` to the GEO metadata by the R1 FASTQ name, but for **3,375** of these runs the links file gives the
      **R2** name (`…_R2.fastq.gz.1`; 3,859 R2 entries of 73,707 links); **3,362** of them would match after replacing `_R2`
      with `_R1`, and all 3,362 are in the expression matrix. The other **158** are not in the links file at all (e.g.
      SRR16390126, GSM5629176); 13 have an R2 name that still does not match. Recoverable in a rebuild; not changed
      (no new processing).
    - per-atlas cell-QC removals not logged.
11. **Split code version:** `createtesttrainmudata.py` first entered git on 2026-09-18, after the TMS (08-02) and pooled
    (06-18) splits; the version is attested only by the log CONFIG lines. The splits are not grouped by mouse (say so in
    both papers).
12. **Computed count:** 279,986 junctions after the usage filter (step 6) is computed by subtraction (consistent with step 7), not printed in a log.
13. **Unannotated junctions with a gene:** 724 (106 in the final catalog). Gene assignment uses within-tolerance end matches,
    while annotation status uses exact matches (`genome_utils.py:288-297` vs `:376-389`). Mechanism not fully traced.
14. **Bulk RNA atlas:** both polyA and total-RNA libraries are listed, though the config says Ribo-Zero.
15. **md5s of the model-ready files over 2 GB:** recorded only where a run record computed them (section 8).

## 8. Model-ready files and readers (all June 2026 build; 79,768 junctions, 24,475 ATSEs, 11,233 genes)

| File (`B/…_2026-06-17.h5mu` unless noted) | Cells | md5 (where recorded) | Read by |
|---|---|---|---|
| `train_70_30_TMS_ONLY_…` | 54,694 | d2ceebe490b7bfe1f1487fe2aaad7424 (LeafletFA run record) | SpliceVI TMS; scVI TMS; tuVI/TRVI (via §170 prep); scQuint (via dumpcounts); LeafletFA baseline and check 8–10 fits |
| `test_30_70_TMS_ONLY_…` | 23,441 | c9149ef41fdc4592d27f42b3d26c4c8e | §170 evaluation (all embeddings); LeafletFA projections; check 1, 10 TEST |
| `train_70_30_AB_ONLY_…` | 38,199 | 3f4918928c75ae4758620108d65a424f | SpliceVI Allen; scVI Allen; tuVI/TRVI; scQuint; LeafletFA |
| `test_30_70_AB_ONLY_…` | 16,372 | f77ba7aa16c6f00e4f5a7104fdb97c84 | §170; LeafletFA |
| `SPLICINGCONTENTONLY_SEED20260926_test_30_70_{TMS,AB}_ONLY_…` | 23,441 / 16,372 | c624f836… / 0954334… | ratio-redrawing test (§170; LeafletFA) |
| pooled `train_70_30_…` / `test_30_70_…` | 92,894 / 39,812 | not recorded | GB 70/30 models and imputation evaluation |
| `train_splice_model_ready_2026-06-17.h5ad` | 132,706 | not recorded | GB full-atlas LeafletFA fits |
| `model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu` | 132,706 | not recorded | check 10.6 atlas benchmark fit |
