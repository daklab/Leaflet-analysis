# 01_junctions — per-cell junction files → pooled junction table

`01_split.sh` (list + chunk) → `02_process.sh` (parse each regtools BED; intron 50–500,000; standard chromosomes; no per-cell QC) →
`03_merge.sh` (pool over all cells; QC ≥ 100 reads and ≥ 10 cells) → `results/final_junctions.pkl` for `02_atsemapper/`.
Per-cell BEDs come from `regtools junctions extract -a 6 -m 50 -M 500000`.

## Frozen input list (used for both papers)

The cell set used by the LeafletFA (Genome Biology) atlas and the SpliceVI models is the **2026-05-11 input list**:
`frozen_inputs/mouse_smartseq2_junction_files_2026-05-11.txt.gz`. It has 169,509 files (63,304 Allen Brain + 106,205 TMS) and is
byte-identical to `…/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511/junction_files_combined_20260511.txt`
(md5 of the uncompressed list `fee8da43…`). To reproduce the published results, use this list rather than re-globbing the folders.

## Allen Brain: per-cell folders vs flat files (read before any rebuild)

- `01_split.sh` lines 64–66 build the Allen list with `glob(junction_dir + "/" + junction_pattern)`, with `junction_pattern:
  "*_junctions_with_barcodes.bed"` (`config/mouse_smartseq2.yaml` lines 16–17). That matches **flat files only**
  (`junctions/<SRR>_junctions_with_barcodes.bed`, 63,304 cells).
- Every Allen cell also has a **per-cell folder**, `junctions/<SRR>/junctions_with_barcodes.bed`: **72,704 cells**.
  - These were written by an Aug–Sep 2025 re-run of all cells; the flat files come from the original Feb–Jul 2025 run.
  - Where both exist they are byte-identical.
- **9,400 cells have only the per-cell folder**, so they never entered the pipeline. This is a file-layout gap, not a QC or cell-type rule.
  - They match the included cells in class, subclass, region and donor composition.
  - Supplementary Table SX: `figures/supplemental/data_quality/allen_omitted_cells/`.
  - Analysis: `SpliceVI-utils/manuscript/easysci_processing/joint_atse/REPORT.md` (follow-up section).
- **For a future rebuild that should include all 72,704 Allen cells:** list the per-cell folders instead (pattern
  `*/junctions_with_barcodes.bed`, with the cell id taken from the folder name). This changes the cell set, so it is a new atlas version,
  not a reproduction of the published one.
