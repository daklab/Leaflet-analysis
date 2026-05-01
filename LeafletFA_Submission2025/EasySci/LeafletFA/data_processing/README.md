# EasySci LeafletFA Data Processing

End-to-end pipeline for processing EasySci single-nucleus RNA data (mouse brain) into
pseudobulk metacell splicing anndatas for Leaflet-FA model training.

---

## Overview

```
00_raw_data_pipeline/   Raw BAM processing per cell (RH and DT reads)
01_Metacell_analysis/   Define metacells; merge RH BAMs into pseudobulk
02_snakemake/           Leaflet junction calling on RH pseudobulk BAMs
03_DT_to_expression_counts/  Gene count matrices from DT BAMs → anndata
ATSE_ANNDATA/           Map junctions to ATSEs; build per-metacell anndatas
```

---

WD: /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells

## Step 0: Raw data pipeline — `00_raw_data_pipeline/`

Per-cell BAM processing from raw EasySci reads. Run scripts in order:

| Script | What it does |
|--------|-------------|
| `1.0-easysci.sh` | Initial alignment / demux |
| `1.1-easysci-prepBAMs.sh` | Prepare per-cell BAMs |
| `1.2-easysci-addCBs.sh` | Add cell barcodes to BAMs |
| `1.3-easysci-cleanup.sh` | Cleanup intermediate files |
| `1.4-get_pseudobulks.sh` | Build initial pseudobulk BAMs |
| `1.5-get_junctions.sh` | Extract junctions per cell |
| `1.6-get_junctions_pseudobulk_level.sh` | Aggregate junctions to pseudobulk level |

---

## Step 1: Metacell analysis — `01_Metacell_analysis/`

Defines metacells from DT gene expression (KMeans clustering), maps paired RH cells,
and merges RH BAMs into one pseudobulk BAM per metacell.

### 1a. Define metacells + produce mapping tables

```bash
# Run all five sizes (100, 500, 1000, 2000, 4000) sequentially:
python run_metacell_sizes.py

# Or a specific size:
python 00_MetaCellAnalysis.py --size 2000 --suffix _2000
```

**Outputs** (per size suffix, written to the LeafletFA working directory):
- `DT_cells{suffix}.csv` — DT cell → metacell assignment
- `DT_w_RH_cells_anndata_meta{suffix}.tsv` — full cell table with both DT + RH BAM filenames per metacell (input for step 1b)
- `EASYSCI_meta_data{suffix}.tsv` — one row per metacell (cell type, sex, n_cells, ...)
- `metacell_RT_anndata{suffix}.h5ad` — gene expression anndata aggregated to metacell level

### 1b. Merge RH BAMs into pseudobulk per metacell (SLURM array)

```bash
# Run from 01_Metacell_analysis/ or anywhere with the full path:
METACELL_SUFFIX=_500  bash /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/run_MetaCellMakePseudobulk_RH_DT.sh
METACELL_SUFFIX=_100  bash /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/run_MetaCellMakePseudobulk_RH_DT.sh
METACELL_SUFFIX=_1000 bash /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/run_MetaCellMakePseudobulk_RH_DT.sh
```

- The wrapper computes N from the TSV and calls `sbatch --array=1-N 01_MetaCellMakePseudobulk_RH_DT.sh`
- **Never call the worker script directly** — use the wrapper
- No conda needed — uses `module load samtools`
- SLURM logs → `LeafletFA/slurm/%A_%a.out` / `%A_%a.err`
- **Outputs**: `MetaCells/SpliceVI{suffix}/{date}/RH/<metacell>/<metacell>.bam`

### 1c. QC and fix missing metacells

All commands run from the MetaCells working dir:
`/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells`

**Check what's missing** (pass the base dir — auto-scans all date subdirs combined):
```bash
METACELL_SUFFIX=_500 bash /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/02_double_check_metacells_missing.sh SpliceVI_500
```
Writes `csv_clusters.txt`, `rh_with_bams.txt`, and `missing_rh_from_csv.txt` to the most
recent date subdir. Prints N missing and the exact `sbatch` command to run next.

**If N_MISSING > 0:** update `--array=1-N` in `03_remake_missing.sh`, then run the
printed command, which looks like:
```bash
METACELL_SUFFIX=_500 \
METACELL_OUTPUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI_500/20260301 \
sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/03_remake_missing.sh
```
New BAMs land in the most recent date dir. `03_remake_missing.sh` logs only the first 5
tasks; the rest are suppressed to save disk space.

**After `03` finishes, re-run `02` to confirm N_MISSING = 0:**
```bash
METACELL_SUFFIX=_500 bash /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/01_Metacell_analysis/02_double_check_metacells_missing.sh SpliceVI_500
```
Repeat resubmit → verify until missing = 0.

---

## Step 2: Junction calling — `02_snakemake/snakemakeRH/`

Runs regtools junction extraction on the RH pseudobulk BAMs produced in step 1b.
**No manual YAML editing needed** — `run_snakemake.sh` derives all paths from `METACELL_SUFFIX`
and passes them to snakemake via `--config` overrides.

BAMs may be spread across multiple date subdirs (e.g. `20260228/` + `20260301/`);
the script auto-aggregates them into `SpliceVI{suffix}/combined_RH/` via symlinks.

```bash
# Submit from anywhere — just set METACELL_SUFFIX:
METACELL_SUFFIX=_500 sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH/run_snakemake.sh
METACELL_SUFFIX=_100 sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH/run_snakemake.sh
```

- Max 64 concurrent SLURM jobs (`-j 64` in snakemake)
- The master job itself needs to stay alive for the duration — submitted via `sbatch`, runs up to 5 days
- If the master job is killed before all samples finish, resubmit the same command — snakemake will skip already-completed samples (`--rerun-incomplete` handles partial outputs)

**Outputs**: `MetaCells/SpliceVI{suffix}/junctions/{sample}/junctions.bed` etc.
**SLURM logs**: `MetaCells/SpliceVI{suffix}/slurm/{date}/slurm-{jobid}.out`

Key files: `Snakefile`, `Easysci.yaml` (static config: gtf, regtools path), `cluster.json` (SLURM resources).

---

## Step 3: DT gene counts → anndata — `03_DT_to_expression_counts/`

Converts DT per-cell BAM counts into a gene expression anndata (separate from the
metacell-level anndata produced in step 1a).

```bash
# Edit config.yaml for paths, then:
bash run_DT_gene_counts_slurm.sh   # submits SLURM array
# After completion:
python counts_to_anndata.py
```

---

## Step 4: ATSE mapping + anndata assembly — `ATSE_ANNDATA/RH/`

Maps Leaflet junction calls to ATSEs and builds per-metacell splicing anndatas.

```bash
# 1. Map junctions to ATSEs (SLURM, one job per metacell):
#    See ATSEmap_SLURM/ for submission scripts
python 04_ATSEmapping.py

# 2. Build per-metacell anndata (SLURM array):
bash 05_AnndataMake_slurm.sh   # submits 05_AnndataMake.py per metacell

# 3. Merge all metacell anndatas into one:
python 06_MergeAnndatas.py
```
