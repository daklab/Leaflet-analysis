# LeafletFA + ATSEmapper — Genome Biology Submission 2026
## Complete Workflow

All scripts are **config-driven**: pass the dataset YAML as the first argument.
No hardcoded paths anywhere. Everything is in `config/`.

---

## ✓ Partial ATSE fix — complete for mouse and human (2026-06-17)

When `09b_filter_aligned.py` applies the junction filter (min cells per junction),
some ATSEs lose only a subset of their junctions. Keeping these "partial ATSEs" is
problematic: a removed junction may have been the connecting edge between two
sub-groups, meaning surviving junctions no longer form a coherent ATSE.

**Fix applied (2026-06-17) for both mouse and human:**
After the junction cell-count filter, `09b_filter_aligned.py` now calls
`_regroup_atses()`, which re-runs ATSE grouping from scratch on the surviving
junctions using ATSEmapper's own `ATSEAnalyzer.build_splice_graph()` +
`find_atse_groups()` (same logic as the original Step 4 ATSE mapping). Connected
components with ≥2 junctions become new ATSEs; singletons are dropped.
Event IDs are re-assigned in the original format (`gene_id_atse_N`).
The original ATSEmapper IDs are preserved in `var["original_event_id"]`.

**Impact:**
- Mouse: 88,833 → 79,768 junctions (9,065 singletons dropped); 24,475 ATSEs
- Human: 106,586 → 97,144 junctions (9,442 singletons dropped); 27,082 ATSEs

Old 2026-06-13 mouse filtered h5ads are **superseded** by the 2026-06-17 versions.
Send Smriti the 2026-06-17 files (see Step 9c outputs below).

### ⚠️ Known limitation of the shared-splice-site ATSE definition (METHODS — write this up)

**`singleton_junctions.tsv` contains real, highly-expressed constitutive junctions — NOT
noise.** Do not mine that file as a discard pile of artifacts.

Because an ATSE is defined as a connected component of junctions **sharing a splice site**,
a junction that competes with nothing shares no splice site with any other junction, forms a
component of size 1, and is dropped — *however deeply it is sequenced*.

Worked example (RNA atlas, COL1A1 = `ENSG00000108821.14`):

```
exon…50189010 |intron| 50189167‥50189275 |intron| 50189376‥50189538 |intron| 50189678…
                          ^exon A       50189275-50189376    ^exon B
                                        2,883,894 reads / 2,095 samples  -> DROPPED
```

`ENSG00000108821.14_atse_5` is a **double-exon skipping** event. Its three junctions —
two inclusion (2.94M and 2.75M reads) plus one skip (67K reads, 50189010–50189678) — all
share splice site 50189010 or 50189678. But the **internal** junction connecting exon A to
exon B (2.88M reads, 2,095/2,228 samples) shares a splice site with none of them, so it is a
singleton and never enters an ATSE. Same in `atse_4` (50189912–50190000, 1.71M reads).

**Why this is acceptable — no splicing-choice information is lost.** That internal junction
is *constitutive within the inclusion path*: it is used whenever the exon pair is included and
never when it is skipped. It therefore carries zero information about the inclusion-vs-skip
decision, and the ATSE's PSI denominator remains correct. And if a genuine alternative event
existed at that position (e.g. an alternative 5′ site), it would by definition have ≥2
competing junctions sharing a splice site and would form its own ATSE — so it would not be
missed.

**Consequences to state in the methods:**
1. ATSEs quantify splicing *choices*, not the full splice graph. Constitutive junctions are
   deliberately excluded, so junction counts in ATSEs undercount total splicing in a gene.
2. A browser view of the ATSE BED will show apparent "gaps" where a high-coverage
   constitutive junction sits — expected, not a bug.
3. Multi-exon skipping events appear as an ATSE whose skipped region contains internal
   exon–exon junctions that are absent from the ATSE.

---

## Datasets

| Label | Sources | Species | Status |
|-------|---------|---------|--------|
| `mouse_smartseq2` | Tabula Muris Senis (FACS) + Allen Brain Mouse | Mouse | ✓ Steps 1–9c complete; filtered h5ads ready (132,706 × 79,768 junctions / 24,475 ATSEs, 2026-06-17 with ATSE re-grouping fix); ⚠️ Smriti h5mu (2026-06-13) based on old files — new h5mu needed from Smriti |
| `human_allen_brain` | Allen Brain Human Cortex (lein-human-cortex, 50,625 cells) | Human | Steps 1–9c ✓ done (2026-06-17); filtered h5ads ready: 43,442 × 97,144 junctions / 27,082 ATSEs; config: `config/human_allen_brain.yaml`; genome: Gencode v45 GRCh38; next: Smriti h5mu train/test split |
| `parsebio_pbmc`   | Parse Bio PBMC public dataset | Human | Not started |

---

## Step 0: Software

```bash
# ATSEmapper v2 (renamed from leafletfa_utils)
pip install -e /gpfs/commons/home/kisaev/ATSEmapper

# LeafletFA
pip install -e /gpfs/commons/home/kisaev/LeafletFA

conda activate LeafletSC
```

---

## Step 1: Discover junction files + split into chunks

```bash
CONFIG=config/mouse_smartseq2.yaml
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
bash pipelines/01_junctions/01_split.sh $CONFIG
# Creates: $WD/ATSE_mapper/junction_processing_YYYYMMDD/
PROC_DIR=$WD/ATSE_mapper/junction_processing_YYYYMMDD
```

**mouse_smartseq2 — 2026-05-11:** 63,304 Allen Brain + 106,205 TMS = 169,509 files → 100 chunks
`PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511`

## Step 2: Process chunks (SLURM array)

```bash
N_CHUNKS=100
sbatch --array=0-$((N_CHUNKS-1))%50 \
    --output=${PROC_DIR}/logs/junc_%A_%a.out \
    --error=${PROC_DIR}/logs/junc_%A_%a.err \
    pipelines/01_junctions/02_process.sh $CONFIG $PROC_DIR
```

**mouse_smartseq2 — 2026-05-11 (resubmit after BED column fix):**
`PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511`

Before Step 3, verify all 100 chunks completed:
```bash
ls $PROC_DIR/results/chunk_*_results.pkl | wc -l   # should be 100
find $PROC_DIR/results -name "*_results.pkl" -size 5c  # should be empty
```

> Wait for all array jobs to finish, then:

## Step 3: Merge junction chunks

```bash
PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511
CONFIG=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/config/mouse_smartseq2.yaml

cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

sbatch --mem=64G --cpus-per-task=4 -p cpu \
    --output=${PROC_DIR}/logs/junc_merge_%j.out \
    --error=${PROC_DIR}/logs/junc_merge_%j.err \
    pipelines/01_junctions/03_merge.sh $CONFIG $PROC_DIR
# Produces: $PROC_DIR/results/final_junctions.pkl
```

QC thresholds applied at merge (from config `atsemapper` section):
- `min_junc_reads: 100` — junction must have ≥ 100 total reads summed across all cells
- `min_num_cells: 10`  — junction must be seen in ≥ 10 cells

> Filtering is applied **after** the full merge (not per-chunk), so a junction seen in
> 5 cells in chunk A and 6 in chunk B correctly passes min_cells=10 in aggregate.

**mouse_smartseq2 — 2026-05-12:** 828,510 junctions before QC → `final_junctions.pkl`

## Step 4: Map ATSEs

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

sbatch --mem=300G --time=2-00:00:00 -J atsemap -p bigmem \
    --output=${PROC_DIR}/logs/atsemap_%j.out \
    --error=${PROC_DIR}/logs/atsemap_%j.err \
    --wrap "python pipelines/02_atsemapper/04_run_atsemapper.py $CONFIG $PROC_DIR"
```

**mouse_smartseq2 — 2026-05-12:** 39,729 ATSEs across 17,604 genes, 142,896 junctions
```
ATSE_FILE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz
```

## Step 5: Build per-chunk AnnDatas + audit trail

```bash
ATSE_FILE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz

bash pipelines/02_atsemapper/05_make_anndata_slurm.sh $CONFIG $PROC_DIR $ATSE_FILE
# Produces per chunk:
#   $PROC_DIR/anndatas/chunk_N_anndata.h5ad
#   $PROC_DIR/anndatas/chunk_N_audit.json   ← ground truth counts from raw BED
```

## Step 6: Merge AnnDatas

```bash
ANNDATA_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511/anndatas

cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

sbatch --mem=64G --cpus-per-task=4 -p cpu \
    --output=${ANNDATA_DIR}/merge_anndata_%j.out \
    --error=${ANNDATA_DIR}/merge_anndata_%j.err \
    --wrap "python pipelines/02_atsemapper/06_merge_anndatas.py \
        --input-dir ${ANNDATA_DIR} \
        --output-file ${ANNDATA_DIR}/merged_anndata.h5ad"
# Produces: $ANNDATA_DIR/merged_anndata.h5ad
```

**mouse_smartseq2 — 2026-05-22:** 169,503 cells × 142,896 junctions
- Layers: `cell_by_junction_matrix` (raw junction counts), `cell_by_cluster_matrix` (ATSE total per junction)
- Junction-level features; each junction maps to an ATSE via `var["event_id"]`

## Step 6b: ✓ AUDIT — verify counts match raw data

```bash
ANNDATA_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511/anndatas

cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

# Mandatory before proceeding — verifies AnnData counts match raw BED files
python tests/test_count_audit.py \
    ${ANNDATA_DIR}/merged_anndata.h5ad \
    ${ANNDATA_DIR}
```

**mouse_smartseq2 — 2026-05-22: ✓ PASSED 20,000/20,000 checks**

**If this fails, stop and investigate before continuing.**

---

## Step 7: Metadata preparation

```bash
# Mouse: combines TMS FACS + Allen Brain Mouse metadata into one CSV
# Also applies broad_cell_type from cell_type_maps/mouse_celltype_map.json
CONFIG=config/mouse_smartseq2.yaml
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
python pipelines/03_gene_expression/00_prepare_metadata.py $CONFIG
# Produces: $WD/metadata_combined.csv
```

```bash
# Human: combines Tabula Sapiens + Allen Brain Human MTG into one CSV
# Requires ge_h5ad (Tabula Sapiens processed h5ad) from config
CONFIG=config/human_smartseq2.yaml
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
python pipelines/03_gene_expression/00_prepare_human_metadata.py $CONFIG
# Produces: $WD/metadata_combined.csv
```

**Output columns:** `cell_id, age, cell_ontology_class, mouse.id, sex, subtissue, tissue, broad_cell_type`

**Cell type map:** `cell_type_maps/mouse_celltype_map.json` — maps all 501 unique
`cell_ontology_class` labels (388 numbered Allen Brain subtypes + 113 TMS ontology terms)
to broad types (INHIBITORY NEURON, EXCITATORY NEURON, HIPPOCAMPAL NEURON, T CELL, etc.).
Regenerate with all current labels: `python -c "..."` (see script header).

**Mouse cell ID formats:**
- TMS (3m): `{barcode}-{plate}-1-1` (. and _ replaced with -)
- TMS (other ages): strip `.mm10-plus-0-0` or `.mus-2-1` platform suffix
- Allen Brain: SRR accession (from `sra_table` → `Run` column)

**Config filters applied:**
- `metadata.exclude_ages: ["21m"]` — 728 cells dropped (too few for statistical analysis)

**mouse_smartseq2 — 2026-05-22 results:**
- 179,403 metadata rows (180,131 − 728 dropped 21m)
- AnnData coverage: 165,970 / 169,503 = **97.9%**
- 3,533 Allen Brain cells in AnnData have no metadata: their SRR accessions are in
  `SraRunTable.txt` but their FASTQ filenames are absent from `sra_s3_links.txt`,
  so they cannot be joined to the Allen Brain cell-type/donor metadata. No column
  bridges SRR → internal lab sample ID (`US-XXXXXXX_EX_SXX`). These cells are
  excluded at Step 9 alignment. **Not a bug — accept the 2.1% loss.**
- `subtissue` null: 17,968 (16.3% of TMS cells only — Allen Brain always has region_label)

**Human cell ID formats:**
- Tabula Sapiens: `{TSP_project_id}_{sample}_{platform}_{donor1}_{donor2}_{tissue}_{subtissue}.multi_star.output_raw.output_raw_per_{old_index}`
- Allen Brain Human: `sample_name` column (matches BED file stems)

---

## Step 8: Gene expression processing

Gene expression uses **exonic counts only** (TMS + Allen Brain exons combined).
Allen Brain intronic counts are kept as a separate file for potential future use
but are **not used in the correction or training pipeline**.
Genes are length-normalized (divide by mean exon length across transcripts),
then library-size normalized to 10,000 counts, then log1p.
Retained biotypes: **protein-coding + lncRNA** (from GTF `gene_type` attribute).
Single-exon genes (no introns) are excluded so exon and intron files share the
same gene set.

```bash
CONFIG=config/mouse_smartseq2.yaml
OUT=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data

# 8a: Load raw counts, compute gene lengths, length-normalize, add broad_cell_type
# Use cpu partition — bigmem has a minimum memory-per-CPU floor (~50 GB/CPU)
# that 100G jobs don't meet. cpu nodes have 235 GB / 28 CPUs (~8.4 GB/CPU).
sbatch --mem=100G --cpus-per-task=12 -p cpu \
    --output=${OUT}/ge_01_%j.out \
    --error=${OUT}/ge_01_%j.err \
    --wrap "python pipelines/03_gene_expression/01_prepare_counts.py $CONFIG"
# Produces:
#   processed_data/tms_ab_exons_combo_ge_adata_YYYY-MM-DD.h5ad  ← main GE input
#   processed_data/ab_adata_introns_YYYY-MM-DD.h5ad             ← kept, not used downstream

# 8b: Run scVI on the ALIGNED GE AnnData (run AFTER Step 9).
#     Using aligned GE (150,906 cells × 13,119 genes) ensures only final model cells
#     are included in the batch correction / latent space.
#
# ⚠️  SLURM --wrap uses sh, not bash — 'source' and 'conda activate' will fail
#     with exit code 127. Use the full Python path from scvi-env instead.
#
# scvi-env: torch 2.10.0+cu128 = CUDA 12.8; 'scvi_gpu' env is CPU-only despite name
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/scvi-env/bin/python
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
MODEL_INPUT=$WD/MODEL_INPUT/YYYYMMDD   # ← use the dated dir from Step 9
GE=$MODEL_INPUT/ge_adata_aligned_YYYY-MM-DD.h5ad

# With dataset batch correction (TMS vs Allen Brain):
sbatch --mem=100G --gres=gpu:1 -p gpu \
    --output=${OUT}/ge_scvi_batch_%j.out \
    --error=${OUT}/ge_scvi_batch_%j.err \
    --wrap "$PYTHON pipelines/03_gene_expression/02_run_scVI.py $CONFIG --ge $GE --batch-key dataset"

# Without batch correction:
sbatch --mem=100G --gres=gpu:1 -p gpu \
    --output=${OUT}/ge_scvi_nobatch_%j.out \
    --error=${OUT}/ge_scvi_nobatch_%j.err \
    --wrap "$PYTHON pipelines/03_gene_expression/02_run_scVI.py $CONFIG --ge $GE --batch-key none"
# Produces:
#   $WD/scVI/batch_dataset/ge_adata_scVI_YYYY-MM-DD.h5ad
#   $WD/scVI/no_batch/ge_adata_scVI_YYYY-MM-DD.h5ad
```

**mouse_smartseq2 — Step 8b (2026-05-25): ✓ COMPLETE**
- Input: `ge_adata_aligned_2026-05-25.h5ad` (150,906 cells × 13,119 genes)
- `scVI/batch_dataset/ge_adata_scVI_2026-05-25.h5ad` — 16G, with batch correction (TMS vs Allen Brain)
- `scVI/no_batch/ge_adata_scVI_2026-05-25.h5ad` — 16G, no batch correction
- Both include `umap_scVI.png`

**mouse_smartseq2 — 2026-05-24 (Step 8a complete):**
- 184,187 cells × 18,854 genes
- 18,854 = 17,836 protein_coding + 1,018 lncRNA subtypes (lincRNA, antisense, etc.)
- 2,057 single-exon genes dropped (no introns → consistent gene set with intron file)
- 18 duplicate gene symbols dropped per dataset (two Ensembl IDs → same symbol)
- `broad_cell_type` added to obs from `cell_type_maps/mouse_celltype_map.json`

**Layers in combined exon AnnData (`tms_ab_exons_combo_ge_adata_*.h5ad`):**

| Layer | Content | How to use |
|---|---|---|
| `.X` | Raw integer counts (same as `raw_counts`) | Identical to `raw_counts` layer |
| `raw_counts` | Raw integer counts, explicitly named | Use for DE analysis, QC, any tool expecting raw counts |
| `length_norm` | `raw_counts ÷ mean_transcript_length`, then library-size normalized to 10,000 | Intermediate — not the best starting point for recovery |
| `log_norm` | `log1p(length_norm)` | Input to scVI and all downstream models |

> **Note:** `length_norm` was modified in-place by `sc.pp.normalize_total` before `log_norm` was
> computed, so the per-cell library size is not stored separately. To recover raw counts,
> use `raw_counts` directly — do not try to back-calculate from `length_norm`.

**SLURM partition guidance:**
- `cpu`: max ~235 GB / 28 CPUs per node (~8.4 GB/CPU). Use for jobs ≤ 200G.
- `bigmem`: 2 TB / 72 CPUs per node. QOS `MinTRES=mem=500G` — minimum 500G per job.
  Also requires ≥ 50 GB/CPU. Use `--mem=500G --cpus-per-task=10`. Never request less than 500G.
- Never mix `cpu,bigmem` — the more restrictive QOS wins and causes confusion.

---

## Step 9: Align splice + GE AnnDatas

```bash
# ⚠️  Cell ID alignment — the historically tricky step.
# See align_splice_ge.py docstring for full explanation of ID formats.
#
# Key insight (mouse):
#   Splice obs_names are already canonical (SRR for AB, cleaned TMS format).
#   GE obs_names need cleaning:
#     AB cells: exp_component_name (US-/LS-/SM- prefix) → mapped to SRR
#               via metadata CSV (tissue column = exp_component_name)
#     TMS cells: strip .mm10-plus-0-0 suffix (non-3m) or apply 3m special
#               format rule (same as metadata_prep.py format_cell_id)
#   The metadata CSV acts as the bridge for AB GE→splice matching.

CONFIG=config/mouse_smartseq2.yaml
ANNDATA_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260511/anndatas
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
SPLICE=$ANNDATA_DIR/merged_anndata.h5ad
GE=$WD/processed_data/tms_ab_exons_combo_ge_adata_2026-05-24.h5ad
OUT_DIR=$WD/MODEL_INPUT/$(date +%Y%m%d)
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python

cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

# ⚠️  Use full Python path — SLURM --wrap uses sh (not bash), conda activate fails.
# ⚠️  bigmem QOS MinTRES=500G (400G rejected); use 550G + --cpus-per-task=11 (50 GB/CPU floor)
sbatch --mem=550G -p bigmem --cpus-per-task=11 \
    --output=${OUT_DIR}/align_%j.out \
    --error=${OUT_DIR}/align_%j.err \
    --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/04_align/align_splice_ge.py \
        --config $CONFIG \
        --splice $SPLICE \
        --ge $GE \
        --output-dir $OUT_DIR"
# Adds metadata (age, cell_ontology_class, tissue, sex, mouse.id, subtissue) and
# broad_cell_type to splice obs from metadata_csv (set in config).
# GE already has broad_cell_type from Step 8a.
```

**Cell counts (2026-05-25 run, verified):**
- Splice input: 169,503 cells; GE input: 184,187 cells
- After zero-read filter: ~165,941 cells with metadata match
- After gene filter (common genes): 13,119 genes retained
- After MAD-based QC (5 MAD, min 5 junctions, min 200 genes): **150,906 cells**
- TMS: 106,199/106,199 = **100% match** (all TMS splice cells found in GE)
- AB:   59,771/63,304  = **94.4% match** (3,533 AB cells dropped — SRR in splice
  has no FASTQ filename in sra_s3_links, so no GE exp_component_name link;
  matches the 2.1% metadata loss noted in Step 7)

**Cell QC filters** (from `cell_qc` config block, applied in `align_splice_ge.py`):
- Cells with zero junction reads dropped before alignment
- After gene filter, a second zero-read check removes any newly empty cells
- `min_junctions: 5` — cells with fewer than 5 detected junctions dropped
- `max_junctions_nmads: 5.0` — cells > median + 5×MAD junction count dropped (doublets)
- `min_genes: 200` — cells with fewer than 200 detected GE genes dropped
- `max_genes_nmads: 5.0` — cells > median + 5×MAD gene count dropped (doublets)

**obs columns added to splice AnnData at Step 9:**
- `age`, `cell_ontology_class`, `sex`, `mouse.id`, `tissue`, `subtissue` — from metadata CSV
- `broad_cell_type` — from `cell_type_maps/mouse_celltype_map.json`
- `tissue` — for AB cells, overwritten from `exp_component_name` to Allen CCF region category based on `subtissue`: `Brain_Isocortex_Visual` (VISp, VIS), `Brain_Isocortex_Motor` (MOp, ALM), `Brain_Isocortex_Somatosensory` (SSp, SSs), `Brain_Isocortex_Association` (ACA, RSP, RSPv, PL-ILA, ORB, AI, GU, CLA), `Brain_Isocortex_Temporal` (TEa-PERI-ECT, PTLp, AUD), `Brain_Hippocampus` (HIP, SUB-ProS, PAR-POST-PRE, ENTl, ENTm). TMS tissue unchanged.
- `tissue_cell_type` — `tissue + "_" + broad_cell_type`, uniform for both TMS and AB (e.g., `Brain_Hippocampus_EXCITATORY NEURON`, `Liver_HEPATOCYTE`)
- `cell_id` — canonical cell ID from metadata CSV (same format as metadata_combined.csv)

## Step 9b: ✓ AUDIT — verify cell alignment

```bash
python tests/test_cell_id_alignment.py \
    $OUT_DIR/splice_adata_aligned_*.h5ad \
    $OUT_DIR/ge_adata_aligned_*.h5ad

python tests/test_metadata_coverage.py \
    $OUT_DIR/splice_adata_aligned_*.h5ad \
    $WD/metadata_combined.csv
```

---

## Step 9c: Filter aligned objects before training

Filters the aligned splice + GE AnnDatas to remove rare junctions and low-coverage
cells **before** making model-ready inputs. The original aligned h5ads are never
modified — filtered copies are written alongside them.

**Why:** the May 2026 alignment includes ~22K more junctions than Oct 2025 (111,903 vs
89,831), many detected in very few cells. This makes the splice matrix ~2× sparser per
cell (mean 4,136 vs 8,994 junctions/cell), causing unstable latent spaces and slower
training. Filtering recovers Oct 2025 quality without changing upstream steps.

**Filters applied in order:**
1. Junction filter: drop junctions detected in fewer than N cells (computed from sparse matrix)
2. Cell filter: drop cells with fewer than M junctions *after* the junction filter

**Run scan first** to see retention at each threshold (loads matrix, no files written):
```bash
OUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526

python pipelines/04_align/09b_filter_aligned.py \
    --splice $OUT_DIR/splice_adata_aligned_2026-05-27.h5ad \
    --ge     $OUT_DIR/ge_adata_aligned_2026-05-27.h5ad \
    --output-dir $OUT_DIR \
    --scan
```

**Apply filters** (thresholds chosen including cells with ≥500 junctions):
```bash
OUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python

sbatch --mem=150G -p cpu \
    --output=${OUT_DIR}/filter_aligned_%j.out \
    --error=${OUT_DIR}/filter_aligned_%j.err \
    --wrap "export PYTHONUNBUFFERED=1; $PYTHON \
        pipelines/04_align/09b_filter_aligned.py \
        --splice $OUT_DIR/splice_adata_aligned_2026-05-27.h5ad \
        --ge     $OUT_DIR/ge_adata_aligned_2026-05-27.h5ad \
        --output-dir $OUT_DIR \
        --min-cells-per-junction 75 \
        --min-junctions-per-cell 500"
```

**mouse_smartseq2 — 2026-06-17 ✓ COMPLETE (supersedes 2026-06-13):**
- Junction filter=75 → 88,833 junctions; ATSE re-grouping (ATSEAnalyzer) → 79,768 junctions, 24,475 ATSEs (9,065 singletons dropped)
- Cell filter=500 post junction filter → 132,706 cells
- `cell_by_cluster_matrix`: nnz=603,601,993
- `var["event_id"]` = new re-derived ATSEs; `var["original_event_id"]` = original ATSEmapper IDs

**Outputs — 2026-06-17 (send these to Smriti):**
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad` (30G)
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad` (13G)
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/junctions_filtered_minCells75_minJuncs500_2026-06-17.bed` (11M) — junction coords + new ATSE IDs

**⚠️ Superseded (2026-06-13):** `*_minCells75_minJuncs500_2026-06-13.h5ad` — used old partial-ATSE logic; do not use for new training.

**✓ Smriti h5mu received 2026-06-13** (based on old files). New h5mu needed from Smriti using 2026-06-17 filtered h5ads. The originals (150,906-cell aligned h5ads) remain untouched for reference.

### Retraining pipeline on filtered data

```
Filtered h5ads (ready ✓)
    splice: 132,875 × 83,593 junctions (25,750 ATSEs)
    ge:     132,875 × 11,673 genes
    ↓
Smriti h5mu ✓ DONE (2026-06-13, MODEL_INPUT/20260526/)
    → train_70_30_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu  (93,012 cells, 93G)
    → test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu   (39,863 cells, 40G)
    + resampled/RESAMPLED_{10,25,50,75,90,99,100}_PERCENT_test_*.h5mu ✓ all 7 done
    ↓
00_extract_train_splice.py  ← NEXT STEP (run this now)
    → train_splice_filtered_model_ready_YYYY-MM-DD.h5ad
    ↓
01_generate_params.py + 02_submit_jobs.py
    → retrain 24 models on 83,593 → ~81,489 junctions (num_junctions ≤ 10)
    ↓
EasySci (re-run): 01_prep_easysci_for_model.py --train-h5ad <new train h5ad> --max-junctions 10
    → EasySci re-aligned to exact training junction space
    (interim EasySci at 83,593 junctions already produced 2026-06-13)
```

**Note on junction spaces:**
| Training run | Junction space | Status |
|---|---|---|
| 2026-06-01 (old, 24 models) | 101,632 junctions (num_junc ≤ 10 of 111,903) | Superseded — do not use for EasySci |
| **Filtered retrain (pending)** | **~81,489 junctions** (num_junc ≤ 10 of 83,593) | **h5mu ready ✓ — run 00_extract_train_splice.py to start** |

---

## Step 10: Train LeafletFA

Three sub-steps: prep input → generate parameter grid → submit jobs.

### Step 10 pre: Extract train splice from h5mu (train/test split)

The aligned splice+GE data is stored as an h5mu with a 70/30 train/test split
(created by S. Vaidyanathan). LeafletFA trains on the **70% train split only**.

**Train/test files** (in `MODEL_INPUT/20260526/`) — ✓ all present as of 2026-06-13:

| File | Cells | Size | Purpose |
|------|-------|------|---------|
| `train_70_30_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu` | 93,012 | 93G | LeafletFA training ← **use this** |
| `test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu` | 39,863 | 40G | Held-out evaluation |
| `resampled/RESAMPLED_{10,25,50,75,90,99,100}_PERCENT_test_*.h5mu` | varies | 36–49G each | Imputation benchmarking at different masking levels |

The h5mu has `rna` and `splicing` modalities. The dense `junc_ratio` layer in
the splicing modality is ~47GB — loading it naively causes OOM. The extraction
script avoids this by reading only the sparse layers via h5py and recomputing
`junc_ratio`/`junc_ratio_centered` in chunks.

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
H5MU=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_70_30_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu
OUT=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-14.h5ad

sbatch --mem=60G -p cpu --cpus-per-task=4 \
    --output=${OUT%.h5ad}_%j.out \
    --error=${OUT%.h5ad}_%j.err \
    --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/05_leafletfa/00_extract_train_splice.py $H5MU $OUT"
# Reads sparse layers from h5mu via h5py (skips dense junc_ratio)
# Recomputes junc_ratio + junc_ratio_centered in 500-cell chunks
# Peak memory: ~20-30 GB
```

**⚠️ Superseded (2026-06-01):** `train_splice_model_ready_2026-06-01.h5ad` — 105,634 × 111,903
This was extracted from the old (pre-filter) h5mu (150,906-cell, 111,903-junction space). **Do not use.**

**Next output** (run the block above): `train_splice_model_ready_2026-06-14.h5ad` — expected 93,012 × 83,593 junctions.

> **Note:** The extraction script now uses `calculate_centered_psi()` from `leafletfa.waypoints` (beta-binomial weighted mean centering, rho=0.1), matching the 2025 submission pipeline.

---

### Step 10a: QC — PCA + UMAP of centered PSI (run before training)

Sanity check that `junc_ratio_centered` captures biologically meaningful variation.
Reads the sparse layer directly from the h5ad (no full load). Produces PCA + UMAP
plots colored by broad cell type (top 10) and tissue × cell type (top 15).

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
TRAIN_H5AD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_weighted_centered_2026-06-14.h5ad
QC_OUT=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/QC/psi_pca_umap

sbatch --mem=80G -p cpu --cpus-per-task=4 \
    --output=${QC_OUT}/pca_umap_%j.out \
    --error=${QC_OUT}/pca_umap_%j.err \
    --wrap "mkdir -p $QC_OUT && export PYTHONUNBUFFERED=1; \
            $PYTHON pipelines/05_leafletfa/00b_visualize_psi_pca.py $TRAIN_H5AD $QC_OUT"

# Quick PCA-only run (no UMAP, ~5 min):
# $PYTHON pipelines/05_leafletfa/00b_visualize_psi_pca.py $TRAIN_H5AD $QC_OUT --skip-umap
```

**What to check in the outputs:**

| File | What to look for |
|------|-----------------|
| `pca_variance_explained.png` | Elbow shape — sharp drop then plateau means real structure |
| `pca_broad_cell_type.png` | PC1/PC2 should separate major cell types (neurons, immune, epithelial) |
| `umap_broad_cell_type.png` | Cell types cluster together, not randomly scattered |
| `umap_tissue_cell_type.png` | Fine-grained tissue×celltype clusters visible |
| `umap_dataset.png` | TMS and Allen Brain cells should **mix** — not fully separate (batch OK) |
| `umap_age.png` | Age should **not** drive major separation (splicing ≠ expression age effect) |

> If cell types don't separate on PC1/PC2, investigate before training — may indicate a centering bug or data issue.

---

### Step 10b: Generate parameter grid

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

CONFIG=config/mouse_smartseq2.yaml

python pipelines/05_leafletfa/01_generate_params.py $CONFIG
# Writes: LEAFLETFA_MODELS/YYYY-MM-DD/parameter_combinations.json
#         LEAFLETFA_MODELS/YYYY-MM-DD/parameter_combinations.csv
# Prints: "Generated 24 parameter sets (12 with waypoints, 12 random init)"
#         "num_inits=3 per set → 72 total model runs"

MODEL_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/$(date +%Y-%m-%d)
```

**Parameter grid (current — 2026-05-25 run):**

| Parameter | Values | Notes |
|-----------|--------|-------|
| `K` (factors) | **10, 20, 50** | from config `n_factors` |
| `batch_size` | **4096, 8192** | from config `param_grid_batch_sizes` |
| `num_passes` | **5** | from config `param_grid_num_passes` |
| `junc_specific_prior` | True, False | junction-specific Beta prior on PSI |
| `waypoints_use` | True, False | True = PCA metacell init; False = random init |
| `max_junctions` | 10 | ATSEs with >10 junctions dropped at training time |
| `num_inits` | 3 | 3 independent seeds per param set for reproducibility |

With current config: 3 × 2 × 1 × 2 × 2 = **24 parameter sets**, each run 3 times = **72 total model fits**

All combinations estimated within the 48h SLURM wall time (K=50 + batch=4096 worst case ~35 hrs).

**How waypoints initialization works:**
- `waypoints_use=True`: runs PCA on the junction PSI ratio matrix, selects K "waypoint" metacells
  that span the latent space, and uses them to initialize the K×P PSI matrix (Ψ). This gives the
  model a structured starting point close to the data geometry.
- `waypoints_use=False`: Ψ is initialized randomly. Slower convergence but avoids any bias from
  PCA initialization — useful to confirm that waypoints don't force a particular solution.
- With 3 inits per set, you can assess which approach yields more consistent latent factors
  across runs (lower variance = more stable solution).

**To override via config** (`config/mouse_smartseq2.yaml` `leafletfa:` block):
```yaml
param_grid_waypoints_use: [True]    # waypoints only
param_grid_batch_sizes: [4096]      # single batch size
num_inits: 1                        # single run
```

### Step 10c: Submit training jobs

Submits a **single SLURM array job** capped at `--max-concurrent` GPU tasks at once.
All 24 tasks share one job ID — easy to monitor and cancel.

Run after Steps 9 → 10a → 10b complete:

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

CONFIG=config/mouse_smartseq2.yaml
OUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/$(date +%Y%m%d)
MODEL_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/$(date +%Y-%m-%d)
ANNDATA=$(ls $OUT_DIR/model_ready_aligned_splicing_data_*.h5ad | tail -1)

# Dry-run first: writes the .sh script and prints the sbatch command without submitting
python pipelines/05_leafletfa/02_submit_jobs.py $CONFIG $MODEL_DIR $ANNDATA --max-concurrent 2 --dry-run

# Submit (max 2 GPU jobs at a time — lab-fair throttle)
# Note: ne1dg7-[001-010] are NVIDIA RTX PRO 6000 Blackwell (sm_120), not supported
# by PyTorch 2.4 / Triton — always excluded via #SBATCH --exclude in the array script
python pipelines/05_leafletfa/02_submit_jobs.py $CONFIG $MODEL_DIR $ANNDATA --max-concurrent 2

# Monitor / cancel all tasks at once:
squeue -u $USER | grep leaflet
scancel <JOBID>
```

**What it does:**
- Writes `$MODEL_DIR/leaflet_array_job.sh` with `#SBATCH --array=0-23%2` and `--exclude=ne1dg7-[001-010]`
- Uses full Python path (`/miniconda3/envs/LeafletSC/bin/python`) — no conda activation needed
- Calls `run_training.py` which imports from the installed `leafletfa` package
- Logs: `$MODEL_DIR/logs/leaflet_<JOBID>_<TASKID>.out`
- Results: `$MODEL_DIR/run_{i}/` — one dir per param set (24 dirs total)

**Config overrides** (in `leafletfa:` block):
```yaml
python_bin: /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
run_script: pipelines/05_leafletfa/run_training.py  # uses installed leafletfa
```

**Training script** (`run_training.py`):
- Imports: `from leafletfa import LeafletFA`, `import leafletfa.waypoints as wayp`
- GPU-aware: auto-detects CUDA, falls back to CPU
- Logs to wandb project `LeafletFA-MouseFoundation_YYYY-MM-DD`

### Step 10d: Collect and compare results

Run after training completes (or any time during training to check partial results).

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

MODEL_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/YYYY-MM-DD

python pipelines/05_leafletfa/03_collect_results.py $MODEL_DIR
# Prints: ranked table by best ELBO, best run per init type, convergence check per top run
# Saves:  $MODEL_DIR/all_runs_summary.csv   — full merged table (all params + metrics)
#         $MODEL_DIR/all_runs_display.csv   — key columns only for quick review
```

**Per-run outputs saved by `run_training.py`** (in each `run_{i}/`):

| File | Contents |
|------|----------|
| `run_summary.csv` | Final ELBO, pruned K, cell coverage, all hyperparams |
| `pass_summary_results.csv` | Average ELBO per pass — use to check convergence |
| `batch_training_results.csv` | ELBO per mini-batch |
| `pi_evolution.csv` | Factor weights (π) per pass |
| `pi_evolution_plot.png` | π trajectory for top 10 factors |
| `pi_evolution_heatmap.png` | π heatmap across all passes |
| `factor_assignment_probabilities.csv` | Final factor weights sorted |
| `leafletfa_model.pkl.gz` | Saved model (psi, phi/assign_post, pi, etc.) |

**Two logging systems running in parallel:**
- **wandb** (live): every batch and pass logged to `LeafletFA-MouseFoundation_YYYY-MM-DD` project; view at wandb.ai
- **local CSVs** (post-hoc): `03_collect_results.py` merges all `run_summary.csv` files into one ranked table

---

---

## Step 11: Evaluate all 24 trained models

After training completes, run steps 11a–11d before choosing the final model.
Steps 11b and 11c can run **in parallel** with the 11a eval array jobs.

```bash
WD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
MODEL_DIR="$WD/LEAFLETFA_MODELS/2026-06-14"
TRAIN_H5AD="$WD/MODEL_INPUT/20260526/train_splice_model_ready_weighted_centered_2026-06-14.h5ad"
RESAMPLED="$WD/MODEL_INPUT/20260526/resampled"
SCVI_H5AD="$WD/scVI/batch_dataset/ge_adata_scVI_2026-06-13.h5ad"   # 132,875 cells; has X_scVI_linear
EVAL_OUT="$WD/LEAFLETFA_MODELS/2026-06-14/EVALUATION/2026-06-15"
```

**Status (2026-06-15):** 11a eval array submitted as job 17231614 (24 tasks, CPU partition, running).
Steps 11b + 11c can be launched now in parallel.

### Step 11a: Submit imputation evaluation array

Evaluates all 24 models across 7 read-coverage levels (10%–100%) on held-out test cells.
Metric: per-cell Pearson r between imputed and observed PSI, restricted to junctions with AT > 0.

```bash
# Generates and prints sbatch command — copy and run the printed command
python pipelines/05_leafletfa/05_submit_evaluation.py \
    $MODEL_DIR $TRAIN_H5AD $RESAMPLED $EVAL_OUT
# sbatch $EVAL_OUT/eval_array_job.sh   ← run what the script prints
```

Worker script (`04_evaluate_imputation.py`) is called once per run by the SLURM array.
Per-job output: `$EVAL_OUT/run_N_imputation_results.csv`

### Step 11b: Consistency check (parallel with 11a)

Checks whether the same factors are reproducibly learned across random initializations within each K.

```bash
python pipelines/05_leafletfa/06_consistency_check.py \
    $MODEL_DIR $EVAL_OUT/consistency
# Outputs:
#   consistency_factor_level.csv   — Hungarian-matched cosine sim per run pair per K
#   consistency_cell_level.csv     — per-cell Pearson r of imputed PSI per run pair per K
#   consistency_factor_level.png
#   consistency_cell_level.png
```

### Step 11c: Classifier evaluation (parallel with 11a)

Tests whether `assign_post` (N × K cell factor weights) predicts biological metadata better
than scVI and a shuffled null. Submitted as a SLURM array (one job per run_N).

Targets:
- `broad_cell_type`, `tissue_cell_type`, `dataset` — logistic regression (balanced accuracy + macro-F1)
- `age` — Ridge regression predicting numerical age in months (R² + Pearson r)
- `sex` removed (ambiguous/missing values in data)

```bash
# Generate and submit the SLURM array job (one task per run_N):
python pipelines/05_leafletfa/07b_submit_classification.py \
    $MODEL_DIR $TRAIN_H5AD $SCVI_H5AD $EVAL_OUT/classifiers
# then run: sbatch $EVAL_OUT/classifiers/classify_array_job.sh

# Outputs (per-run, in $EVAL_OUT/classifiers/):
#   run_{N}_classifier_results.csv   — one file per run with baselines + LeafletFA results
```

Script notes:
- `07_classify_splicing_programs.py` accepts optional `--run-dir` to process a single run
  (used by the array job). Without `--run-dir` it loops over all runs sequentially (original mode).
- Each array task also computes scVI baseline and permuted-label null for self-contained output.
- Plots and merged table are generated separately after all jobs finish (TBD: `08b_collect_classification.py`).

### Step 11d: Collect imputation results and pick best model

Run after 11a eval array finishes.

```bash
python pipelines/05_leafletfa/08_collect_evaluation.py $EVAL_OUT
# Outputs:
#   all_runs_imputation_results.csv
#   best_models_imputation_results.csv   — best per K by mean Pearson r at 100% masking
#   imputation_accuracy_vs_coverage.png  — Panel A (best/K) + Panel B (all 24 runs)
```

---

## Step 12: Retrain best model on full dataset (train + test)

After Step 11d identifies the best model per K, retrain on all 132,875 cells.
Use `--dry-run` first to confirm the right model is selected.

```bash
TEST_H5MU=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu
FULL_MODEL_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-06-14/full_retrain

# Dry run first — confirm it picks the right param_id
python pipelines/05_leafletfa/09_retrain_best_on_full.py \
    $EVAL_OUT $MODEL_DIR $TRAIN_H5AD $TEST_H5MU $FULL_MODEL_DIR --dry-run

# Submit (prints sbatch command)
python pipelines/05_leafletfa/09_retrain_best_on_full.py \
    $EVAL_OUT $MODEL_DIR $TRAIN_H5AD $TEST_H5MU $FULL_MODEL_DIR
```

**What it does:**
1. Reads best param_id from `$EVAL_OUT/best_models_imputation_results.csv`
2. Extracts test cells from test h5mu (same logic as `00_extract_train_splice.py`)
3. Concatenates train + test → 132,875 cells; recomputes `junc_ratio_centered` (beta-binomial) over full dataset
4. Submits GPU job: `run_training.py` with best hyperparams on full splice h5ad

Output: `$FULL_MODEL_DIR/full_splice_model_ready_YYYY-MM-DD.h5ad` + trained model pkl.

---

## Step 12b: FINAL full-data model sweep (2026-07-06) — supersedes single-best 09_retrain

Model selection was redone because the original classification eval (`07`) decoded the
**training** `assign_post` (phi fit *with* those cells) → measured MEMORIZATION, not
transferable quality (cell-type accuracy rose monotonically with K — the memorization
signature). Replaced by **held-out inferred-phi decoding** (`07c`/`07d`): infer phi for the
30% held-out cells via the frozen-psi E-step, decode cell type + age from THAT.

**Held-out decode findings (96-run grid, `eval_results_heldout/heldout_leaderboard.csv`):**
- Every metric ~2× lower on held-out vs training (memorization gap real, ~0.26 at K=50).
- **K=50 best for BOTH** cell type (held-out broad 0.25) and age (0.20); waypoints ~2× help.
- **lr is a per-metric trade-off:** lr=0.001 → cell type; lr=0.01 → age.
- **3-way interaction:** binomial likelihood is robust to lr=0.01; **beta-binomial COLLAPSES
  at lr=0.01** (K=50 bb lr.01 → broad 0.11). So the best JOINT config is
  **K=50 / waypoints / binomial / lr=0.01** (broad 0.33 + age 0.42).
- Anomaly: some no-waypoint runs have collapsed *training* phi (0.026) yet good *held-out*
  phi — the E-step recovers phi even when the stored assign_post collapsed (held-out decode
  decouples "training-phi quality" from "what the model gives on new cells").

**Final config space submitted (16 configs):** `K{20,50} × likelihood{binom,bb} × lr{.001,.01}
× junc_specific_prior{T,F}`, waypoints=True (dropped lr=0.0001 = worst; dropped K=5,10 = weak).
param_ids: `13,14,19,20,37,38,43,44,61,62,67,68,85,86,91,92`.

Submitted via `10_submit_full_training.py` (filters the grid by criteria, submits
`run_training.py` per param_id as a `%3` GPU array; 300G, exclude ne1dg7-[001-011], 2-day):

```bash
# MOUSE — full 132,706-cell object (has junc_ratio_centered)
python pipelines/05_leafletfa/10_submit_full_training.py \
    $WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad \
    $WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/FULL_2026-07-06 \
    $WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-06-18        # source of parameter_combinations.json
    # defaults: --K 20,50 --lr 0.001,0.01 --lik binom,bb --jsp True,False --waypoints True --max-concurrent 3
    # --dry-run previews matched param_ids first

# HUMAN — aligned file needs the extract step first (only has raw count layers)
sbatch -p cpu --mem=200G --time=8:00:00 --wrap="$PY \
    pipelines/05_leafletfa/00_extract_train_splice.py \
    $WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad \
    $WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad"
# then the SAME 10_submit_full_training.py pointed at human_splice_model_ready_2026-06-17.h5ad
# + $WD/HUMAN_SPLICING_FOUNDATION/LEAFLETFA_MODELS/FULL_2026-07-06 + the mouse 2026-06-18 param grid.
```

**Notes on params / objects:**
- `SLURM_ARRAY_TASK_ID = param_id`; `run_training.py` reads `parameter_combinations.json[param_id]`
  (copied into the FULL_ dir by the submit script). Same 96-config grid = `K{5,10,20,50} ×
  waypoints{T,F} × likelihood{binom,bb} × lr{.0001,.001,.01} × junc_specific_prior{T,F}`.
- **Train on the model_ready object, NOT the aligned one.** Mouse aligned
  (`splice_adata_aligned_..._2026-06-17.h5ad`, 30G) has only raw count layers; the model_ready
  (`train_splice_model_ready_2026-06-17.h5ad`, 132,706 cells) has `junc_ratio` +
  `junc_ratio_centered` (beta-binomial, verified real negatives). Human aligned (43,442 cells)
  is likewise raw → run `00_extract_train_splice.py` to add the PSI/centered/psi_mask layers.
- Param grid is species-agnostic (K/lr/likelihood/prior are generic); data dims read from the
  h5ad at runtime. So the same 16 param_ids train human identically.
- Next after the sweep completes: characterize finalists with imputation (`04f`, now feasible
  on ~16 curated runs) + program interpretability → pick the single released model per species.

---

## Step 13: Apply trained model to EasySci external validation data

Validates that the model trained on TMS SmartSeq2 (mm10) generalises to the EasySci
brain atlas. EasySci uses mm39 (gencode.vM27); TMS uses mm10 (gencode.vM19) — a liftover
is required to map junction coordinates before applying the model.

### EasySci pipeline overview (how the existing data was generated)

```
EasySci raw BAMs (mm39, Rockefeller data)
    ↓
00_raw_data_pipeline/ — STAR alignment, BAM prep
    ↓
01_Metacell_analysis/00_MetaCellAnalysis.py
    → DT-based clustering → 4,931 metacell assignments
    ↓
01_MetaCellMakePseudobulk_RH_DT.sh  (SLURM array, 4931 tasks)
    → merge cell BAMs per metacell → one BAM per metacell cluster
    → SpliceVI/20260210/RH/<cluster>/<cluster>.bam
    ↓
02_snakemake/ (run_snakemake.sh → Snakefile)
    → regtools junctions extract (per metacell BAM, mm39 coords)
    → junctions_with_barcodes.bed files (9,921 files)
    ↓
ATSEmapper (on EasySci junctions, gencode.vM27/mm39)
    → EasySci_RH_ATSE_FILE_unanno_also_2025-05-18_07-48-32.txt.gz
      (EasySci-specific ATSE groupings: 9,748 ATSEs, mm39 junction coords)
    ↓
05_make_anndata.py
    → junction_processing_20250223/anndatas/merged_anndata.h5ad
      (547 × 83,216, mm39, layers: cell_by_junction_matrix, cell_by_cluster_matrix)
```

**Key paths:**
- EasySci merged anndata: `EasySci2024/LeafletFA/MetaCells/ATSEmap/RH/output/junction_processing_20250223/anndatas/merged_anndata.h5ad`
- EasySci ATSE file: `EasySci2024/LeafletFA/MetaCells/ATSEmap/RH/output/ATSEfiles/EasySci_RH_ATSE_FILE_unanno_also_2025-05-18_07-48-32.txt.gz`
- Genome: `EasySci2024/genome_files/gencode.vM27.basic.annotation.gtf` (mm39)
- Pipeline scripts: `LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/`

**Liftover strategy — why and how:**
EasySci junction IDs are mm39 coordinates; the TMS model junction IDs are mm10. A separate
liftover step maps TMS mm10 junctions → mm39 coordinates using `pyliftover` with the
`mm10ToMm39.over.chain.gz` chain file (already on cluster). The output file has mm39
`junction_id` + original mm10 `mouse_foundation_junction_id` so that EasySci junctions
can be directly matched to model junction space.

**Coverage note:**
- Old liftover (2025-10-01 TMS model): covered 71.5% of new 2026 model ATSEs → 54.3% of model junctions covered by EasySci
- **✓ New liftover (2026-06-12):** 142,895/142,896 junctions lifted (0% loss). Coverage stats to be confirmed after retrain (final junction space: 81,489).
- Remaining gap: biology-limited (EasySci is brain-only; TMS has all tissues)

### Step 12-pre: ✓ COMPLETE — Generate mm39 liftover for new 2026 ATSE file

**✓ Done 2026-06-12.** 142,895/142,896 junctions lifted successfully.
Output: `MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52_lifted_mm39.txt.gz`

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
ATSE_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files
ATSE_FILE=$ATSE_DIR/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz
LIFTED=$ATSE_DIR/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52_lifted_mm39.txt.gz

# Runs quickly (~5 min) — can run on login node
$PYTHON pipelines/06_easysci/00_liftover_atse_to_mm39.py \
    --atse-file $ATSE_FILE \
    --output    $LIFTED

# Chain file: /gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/liftover/mm10ToMm39.over.chain.gz
# Script: adapted from LeafletFA_Submission2025/EasySci/LeafletFA/liftover_mouse_foundation_atse.py
```

`LIFTOVER=` in Step 12a already points to `$LIFTED`.

### Step 12a: Prepare EasySci data for model

**Four metacell sizes: 500, 1000, 2000, 4000.**
Run the script once per size; each job is independent.

**What the script does:**
1. Maps EasySci mm39 junctions → mm10 model junction space via liftover
2. Builds `cell_by_junction_matrix` (N × 83,593), zero-filling missing junctions
3. **Recomputes** `cell_by_cluster_matrix` from scratch using mm10 ATSE groupings (EasySci mm39
   cluster values are discarded — they used different ATSE groupings)
4. Spot-checks ATSE invariant: all junctions in observed ATSE have same ATSE total
5. Computes `junc_ratio` using cluster sparsity (PSI=0 for zero-read in observed ATSE)
6. Aligns GE metacell data to 11,673-gene filtered mouse GE space by Ensembl gene_id

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
EASYSCI_BASE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA
META_BASE=$EASYSCI_BASE/MetaCells
LIFTOVER=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52_lifted_mm39.txt.gz
REF_SPLICE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5ad
REF_GE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5ad
OUT_DIR=$EASYSCI_BASE/model_application

# Submit one job per metacell size (500, 1000, 2000, 4000)
for SIZE in 500 1000 2000 4000; do
    SPLICE_H5AD=$META_BASE/SpliceVI_${SIZE}/ATSEmap/junction_processing_20260318/anndatas/merged_anndata.h5ad
    GE_H5AD=$EASYSCI_BASE/metacell_RT_anndata_${SIZE}.h5ad
    sbatch --mem=32G -p cpu \
        --job-name=easysci_prep_${SIZE} \
        --output=${OUT_DIR}/easysci_prep_${SIZE}_%j.out \
        --error=${OUT_DIR}/easysci_prep_${SIZE}_%j.err \
        --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/06_easysci/01_prep_easysci_for_model.py \
            --easysci-splice-h5ad $SPLICE_H5AD \
            --easysci-ge-h5ad    $GE_H5AD \
            --liftover-atse      $LIFTOVER \
            --ref-splice-h5ad    $REF_SPLICE \
            --ref-ge-h5ad        $REF_GE \
            --output-dir         $OUT_DIR \
            --size               $SIZE"
done
```

**Outputs per size** (in `$EASYSCI_BASE/model_application/`):
- `easysci_splice_model_ready_{size}_YYYY-MM-DD.h5ad` — N × 83,593 (junction layer + cluster + junc_ratio)
- `easysci_ge_model_ready_{size}_YYYY-MM-DD.h5ad` — N × 11,673 (raw layer, gene-symbol indexed)

**Note on cluster matrix:** after retraining completes, re-run with updated `REF_SPLICE`
(new train h5ad) if the junction space changes. ATSE groupings come from REF_SPLICE var["event_id"].

### Step 12b: Apply model to EasySci

> **⚠️ ON HOLD** — run after Step 12a completes and Step 11 evaluation identifies best runs.
> Update `MODEL_BASE` to new retrained model directory and `EASYSCI_READY` to Step 12a output.

Runs LeafletFA with PSI fixed to trained values; infers EasySci cell factor assignments.

```bash
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026

PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
EASYSCI_READY=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/model_application/easysci_model_ready_YYYY-MM-DD.h5ad
# ⚠️ Update MODEL_BASE after retraining:
MODEL_BASE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/YYYY-MM-DD
OUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/model_application/results

# Apply the best model for each K (update run_N after Step 11 evaluation)
for RUN in run_0 run_4 run_8; do   # placeholder — use best K=10, K=20, K=50 from Step 11
    sbatch --mem=32G -p gpu --gres=gpu:1 \
        --wrap "$PYTHON pipelines/06_easysci/02_apply_model_to_easysci.py \
            --easysci-h5ad $EASYSCI_READY \
            --model-run-dir $MODEL_BASE/$RUN \
            --output-dir $OUT_DIR"
done
```

**Output per run:**
- `easysci_leafletfa_run_N.h5ad` — with `obsm['X_leafletFA_K{K}_run_N']`
- `easysci_leafletfa_run_N_assign_post.csv` — factor activities per metacell with cell type labels
- `easysci_leafletfa_run_N_pi.csv` — learned factor weights

---

## TODO / Planned next steps

| Priority | Task | Notes |
|----------|------|-------|
| High | **Step 9c ✓ done** | Filtered h5ads written: 132,875 × 83,593 junctions / 11,673 genes |
| High | **Smriti h5mu ✓ received (2026-06-13)** | Train: 93,012 cells (93G); Test: 39,863 cells (40G); all 7 resampled test sets done |
| High | **`00_extract_train_splice.py` ✓ done** | `train_splice_model_ready_weighted_centered_2026-06-14.h5ad` (13G, 93,012 × 83,593); beta-binomial centering; PCA/UMAP QC plots look reasonable |
| High | **Training 24 models ✓ running** (job 17217361) | 2026-06-14; K=10/20/50 × waypoints=T/F × prior=T/F × batch=4096/8192; `max_junctions=null` (all 25,750 ATSEs) |
| High | **Step 11: Run evaluation after training** | 11a: imputation eval; 11b: consistency check; 11c: classifier eval; 11d: collect + pick best |
| High | **Step 12: Retrain best model on full 132,875 cells** | After Step 11d picks winner |
| High | **Step 13a ✓ done** — EasySci prep for sizes 500/1000/2000/4000 | Outputs in `$EASYSCI_BASE/model_application/`; 65.1% junction coverage, 99.6% gene coverage; ATSE invariant spot-checks passed |
| High | **Rerun scVI ✓ done** | Both jobs completed: `scVI/batch_dataset/ge_adata_scVI_2026-06-13.h5ad` and `scVI/no_batch/ge_adata_scVI_2026-06-13.h5ad` (132,875 × 11,673) |
| High | **Sanity check scVI latent space** | Load UMAP from `scVI/{batch_dataset,no_batch}/umap_scVI.png`; check that leiden clusters correspond to expected cell types (Allen Brain / TMS labels visible in obs) |
| High | **Cell ID annotation check ✓ done** | 132,875/132,875 cells found in `metadata_combined.csv`; 15 TMS + 15 Allen Brain cells spot-checked — cell_ontology_class, tissue, subtissue, broad_cell_type all match exactly |
| High | **Regtools junction cross-reference ✓ done** | 3 Allen Brain cells (SRR16430115, SRR16442493, SRR16438582) cross-referenced vs raw `_junctions.bed` files — 100% exact match (2593/2593, 2550/2550, 2420/2420 junctions); h5ad uses intron coords (bed_start+overhang_up, bed_end−overhang_dn) |
| Medium | **Step 11: Evaluation** | Hold — re-run after retraining on filtered data with new test sets from Smriti; update MODEL_BASE + TRAIN_H5AD paths |
| Medium | **Human Allen Brain: prepare GE + splicing data for modeling** | Run same pipeline as mouse (Steps 1–10) on human Allen Brain single nuclei dataset (MTG + broader regions). Share processed h5mu / h5ad outputs with Smriti. Use `config/human_smartseq2.yaml`. Do this after EasySci validation is complete. |
| Medium | **Human brain atlas — Allen Brain single nuclei only** | Skip Tabula Sapiens (data quality concerns). Use Allen Brain single nuclei human brain dataset (MTG and broader regions). Do this after EasySci validation is complete. |
| Low | **NMF baseline comparison** | sklearn NMF at K=10,20,50 on same train split; compare imputation vs LeafletFA on test set |
| Low | **Stratified waypoint selection** | Current global PCA waypoints are dominated by the immune/neural axis; neuronal subtypes (excitatory vs. inhibitory) don't separate in learned Φ even at K=20. Fix: select waypoints proportionally within each `broad_cell_type` group so every cell type gets representation regardless of size. Implement in `run_training.py` as an option `waypoints_stratified=True`. Try K=50 first — if subtypes still don't separate, implement stratified selection. |

### Human brain atlas — rationale

Allen Brain single nuclei human brain dataset (lein-human-cortex, ~50,625 cells) is the
primary human dataset. Tabula Sapiens was previously combined with it but is excluded in
the new `human_allen_brain` config (data quality / cell type diversity concerns).
Uses the same pipeline as the mouse dataset with `config/human_allen_brain.yaml`.
Expected output: human brain splicing foundation model, cross-species factor comparison.

---

## Human Allen Brain Pipeline (`human_allen_brain`)

Same pipeline steps as mouse; all scripts are config-driven and species-agnostic.
Run from `GenomeBiology_Submission2026/` with these variables:

```bash
CONFIG=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/config/human_allen_brain.yaml
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION
PROC_DIR=$WD/ATSE_mapper/junction_processing_20260616
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
```

**Step 1 — Split ✓ COMPLETE (job 17307013, 2026-06-16):**
```bash
sbatch --partition=cpu pipelines/01_junctions/01_split.sh $CONFIG
# Globs: /gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet/*_junctions_with_barcodes.bed
# Result: 50,625 files → 100 chunks of ~507 files each
# Creates: $PROC_DIR/chunks/ (100 chunks), $PROC_DIR/logs/, $PROC_DIR/results/
```

**Step 2 — Process chunks ✓ COMPLETE (job 17307058, 2026-06-16):**
```bash
sbatch --array=0-99%50 \
    --output=$PROC_DIR/logs/proc_%A_%a.out \
    --error=$PROC_DIR/logs/proc_%A_%a.err \
    /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/01_junctions/02_process.sh \
    $CONFIG $PROC_DIR
# Result: 100/100 chunk_*_results.pkl produced
```

**Step 3 — Merge ✓ COMPLETE (job 17308493, 2026-06-16):**
```bash
sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/01_junctions/03_merge.sh \
    $CONFIG $PROC_DIR
# Note: use absolute path to script when not in GenomeBiology_Submission2026/
# Note: log written to ~/Leaflet-analysis/logs/junc_merge_<JOBID>.out (relative to submission dir)
```
Result: 37,101,422 raw junctions → **525,311** after QC (min 100 reads + seen in ≥10 cells)
Output: `$PROC_DIR/results/final_junctions.pkl`

**Step 4 — ATSE mapping ✓ COMPLETE (job 17308681, 2026-06-16):**
```bash
PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260616
CONFIG=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/config/human_allen_brain.yaml
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION

sbatch --mem=300G --time=2-00:00:00 -J atsemap_human -p cpu,bigmem \
    --output=$WD/logs/atsemap_%j.out \
    --wrap "cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026 && \
        /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python \
        pipelines/02_atsemapper/04_run_atsemapper.py $CONFIG $PROC_DIR"
# Genome: Gencode v45 GRCh38
#   GTF:   /gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf
#   DB:    /gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencodev45.db
#   FASTA: /gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/GRCh38.primary_assembly.genome.fa
```
Result: 515,166 canonical junctions → **39,635 ATSEs** across **15,024 genes**, **153,766 junctions**
(525,311 raw → 515,166 after removing 10,145 non-canonical; splice motifs: 93.2% GT-AG, 4.6% GC-AG, 0.3% AT-AC)
```
ATSE_FILE=$WD/ATSE_mapper/ATSE_files/HUMAN_ALLEN_BRAIN_ATSE_FILE_unanno_also_2026-06-16_17-50-20.txt.gz
```

**Step 5 — Build per-chunk AnnDatas ✓ COMPLETE (2026-06-16):**
```bash
ATSE_FILE=$WD/ATSE_mapper/ATSE_files/HUMAN_ALLEN_BRAIN_ATSE_FILE_unanno_also_2026-06-16_17-50-20.txt.gz
bash pipelines/02_atsemapper/05_make_anndata_slurm.sh $CONFIG $PROC_DIR $ATSE_FILE
# Result: 100/100 chunk_N_anndata.h5ad + chunk_N_audit.json produced
```

**Step 6 — Merge AnnDatas (~1 hr, 128G) — SUBMITTED 2026-06-16:**
```bash
PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260616
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python

sbatch --mem=128G -p cpu \
    --output=$PROC_DIR/logs/merge_adata_%j.out \
    --error=$PROC_DIR/logs/merge_adata_%j.err \
    --wrap "$PYTHON /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/02_atsemapper/06_merge_anndatas.py \
        --input-dir $PROC_DIR/anndatas \
        --output-file $PROC_DIR/anndatas/merged_anndata.h5ad"
# Produces: merged_anndata.h5ad (N cells × J junctions)
```

**Step 7 — Metadata ✓ COMPLETE (2026-06-17):**
```bash
PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
$PYTHON pipelines/03_gene_expression/00_prepare_human_ab_metadata.py
```
Output: `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/metadata_human_allen_brain_combined.csv`
- 47,432 rows (50,625 − 1,985 outlier cells dropped); 6 tissues, 19 cell types
- Donor ages: H200.1030=54y, H200.1023=43y, H200.1025=50y
- `cell_id` = sample_name (matches splice obs_names directly)

**Step 8a — GE processing ✓ COMPLETE (2026-06-17, job 17387549):**
```bash
OUT=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
sbatch --mem=100G -p cpu --output=${OUT}/ge_01_%j.out \
    --wrap "$PYTHON pipelines/03_gene_expression/01_prepare_counts.py config/human_allen_brain.yaml"
```
Output: `processed_data/ab_adata_exons_2026-06-17.h5ad` — 47,432 × 19,162 genes
- Gencode v45, protein_coding + lncRNA, length-norm → library-size norm → log1p

**Step 9 — Alignment ✓ COMPLETE (2026-06-17, job 17387734):**
```bash
HUMAN_OUT=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617
sbatch --mem=150G -p cpu --output=${HUMAN_OUT}/align_%j.out \
    --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/04_align/align_splice_ge_human.py \
        --config config/human_allen_brain.yaml \
        --splice $PROC_DIR/anndatas/merged_anndata.h5ad \
        --ge     $WD/processed_data/ab_adata_exons_2026-06-17.h5ad \
        --output-dir $HUMAN_OUT"
```
Output: `splice_adata_aligned_2026-06-17.h5ad` (46,762 × 127,465) + `ge_adata_aligned_2026-06-17.h5ad`
- Direct intersection on sample_name obs_names (no ID cleaning needed for human)

**Step 9c — Filter ✓ COMPLETE (2026-06-17, job 17388362):**
```bash
sbatch --mem=100G -p cpu --output=${HUMAN_OUT}/filter_aligned_%j.out \
    --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/04_align/09b_filter_aligned.py \
        --splice $HUMAN_OUT/splice_adata_aligned_2026-06-17.h5ad \
        --ge     $HUMAN_OUT/ge_adata_aligned_2026-06-17.h5ad \
        --output-dir $HUMAN_OUT \
        --min-cells-per-junction 25 \
        --min-junctions-per-cell 500"
```
- Junction filter=25 → 106,586 junctions; ATSE re-grouping → 97,144 junctions, 27,082 ATSEs
- Cell filter=500 → 43,442 cells

**Outputs — Human (send to Smriti):**
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad` (7.7G) — 43,442 × 97,144 junctions, 27,082 ATSEs
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/ge_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad` (11G) — 43,442 cells
  - `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/junctions_filtered_minCells25_minJuncs500_2026-06-17.bed` (13M) — junction coords + new ATSE IDs

**Steps 10+ — Training:**
- Run `00_extract_train_splice.py` → `01_generate_params.py` → `02_submit_jobs.py` (same as mouse)
- Smriti needs to produce h5mu train/test split from the 2026-06-17 filtered h5ads before training

---

## Key files produced at each step

| Step | Output |
|------|--------|
| 3 | `results/final_junctions.pkl` |
| 4 | `ATSE_files/{DATASET}_ATSE_FILE_*.txt.gz` |
| 5 | `anndatas/chunk_N_anndata.h5ad`, `anndatas/chunk_N_audit.json` |
| 6 | `anndatas/merged_anndata.h5ad` — 169,503 cells × 142,896 junctions (layers: `cell_by_junction_matrix`, `cell_by_cluster_matrix`) |
| 7 | `metadata_combined.csv` — 179,403 rows, cols: cell_id/age/cell_ontology_class/broad_cell_type/sex/tissue/subtissue |
| 8a | `processed_data/tms_ab_exons_combo_ge_adata_YYYY-MM-DD.h5ad` — 184,187 cells × 18,854 genes |
| 8a | `processed_data/ab_adata_introns_YYYY-MM-DD.h5ad` — 73,363 cells × 18,854 genes (kept, not used downstream) |
| 8b | `scVI/batch_dataset/ge_adata_scVI_2026-05-25.h5ad` — 16G (with TMS/AB batch correction) |
| 8b | `scVI/no_batch/ge_adata_scVI_2026-05-25.h5ad` — 16G (no batch correction) |
| 9 | `MODEL_INPUT/YYYYMMDD/splice_adata_aligned_*.h5ad` — 150,906 cells × 111,903 junctions, 31,993 ATSEs |
| 9 | `MODEL_INPUT/YYYYMMDD/ge_adata_aligned_*.h5ad` — 150,906 cells × 13,119 genes |
| 9c | **⚠️ SUPERSEDED** `MODEL_INPUT/20260526/*_2026-06-13.h5ad` — old partial-ATSE logic; do not use |
| 9c | `MODEL_INPUT/20260526/splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad` (30G) — **132,706 × 79,768 junctions, 24,475 ATSEs** (mouse, ATSE re-derived) |
| 9c | `MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad` (13G) — 132,706 cells, matching genes |
| 9c | `MODEL_INPUT/20260526/junctions_filtered_minCells75_minJuncs500_2026-06-17.bed` (11M) — mouse junction coords + new ATSE IDs |
| 9c | `MODEL_INPUT/20260617/splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad` (7.7G) — **43,442 × 97,144 junctions, 27,082 ATSEs** (human) |
| 9c | `MODEL_INPUT/20260617/ge_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad` (11G) — 43,442 cells, matching genes |
| 9c | `MODEL_INPUT/20260617/junctions_filtered_minCells25_minJuncs500_2026-06-17.bed` (13M) — human junction coords + new ATSE IDs |
| 12-pre | `ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52_lifted_mm39.txt.gz` — 142,895 junctions lifted mm10→mm39 ✓ |
| 10-pre | `MODEL_INPUT/20260526/train_splice_model_ready_2026-06-01.h5ad` — **⚠️ superseded** (105,634 × 111,903, old unfiltered space) |
| 10-pre | `MODEL_INPUT/20260526/train_splice_model_ready_2026-06-14.h5ad` — **pending** (run `00_extract_train_splice.py` on new h5mu; expected 93,012 × 83,593) |
| 10a | `MODEL_INPUT/20260526/model_ready_aligned_splicing_data_*.h5ad` — full 150,906-cell prep (skip if using train split from 10-pre) |
| 10b | `LEAFLETFA_MODELS/YYYY-MM-DD/parameter_combinations.json` — 24 param sets (K=[10,20,50] × batch=[4096,8192] × passes=[5] × prior=[T,F] × waypoints=[T,F]) |
| 10c | `LEAFLETFA_MODELS/YYYY-MM-DD/run_{i}/` — 24 dirs; each has `leafletfa_model.pkl.gz`, `run_summary.csv`, plots |
| 10d | `LEAFLETFA_MODELS/YYYY-MM-DD/all_runs_summary.csv` — ranked comparison across all runs |
| 11 | `LEAFLETFA_MODELS/YYYY-MM-DD/EVALUATION/YYYY-MM-DD/run_N_imputation_results.csv` — per-run per-downsampling-level Pearson r |
| 11 | `EVALUATION/all_runs_imputation_results.csv`, `best_models_imputation_results.csv`, `imputation_accuracy_vs_coverage.png` |
| 12a | `EasySci2024/LeafletFA/model_application/easysci_model_ready_YYYY-MM-DD.h5ad` — 547 × 81,489 (mm10-indexed, zero-filled; after retrain on filtered data) |
| 12b | `model_application/results/easysci_leafletfa_run_N.h5ad`, `_assign_post.csv`, `_pi.csv` |

---

## Cell ID cleaning — align_splice_ge.py approach

`align_splice_ge.py` does not use the `cell_id_cleaners/` modules.
Instead it applies dataset-aware transformations directly:

| Source | Splice obs_names | GE obs_names | Transformation |
|--------|-----------------|--------------|----------------|
| Allen Brain (mouse) | SRR accession — **already canonical** | `exp_component_name` (US-/LS-/SM- prefix) | Map via metadata CSV: `tissue → cell_id` |
| TMS non-3m | cleaned ID (no suffix) — **already canonical** | `{ID}.mm10-plus-0-0` | Strip at first `.` |
| TMS 3m | `A1-B000126-3_39_F-1-1` — **already canonical** | `A1.B000126.3_39_F.1.1-1-1` | `format_tms_cell_id()` (mirrors `metadata_prep.py`) |
| Tabula Sapiens | TBD | TBD | |
| Allen Brain (human) | TBD | TBD | |

**Why splice obs_names are already canonical:** the junction file basenames
(which become splice cell IDs) match the `cell_id` column in `metadata_combined.csv`
exactly, because `metadata_prep.py` was written to produce IDs matching the junction filenames.

---

## RNA Atlas (GSE138734) — FIGURE5 conservation dataset

Human multi-tissue bulk RNA-seq atlas (SRP225193). 2,901 PAIRED-end **runs** across
45 tissue types and primary cell types (tissue + cell type sample_type; excludes cancer
cell lines). Used for FIGURE5: cross-species splicing conservation.

**⚠️ SRR ≠ sample. Read this before analysing anything.** For the cell-type subset:

```
2,228 SRR runs → 316 GSM libraries → 160 RNA samples → 160 cell types → 23 organ systems
```

GSE138734 prepped **every RNA sample three ways**, as consecutive GSMs:

```
GSM4117198  RNAAtlas001_polyA_RNA  [RNA_AT001]   ~60M PE reads   ← in our data
GSM4117199  RNAAtlas001_sRNA       [RNA_AT001]                   ← not downloaded
GSM4117200  RNAAtlas001_total_RNA  [RNA_AT001]   ~125M PE reads  ← in our data (Ribo-Zero Gold)
```

so our paired accessions sit **2 apart**. Verified against the GEO series matrix: our 316 GSMs
are **158 polyA + 158 total RNA + 0 small RNA**, over 160 RNA samples (156 have both libraries).

Three consequences:
1. **Each GSM library is split across 4–16 SRR runs.** Runs are re-sequencing of the *same*
   library — not independent observations, and they carry ~0% of the variance. **Collapse runs →
   GSM by summing raw counts** (never average TPMs) before any model or plot.
2. **Run count is confounded with protocol** (total RNA sequenced ~2× deeper ⇒ more lanes):
   polyA has 4 runs (142 libs) or 8 (16); total RNA has 8 (120), 16 (32), 11 (5), 5 (1).
   At the SRR level this hands total-RNA libraries **~2–4× the ELBO weight** of polyA ones — the
   nascent-RNA-heavy protocol gets over-weighted in exactly the model that fits PSI.
3. **There are NO biological replicates.** The 2 GSMs of a cell type are the *same RNA* through
   two library preps, so "replicate" here means *protocol replicate*. n(independent biological
   samples) = n(cell types) = 160.

Protocol/`sample_id` labels come from the GEO series matrix — the series spans two platforms, so
there are two matrix files and **no combined one**:

```bash
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734
for P in GPL18573 GPL20301; do
  wget -qO- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138734/matrix/GSE138734-${P}_series_matrix.txt.gz" \
    | zcat | grep -E '^!Sample_geo_accession|^!Sample_title'
done > $WD/gsm_titles.txt
```

Do **not** try to infer protocol from the GSM number (e.g. mod 3): the series is missing libraries
(298 sRNA / 295 polyA / 296 total, not 300 each), so the triples have gaps and the phase shifts.
That shortcut was tried and produces nonsense.

See `figures/FIGURE5_conservation/rna_atlas_explore.ipynb` §4b for the full derivation.

**GEO:** GSE138734 | **SRA:** SRP225193
**SraRunTable:** `pipelines/07_conservation/SraRunTable_RNAAtlas.csv`
**Scripts:** `pipelines/07_conservation/`
**Working dir:** `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/`

**Sample breakdown (SraRunTable):** 5,329 total rows; 4,105 PAIRED / 1,224 SINGLE
- tissue: 853 rows (673 PAIRED) — 45 unique tissue types, 15 organ systems
- cell type: 2,868 rows (2,228 PAIRED) — primary cell types
- cancer cell line: 1,492 (excluded)
- cell line: 84 (excluded)

**Strandedness:** fr-firststrand (RF). STAR uses `--outSAMstrandField intronMotif`;
regtools uses `-s XS` to follow the XS tag STAR sets from intron motif.

### Step A: Generate manifest + download commands

Run from `GenomeBiology_Submission2026/`:

```bash
FASTQ_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs
mkdir -p $FASTQ_DIR $FASTQ_DIR/logs

# Writes manifest.csv + prints fasterq-dump commands to download_commands.sh
python pipelines/07_conservation/00_download_gse138734.py \
    pipelines/07_conservation/SraRunTable_RNAAtlas.csv \
    --outdir $FASTQ_DIR \
    --sample-types tissue "cell type" \
    > $FASTQ_DIR/download_commands.sh

# Build SRR list for SLURM array (1 SRR per line, no header)
tail -n +2 $FASTQ_DIR/manifest.csv | cut -d',' -f1 > $FASTQ_DIR/srr_list.txt
wc -l $FASTQ_DIR/srr_list.txt   # should be 2901
```

**⚠️ Notes:**
- `mkdir -p $FASTQ_DIR` must run before the python redirect — bash opens the output
  file before the script runs, so the directory must already exist.
- Always run from `GenomeBiology_Submission2026/` and pass relative paths to the script.

### Step B: Submit SLURM download array

**✓ COMPLETE (2026-06-17): 2,874/2,901 succeeded (98.9%); 27 failed — see Step B-retry below.**

```bash
# Use full absolute path to the script — sbatch resolves relative to submission dir
sbatch --array=1-2901%20 \
    /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/07_conservation/02_download_array.sh
```

- Caps at 20 concurrent downloads to avoid hammering NCBI.
- Uses `module load SRA-Toolkit/3.1.1-gompi-2023b` for fasterq-dump.
- Uses `pigz` from `/gpfs/commons/home/kisaev/miniconda3/envs/spipe/bin/pigz`.
- Logs: `$FASTQ_DIR/logs/dl_{JOBID}_{TASKID}.out/err`
- Each sample: `$FASTQ_DIR/{SRR}/{SRR}_R1.fastq.gz` + `_R2.fastq.gz`

**Space estimate:** ~3–6 TB total. Check first: `df -h /gpfs/commons/groups/knowles_lab/`

**To check completeness:**
```bash
FASTQ_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs
complete=0; empty=0
for d in $FASTQ_DIR/SRR*/; do
    srr=$(basename $d)
    [[ -f "$d${srr}_R1.fastq.gz" && -f "$d${srr}_R2.fastq.gz" ]] && ((complete++)) || ((empty++))
done
echo "Complete: $complete / Empty/failed: $empty"
```

### Step B-retry: Redownload 27 failed samples

**Failure cause:** `fasterq-dump` error code 3 — reads all data successfully but fails during
temp-file concatenation (`execute_concat_un_compressed: rcNotFound`). Transient NFS I/O issue;
simple retry fixes it.

**Failed SRRs (27):** SRR10261873, SRR10261878–83, SRR10262005–06, SRR10264361–68,
SRR10264400–04, SRR10264407–11

**Retry list already written:** `$FASTQ_DIR/srr_retry_list.txt` (27 lines, 2026-06-17)

```bash
FASTQ_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs

# Regenerate retry list if needed (finds dirs missing R1 or R2)
for d in $FASTQ_DIR/SRR*/; do
    srr=$(basename $d)
    [[ ! -f "$d${srr}_R1.fastq.gz" || ! -f "$d${srr}_R2.fastq.gz" ]] && echo $srr
done > $FASTQ_DIR/srr_retry_list.txt

# Make a retry script pointing at the retry list
sed 's|srr_list.txt|srr_retry_list.txt|' \
    /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/07_conservation/02_download_array.sh \
    > $FASTQ_DIR/02_download_retry.sh

sbatch --array=1-27%10 $FASTQ_DIR/02_download_retry.sh
```

### Step C: Align + extract junctions

**✓ Dry run passed 2026-06-17 (14,506 jobs: 2,901 × 5 rules). Ready to submit.**

Pre-flight checklist (all done 2026-06-17):
- `bams/` dir created ✓
- `junctions/` dir created ✓
- `pipelines/07_conservation/logs/` dir created ✓
- Activate `python3ENV` before submitting (snakemake lives there) ✓

```bash
conda activate python3ENV
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/07_conservation

# Dry run (verify DAG, no jobs submitted):
bash run_snakemake.sh --dry-run

# Real run (capped at 16 concurrent SLURM jobs):
sbatch run_snakemake.sh
```

Runs STAR (GRCh38 v45, reusing existing `human_allen_brain` index at
`/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/star`) → `add_cb_tag`
(stamps SRR accession as CB barcode) → `samtools index` → `regtools junctions extract`
→ per-sample `*_junctions_with_barcodes.bed`.

- Rule logs: `RNA_ATLAS/GSE138734/logs/snakemake_YYYYMMDD/slurm-{JOBID}.out/err`
- Output: `RNA_ATLAS/GSE138734/junctions/{SRR}_junctions_with_barcodes.bed`
- 3 SRRs (SRR10264361/62/64) will fail first pass if still downloading — resubmit after they finish; `--rerun-incomplete` picks them up automatically.

**To check completion:**
```bash
ls /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/junctions/*_junctions_with_barcodes.bed | wc -l  # should be 2901
```

### Step D: Feed into ATSEmapper (after junctions complete)

**Progress (as of 2026-07-08):**

| Step | Status | Result |
|---|---|---|
| D1 split | ✓ done | 2,228 junction files → 97 chunks |
| D2 process | ✓ done (array 18718043, 0-96%50) | 97/97 COMPLETED |
| D3 merge | ✓ done (inline `bash`) | 3,985,398 → **383,792** junctions after QC |
| D4 ATSE calling | ✓ done (job 18719888, 2h01m) | **36,800 ATSEs** / 133,132 junctions / 14,272 genes |
| D4b make_anndata + merge | ✓ done (array 18732619, 91 chunks) | **2,228 × 133,132** `merged_anndata.h5ad` |
| D4b count audit | ⏳ running | — |
| D7 metadata join | ⏭ next | script ready |
| D8 GE (featureCounts) | ☐ TODO | critical path (D9 needs GE) |
| D9 filter + ATSE regroup | ☐ TODO | needs GE+align first |
| D6 reference ATSE catalog | ☐ TODO | independent |

**Why this dataset matters:** ideal **validation testbed before scaling to >100k cells**
while keeping cell-type diversity — 2,228 samples, **160 primary cell types** (~14
replicates each), 23 organ systems. Small enough to run LeafletFA on an **interactive GPU
node, unbatched (all samples at once)** → a clean test that **minibatching == full-dataset
performance** before relying on it at scale. Replicated cell types also give LeafCutter a
trustworthy DS reference to validate LeafletFA against (the original goal of this track).

**Config:** `config/rna_atlas.yaml` (created 2026-07-07). Reuses the **same GRCh38 v45
reference files as `human_allen_brain`** (gtf/db/fasta under `BRAIN_NeMO/human-reference/
gencode/`). This is valid because Step C STAR-aligned the atlas to the human_allen_brain
index, so atlas junction coordinates are already in GRCh38 v45 — no liftover. ATSEs are
called **de novo** on the atlas (04_run_atsemapper.py always calls fresh; it does not
project onto the existing human brain ATSE file). Each SRR = one pseudocell.

**PRIMARY CELL TYPES ONLY.** The `junctions/` dir holds all 2,901 downloaded files
(2,228 `sample_type == "cell type"` + 673 `tissue` bulk mixed tissues). We restrict to
the **2,228 primary cell-type** samples, excluding bulk mixed tissues (cancer/normal cell
lines were already excluded at download). The config points `sources.rna_atlas.junction_filelist`
at an explicit list rather than globbing `junction_dir`:

```bash
# Build the 2,228-sample filelist (sample_type == "cell type" from the manifest):
python3 -c "
import pandas as pd, os
WD='/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734'
m = pd.read_csv(os.path.join(WD,'fastqs','manifest.csv'))
ct = m[m['sample_type']=='cell type']
paths = [os.path.join(WD,'junctions',f'{s}_junctions_with_barcodes.bed') for s in ct['srr']]
paths = [p for p in paths if os.path.isfile(p)]
open(os.path.join(WD,'junctions_celltype_only.txt'),'w').write('\n'.join(paths)+'\n')
print(len(paths),'files')   # 2228
"
```

**ATSEmapper params** in the config match `human_allen_brain.yaml` for consistency
(min_intron 50, max_intron 500000, min_junc_reads 100, min_num_cells 10, unanno_also).

#### D1–D3: Junction split → process → merge (Steps 01_junctions) — ✓ done 2026-07-07/08

Actual run used `PROC_DIR=.../ATSE_mapper/junction_processing_20260707`.

```bash
cd GenomeBiology_Submission2026/pipelines/01_junctions

# D1 — discover the 2,228 files + split into chunks (runs inline, quick)
bash 01_split.sh ../../config/rna_atlas.yaml
# → "rna_atlas: 2228 junction files from pre-collected list"
# → creates ATSE_mapper/junction_processing_<TODAY>/ with 97 chunks (NOT 100 — splitter
#   makes fewer; ALWAYS check `ls chunks/ | wc -l` and use the real 0..N-1 range below)

PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/ATSE_mapper/junction_processing_20260707

# D2 — process each chunk. BOTH the --array flag AND $PROC_DIR are required:
#   without --array, SLURM_ARRAY_TASK_ID is unset → every task reads chunks/chunk_.txt
#   without $PROC_DIR, the script exits immediately with a usage error
sbatch --array=0-96%50 02_process.sh ../../config/rna_atlas.yaml $PROC_DIR

# D3 — merge chunk pkls → results/final_junctions.pkl
sbatch 03_merge.sh ../../config/rna_atlas.yaml $PROC_DIR
```

**Result:** 97/97 chunks COMPLETED → merge: `After QC (min_cells=10, min_reads=100):
3,985,398 → 383,792 junctions`.

**⚠️ QOSMinMemory gotcha (this cluster):** the `bigmem` partition/QOS has a **500G
minimum-memory floor** — any job requesting <500G on `bigmem` is hard-rejected and sits
`PD (QOSMinMemory)`. 02/03 request 32G/128G, so `03_merge.sh` was pinned to
`--partition=cpu` (not `cpu,bigmem`). If a merge ever needs >500G use `--mem=500G -p bigmem`.

To skip the batch queue entirely, the merge can run **inline** (it needs no SLURM env
vars — same pattern as 01_split.sh):

```bash
bash 03_merge.sh ../../config/rna_atlas.yaml $PROC_DIR
# or with a SLURM allocation but no batch wait:
# srun --mem=128G -c 8 -p cpu --pty bash 03_merge.sh ../../config/rna_atlas.yaml $PROC_DIR
```

#### D4: Call ATSEs de novo (Step 02_atsemapper/04) — ✓ done 2026-07-08

After `results/final_junctions.pkl` exists:

```bash
cd GenomeBiology_Submission2026
CONFIG=config/rna_atlas.yaml
PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/ATSE_mapper/junction_processing_20260707

# -p cpu (300G < 500G bigmem floor, so bigmem would QOSMinMemory-reject)
sbatch --mem=300G --time=2-00:00:00 -J atsemap -p cpu \
    --wrap "python pipelines/02_atsemapper/04_run_atsemapper.py $CONFIG $PROC_DIR"
```

**Result (job 18719888, 2h01m, 1.6G peak RSS):** 36,800 ATSE groups, 133,132 junctions,
14,272 genes (avg 2.6 ATSEs/gene, max 20). Output:
`ATSE_files/RNA_ATLAS_ATSE_FILE_unanno_also_2026-07-08_13-03-14.txt.gz`

#### D4b: Build AnnData (05_make_anndata → 06_merge → audit) — ✓ merge done 2026-07-08

```bash
cd GenomeBiology_Submission2026
PROC_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/ATSE_mapper/junction_processing_20260707
ATSE_FILE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/ATSE_mapper/ATSE_files/RNA_ATLAS_ATSE_FILE_unanno_also_2026-07-08_13-03-14.txt.gz

# submits a SLURM array, one task per chunk (91 tasks, %30 throttle)
bash pipelines/02_atsemapper/05_make_anndata_slurm.sh config/rna_atlas.yaml $PROC_DIR $ATSE_FILE

# after the array finishes (squeue empties):
ANNDATA_DIR=$PROC_DIR/anndatas
ls $ANNDATA_DIR/*.h5ad | wc -l    # expect 97 chunk anndatas

# ALWAYS run the merge DETACHED — never in a foreground shell (see gotcha below)
cd pipelines/02_atsemapper
sbatch --mem=128G -c 4 -p cpu -J merge_ada -t 4:00:00 \
    --wrap "python 06_merge_anndatas.py --input-dir $ANNDATA_DIR --output-file $ANNDATA_DIR/merged_anndata.h5ad"

# only AFTER the merge job shows COMPLETED:
python ../../tests/test_count_audit.py $ANNDATA_DIR/merged_anndata.h5ad $ANNDATA_DIR
```

**Result (array 18732619):** `Merge complete: 2,228 cells × 133,132 junctions`, layers
`cell_by_junction_matrix` + `cell_by_cluster_matrix` → `anndatas/merged_anndata.h5ad`.

**⚠️ Truncated-h5ad gotcha (hit 2026-07-08):** the merge writes `X` + two sparse layers
(several hundred MB). If it's run in a foreground shell and interrupted (Ctrl-C, ssh
disconnect) **during the save**, it leaves a corrupt h5ad that fails later with:

```
AttributeError: 'dict' object has no attribute 'shape'
Error raised while reading key '' of <class 'h5py._hl.files.File'> from /
```

Cause: AnnData writes `X` first, then layers. An interrupted write leaves layer groups
missing `indices`/`indptr` and — critically — missing the `encoding-type` attr, so anndata
reads the group back as a plain `dict`. **Not** a code bug; the file is simply incomplete.

Check a suspect file WITHOUT loading it — every layer must have `encoding-type=csr_matrix`
plus `data` + `indices` + `indptr`:

```bash
h5ls -v $ANNDATA_DIR/merged_anndata.h5ad/layers
# or:
python3 -c "
import h5py,sys; f=h5py.File(sys.argv[1],'r')
for k in f['layers']:
    g=f['layers'][k]; print(k, dict(g.attrs), list(g.keys()))
" $ANNDATA_DIR/merged_anndata.h5ad
```

Fix: `rm` the partial file and re-run the merge via `sbatch` (chunk anndatas are intact —
nothing upstream needs redoing).

**obs_names = SRR accession** (`extract_cell_id` strips `_junctions_with_barcodes` from the
file stem), so each SRR is one pseudocell. This is what makes the D7 metadata join trivial.

**⚠️ Before the GE-align stage:** no SRR `cell_id_cleaner` is registered yet — add an
`rna_atlas` cleaner under `pipelines/04_align/cell_id_cleaners/` (not needed for ATSE
calling or the splice AnnData build).

#### D5: Junction-count comparison — bulk primary cell types vs SC atlases (report this)

Interesting resource stat: how many unique splice junctions the **2,228 bulk primary
cell-type** samples recover vs the single-cell atlases built from tens of thousands of
cells. Bulk samples are deep, so even a modest number of samples recovers many more
QC-passing junctions than SC.

| Dataset | Cells / samples | Unique junctions | ATSEs | Pipeline stage |
|---|---|---|---|---|
| **RNA atlas (bulk, primary cell types)** | 2,228 samples | **133,132** (in ATSEs; 383,792 QC'd) | **36,800** (14,272 genes) | post-ATSE-mapping (2026-07-08) |
| Mouse SC atlas (TMS + Allen Brain Mouse) | 132,706 cells | 79,768 | 24,475 | model-ready filtered h5ad |
| Human SC atlas (Allen Brain Human Cortex) | 43,442 cells | 97,144 | 27,082 | model-ready filtered h5ad |

**Headline:** just 2,228 bulk primary cell-type samples yield **more ATSEs (36,800) and
junctions-in-ATSEs (133,132)** than either SC atlas built from tens of thousands of cells
— bulk depth recovers more of the splicing landscape per sample.

**⚠️ Caveat — stage not identical:** the atlas number is post-ATSE-mapping (pre-anndata
cell filtering); the SC numbers are model-ready (post cell-filtering + ATSE re-grouping).
Atlas raw pre-QC total was 3,985,398 junctions → 383,792 after QC (≥100 reads, ≥10
samples) → 133,132 assigned to 36,800 ATSEs. Report with the stage/QC thresholds stated.
This is the coverage story from
[[project_rna_atlas_leafcutter]]: bulk depth → many junctions per sample, motivating why
imputation is needed for SC.

#### D6: TODO — reference-genome ATSE catalog (annotation-only BED)

Build an **ATSE map directly from the Gencode reference annotation** (all annotated
splice sites/exons) — i.e. enumerate every possible ATSE in the baseline reference genome,
independent of any dataset. Gives a fixed "annotation ATSE catalog" BED to annotate/match
data-derived ATSEs against (mouse, human SC, and the bulk atlas).

**Hard part — the ±1 splice-site coordinate offset (exon boundary vs intron boundary):**
regtools/STAR junction coordinates are **intron** boundaries (first/last intronic base),
while the GTF stores **exon** boundaries. They differ by ±1 (exon_end vs intron_start,
exon_start vs intron_end), and conventions are 0- vs 1-based and strand-dependent. The
annotation-derived junctions must be shifted to match the data junction coordinate
convention *exactly*, or annotated ATSEs won't line up with observed ones. Reuse
ATSEmapper's own splice-site handling (`JunctionAnalyzer` uses `tolerance=1`) rather than
re-deriving the offset. Deliverable: a reference ATSE BED (coords + ATSE ids) usable as a
shared annotation across all datasets.

#### D7: Cell/sample metadata (join manifest → AnnData obs)

The merged AnnData obs is keyed by **SRR accession** (05_make_anndata's `extract_cell_id`
turns `SRR…_junctions_with_barcodes.bed` → `SRR…`), which joins 1:1 to `fastqs/manifest.csv`
`srr`. `pipelines/07_conservation/03_prepare_atlas_metadata.py` builds the metadata and
attaches it:

```bash
cd GenomeBiology_Submission2026/pipelines/07_conservation
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734
ANNDATA=$WD/ATSE_mapper/junction_processing_20260707/anndatas/merged_anndata.h5ad

python 03_prepare_atlas_metadata.py $WD/fastqs/manifest.csv $WD/metadata_rna_atlas_celltype.csv \
    --anndata $ANNDATA
```

Adds obs columns: `cell_type` (source_name; **160 primary cell types**, ~14 replicate
samples each), `organ_system` (**23 systems**), `gsm` (**316** GEO samples grouping
replicate SRRs), `sample_type` (all "cell type"). **`cell_type` is the natural DS grouping
variable** for LeafletFA-vs-LeafCutter (organ_system = coarser grouping).

#### D8: Gene expression count matrix (featureCounts from existing BAMs)

STAR BAMs already exist (`bams/{SRR}/Aligned.sortedByCoord.out.bam` — use the **plain**
BAM, NOT `.CB.bam`; the CB copy exists only so ATSEmapper sees one pseudocell per sample).
STAR ran **without** `--quantMode GeneCounts`, so there are no ReadsPerGene tabs → quantify
with **featureCounts** over the **same gencode.v45 GTF** used for the STAR index and ATSE
mapping, so GE and splicing share one annotation. Restricted to the **2,228 primary
cell-type** BAMs (all present, verified) to match the splice AnnData scope.

Scripts (all in `pipelines/07_conservation/`):
`04_prepare_bam_list.py` → `04_check_strandedness.sh` → `04_featurecounts_array.sh` →
`05_build_ge_anndata.py`

```bash
cd GenomeBiology_Submission2026/pipelines/07_conservation
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734

# 1) BAM list (2,228 primary cell-type BAMs → $WD/bams_celltype_only.txt)
python 04_prepare_bam_list.py

# 2) VERIFY strandedness before the full run (a wrong -s silently discards most fragments).
#    Runs featureCounts at -s 0/1/2 on one BAM; pick the highest assigned%.
bash 04_check_strandedness.sh SRR10261826
#    Expect -s 2 to win (fr-firststrand / RF). If not, edit STRAND= in the array script.

# 3) featureCounts array: 2,228 BAMs / BATCH_SIZE=50 → 45 tasks (0-44)
sbatch --array=0-44%15 04_featurecounts_array.sh
#    -p --countReadPairs (count fragments), -s 2, -t exon -g gene_id, --primary, -T 8
#    Idempotent: finished batches are skipped, so failed tasks can just be resubmitted.

# 4) combine batches → GE AnnData (obs=SRR, var=gene_id, layer 'counts')
ANNDATA_DIR=$WD/ATSE_mapper/junction_processing_20260707/anndatas
python 05_build_ge_anndata.py --splice $ANNDATA_DIR/merged_anndata_with_metadata.h5ad
#    --splice prints the GE↔splice obs overlap (expect 2,228 intersect, 0 either-only)
```

Output: `$WD/gene_expression/ge_adata_rna_atlas.h5ad` (2,228 samples × ~62k genes, raw
integer fragment counts in `X` and `layers['counts']`, `var['gene_name']`, `var['gene_length']`).

**Strandedness confirmed empirically 2026-07-09** on SRR10261826:
`-s 2` = 79.9% assigned, `-s 0` = 77.5%, `-s 1` = **4.4%** → fr-firststrand, `-s 2` correct.
(`-s 0` sits below `-s 2` because unstranded quant discards overlapping-feature fragments
as ambiguous.)

**QC of the featureCounts run (job 18734890, 45/45 tasks COMPLETED):**

⚠️ **Compute assignment rate against PRIMARY alignments, not all records.** The `.summary`
denominator includes `Unassigned_Secondary` (multi-mapping records that `--primary`
correctly declines to count). Dividing by the raw total badly understates the rate
(looked like 144 samples <50%; the true number is 28).

```
assigned / (total - Unassigned_Secondary):  median 83.6%,  p5 67.6%,  28 samples <50%
Unassigned_Secondary % of records:          median 18.8%,  48 samples >50% (max 76%)
assigned fragments per sample:              median 10.8M,  p5 3.75M,  min 1.08M
```

Two real observations (not pipeline errors):
- **High multimapping in a subset** (48 samples >50% secondary; one CD14+ monocyte had STAR
  reporting 46.9% multi-locus reads) → genuinely less usable depth for those samples.
- **Immune cells have lower exonic assignment**: CD14+ Monocytes 36.5%, CD56+ NK 40.5%,
  CD3+ T cells 48.7% (vs 83.6% median). Their unassigned reads are mostly `NoFeatures`
  (non-exonic).

> **⚠️ CORRECTED 2026-07-11 — the old explanation here was wrong.** This note used to say the
> low exonic assignment was "the signature of Ribo-Zero total-RNA libraries" and that
> "GSE138734 mixes total-RNA and polyA protocols", implying Ribo-Zero was a *subset* of samples
> (the immune ones). Both halves are wrong:
>
> - **Ribo-Zero is not a subset — it is half of everything.** *Every* RNA sample was prepped as
>   both a polyA and a Ribo-Zero total-RNA library: 158 + 158 of our 316 GSMs, one of each per
>   cell type (see the dataset header above). So protocol is uniform across cell types and
>   **cannot explain why monocytes/NK/T specifically are low** — their polyA libraries are polyA
>   like everyone else's.
> - **The cause of the immune-specific drop is therefore still unexplained.** Candidates: genuinely
>   higher nascent/intronic content in those cell types, higher multimapping (see above — the two
>   observations overlap), or rRNA carry-over. To settle it, compare exonic assignment
>   **by protocol × cell type**, not by cell type alone; if immune cells are low in *both* their
>   polyA and their total-RNA library, it is biology, not chemistry.
>
> The consequence that *does* survive: Ribo-Zero libraries carry more intronic/nascent signal, and
> that shifts PSI for intron-retention-like events across **half the dataset** — a systematic
> effect on the quantity LeafletFA models, not a quirk of a few immune samples.

**→ D9 implication:** don't set coverage floors from the median sample. Inspect the `--scan`
retention curve **stratified by cell_type**, or the monocyte / NK / T-cell groups (exactly
the ones you want for DS) may be silently dropped.

**⚠️ Two things `05_build_ge_anndata.py` must do, or D9 breaks (one silently):**
1. **`var['gene_name']` (parsed from the GTF).** 09b's `filter_ge()` matches splice
   `.var['gene_name']` (symbols) against GE `.var['gene_name']` if present, **else GE
   `.var_names`** — which for us are versioned gene_ids (`ENSG…​.16`). Symbols vs versioned
   ids match nothing → GE silently filtered to **ZERO genes**, and 09b's gene-overlap
   assert passes trivially because both sides are empty.
2. **obs order must match splice.** 09b asserts `list(splice.obs_names) == list(ge.obs_names)`
   *after* filtering, but its `filter_ge()` subsets GE with `.isin()`, preserving GE's own
   order. Splice obs order is chunk-concat order (`SRR10261826, …828, …827, …`), NOT manifest
   order. **Always pass `--splice`** — it reorders GE obs to the splice order.

**Why no align step is needed:** both AnnDatas are keyed by bare SRR accession and 09b
subsets GE to the splice cells itself, so `merged_anndata_with_metadata.h5ad` + the GE h5ad
feed straight into D9. `align_splice_ge.py` and the `rna_atlas` cell_id_cleaner are **not
required** on this path.

All h5ad writes go through `.tmp` + atomic rename (see the D4b truncated-h5ad gotcha).

#### D9: coverage filter + ATSE re-grouping (match SC pipeline)

**Stratified scan run 2026-07-09** (`06_scan_thresholds_by_celltype.py` — replicates 09b's
filter semantics exactly, broken down by cell_type). Two findings that set the thresholds:

**1. The CELL filter is a non-issue for bulk.** Every one of the 160 cell types — including
the low-GE immune types (CD14+ Monocytes med 38k, CD3+ T 45k, CD56+ NK 43k junctions/cell) —
retains **100% of cells even at min-junctions-per-cell = 500**. Bulk samples carry tens of
thousands of junctions each (pooled median ~57,600), so the SC-style cell filter is trivially
met. The earlier worry about dropping immune cell types was about GE assignment depth, not
splice coverage — splice coverage is deep for all types. Set the cell filter LOW (e.g. 500,
same as SC) purely as a safety floor; it removes nothing here.

**2. The JUNCTION filter is the only real lever**, and note it starts at t≥10 for free:
ATSE mapping already required `min_num_cells=10`, so t≤10 filters nothing. Retention curve
(junctions kept / ATSEs retaining ≥1 junction), out of 2,228 samples:

```
  t=10   133,132 (100.0%)   36,800 ATSEs
  t=20   132,641 ( 99.6%)   36,786
  t=50   129,736 ( 97.4%)   36,611
  t=100  118,346 ( 88.9%)   36,027
  t=223  96,228  ( 72.3%)   34,161    (~10% of samples)
  t=446  76,365  ( 57.4%)   31,704    (~20% of samples)
```

Recommendation: **junction filter t≈50–100** (keep junctions seen in ≥50–100 of 2,228
samples) — drops the rare/noisy tail while keeping ≥97/89% of junctions and nearly all
ATSEs; cell filter = 500 (safety floor). Think in fraction-of-samples, NOT the SC absolute
75. Confirm with a final `--scan` on the aligned pair before committing.

Inputs are READY: splice `merged_anndata_with_metadata.h5ad` + GE `ge_adata_rna_atlas.h5ad`,
both keyed by bare SRR. No align step / cell_id_cleaner needed (09b subsets GE itself).

**✓ DONE 2026-07-11 (job 18774633, 1m47s), t=50 junction / M=500 cell:**
```bash
OUTDIR=$WD/MODEL_INPUT/20260709
sbatch --mem=150G -p cpu -J filt_atlas --wrap "python pipelines/04_align/09b_filter_aligned.py \
    --splice <merged_with_metadata> --ge <ge_adata_rna_atlas> --output-dir $OUTDIR \
    --min-cells-per-junction 50 --min-junctions-per-cell 500"
```
Outputs in `$WD/MODEL_INPUT/20260709/`:
- `splice_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad` — **2,228 × 128,138**
  junctions, **35,614 ATSEs** (regrouped; min 2 junc/ATSE, median 3, max 148, **0 singletons**)
- `ge_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad` — **2,228 × 13,993**
  genes (= genes with surviving junctions; NON-zero confirms the gene_name fix worked)
- **All 2,228 cells retained** (bulk depth → cell filter drops nothing, as the scan predicted).

This is the model-ready pair. Next: train LeafletFA (interactive GPU, unbatched — the
batched-vs-unbatched equivalence test this dataset is for).

**TODO — original SC-parity note (still applies):**

Before training, apply the SAME coverage filtering + second-round ATSE mapping the SC
atlases got, so decent-coverage junctions go into the model. Reuse
`pipelines/04_align/09b_filter_aligned.py` verbatim — it does both:
  1. Junction filter (`--min-cells-per-junction`) + cell filter (`--min-junctions-per-cell`),
     recomputing cell_by_cluster_matrix (PSI denominator) over surviving junctions.
  2. **Second round of ATSE mapping** — `_regroup_atses()` re-runs
     `ATSEAnalyzer.build_splice_graph()` on surviving junctions, drops singleton junctions,
     re-forms ATSEs (≥2 junctions), preserves old id as `original_event_id`.
     (SC results: mouse 88,833→79,768 junc / 9,065 singletons; human 106,586→97,144.)

Sequence: splice AnnData (D1–D4) → GE (D8) → align splice+GE (both keyed by SRR → trivial
intersect) → **09b filter+regroup** → train. `--ge` is required by the script.

**⚠️ Bulk thresholds ≠ SC thresholds.** Mouse used `--min-cells-per-junction 75`
(≈0.06% of 132,706 cells) — for 2,228 samples that's ~1.3 samples, far too permissive;
and `--min-junctions-per-cell 500` is trivially met by deep bulk samples. **Run `--scan`
first** and pick cutoffs from the atlas retention curve (think in fraction-of-samples,
not the SC absolute counts).

```bash
# scan only (no files written) — pick thresholds from the printed retention table:
python pipelines/04_align/09b_filter_aligned.py \
    --splice <atlas_splice_aligned.h5ad> --ge <atlas_ge_aligned.h5ad> \
    --output-dir <MODEL_INPUT_dir> --scan
```

Check first that the atlas splice AnnData var carries `event_id` + `gene_id`/`gene_name`
(required by `_regroup_atses`); 05_make_anndata should set these from the ATSE file.

---

#### D10: Export ATSE junctions to BED — ✓ done 2026-07-11

`pipelines/08_leafcutter_ds/01_export_atse_junctions_bed.py` reads only var from the
FILTERED splice h5ad and writes a BED of every junction with its ATSE grouping.

```bash
SPLICE_FILT=$WD/MODEL_INPUT/20260709/splice_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad
python pipelines/08_leafcutter_ds/01_export_atse_junctions_bed.py \
    $SPLICE_FILT $WD/atlas_atse_junctions_filtered.bed
```

Output `atlas_atse_junctions_filtered.bed`: **128,138 junctions / 35,614 ATSEs**, all 24
standard chromosomes. Headerless BED6+3, columns:
`chrom start end event_id score strand junction_id gene_id gene_name` — col4 = ATSE id
(junctions in one ATSE share it), col5 = ATSE junction count.

⚠️ Coordinates are emitted verbatim (regtools intron coordinates), NOT shifted to exon
boundaries / strict BED half-open. Fine for internal use + LeafCutter (parses our IDs
directly); reconcile the ±1 offset before intersecting with an annotation catalog (D6).

Run on the FILTERED object so `event_id` reflects the post-regroup ATSEs (the pre-filter
`original_event_id` is preserved in var if you need it).

---

## FIGURE 2 (benchmark) — TODO: rebuild the simulation for robustness + calibration

**Goal:** Figure 2 is about **robustness and calibration**, so the simulated data must be
generated under **both** likelihoods the model can assume — **Binomial** and **Beta-Binomial**
— and we then check whether LeafletFA recovers the truth and is *calibrated* under each
(i.e. does the Beta-Binomial model correctly recover overdispersion, and does the Binomial
model mis-calibrate when the truth is overdispersed?).

**Design (agreed 2026-07-11):**
- Take **one tissue** (e.g. **heart**) from **Tabula Muris Senis** — small, clean, avoids
  whole-atlas cost.
- Simulate cell types following **K latent clusters / programs** (the LeafletFA `phi`/`psi`
  structure), rather than reusing observed labels.
- Simulate a subset of **junctions that differ between programs** (the ground-truth
  differential set); the rest are non-differential (null) — this gives the calibration axis
  (FPR / FDR on the nulls, power on the true positives).
- Condition on **real ATSE (cluster) total counts** so coverage/depth is realistic.

### The old code exists — start here, don't rewrite

`LeafletSC/simulation/simulate_counts.py`  (repo: `/gpfs/commons/home/kisaev/LeafletSC/`;
also installed in the `LeafletSC` conda env)

```python
sim.simulate_junc_counts(cluster_counts, junc_info, cell_types=None,
                         psi_prior_shape1=0.5, psi_prior_shape2=0.5)
# -> simulated_counts, cell_types, cell_type_psi, cluster_labels
```
- Draws per-junction × per-cell-type PSI from `Beta(0.5, 0.5)` (lines 84, 98)
- Labels each intron cluster differential vs not (`cluster_labels`, line 64) — this is the
  ground truth for calibration
- Conditions on the observed cluster counts (realistic depth)

**Example usage:** `AdditionalDatasets/CZhang_2018_mouse/Leaflet/03_simulate_data.ipynb`

### ⚠️ What's MISSING and must be added

**The existing simulator emits counts under a pure BINOMIAL only** (line 121:
`torch.distributions.binomial.Binomial(...)`). The `Beta(0.5,0.5)` is only used to *generate
psi*, not to add overdispersion to the counts — so there is currently **no Beta-Binomial
emission**, which is precisely the arm Figure 2 needs.

Add a Beta-Binomial emission path:
```python
# Beta-Binomial: overdispersed counts around psi with concentration `conc`
p = Beta(psi * conc, (1 - psi) * conc).sample()   # per cell x junction
counts = Binomial(total_count=cluster_counts, probs=p).sample()
# conc -> inf recovers the Binomial case (sanity check the two arms agree in that limit)
```
Sweep `conc` (e.g. 1, 5, 20, 100, inf) to span strong → no overdispersion. The `inf` arm must
reproduce the existing Binomial simulator exactly — use that as a regression test.

**Also swap the program structure:** the current function keys PSI off observed `cell_types`;
for Figure 2 we want K latent programs with cells assigned via `phi` (Dirichlet), so the truth
matches LeafletFA's generative model.

---

## Parse Bio workflow (new — run from scratch)

Parse Bio junction files are in a different format (STAR SJ.out.tab, one per
barcode/cell). The same pipeline applies but with `config/parsebio_pbmc.yaml`.
Lower ATSEmapper thresholds are set in that config.
LeafletFA is initialized from the mouse SS2 prior (transfer learning) rather
than trained from scratch — see `leafletfa.transfer_model` in the config.

---

## Tests

Run these at the checkpoints shown above. All are standalone scripts.

```bash
# After merge:
python tests/test_count_audit.py <merged.h5ad> <anndata_dir>

# After alignment:
python tests/test_cell_id_alignment.py <splice.h5ad> <ge.h5ad>
python tests/test_metadata_coverage.py <splice.h5ad> <metadata.csv>
```
