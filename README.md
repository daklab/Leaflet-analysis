# Leaflet-analysis

This repository contains the analysis pipelines and scripts for the results presented in:
**[Aging-associated alternative splicing programs conserved between human and mouse tissues](https://www.biorxiv.org/content/10.64898/2025.12.30.697080v2)** (bioRxiv, 2025).

The core methodology utilizes the **LeafletFA** model.

* **Source Code:** [LeafletFA Repository](https://github.com/daklab/LeafletFA)
* **Utility Functions:** [LeafletFA-utils](https://github.com/daklab/LeafletFA-utils)


## Workflow Overview

Our study's methods includes the following steps:

1. **Junction Calling:** Raw single-cell BAM files to Splice Junctions (Snakemake).
2. **Junction Processing:** Filtering and merging across batches.
3. **ATSE Mapping:** Mapping Alternative Transcript Structure Events (ATSE).
4. **Anndata Integration:** Aligning splicing data with pre-processed total gene expression.
5. **Mouse Foundation Training:** Training LeafletFA on the mouse dataset.
6. **Cross-Species Comparison:** Identifying conserved splice junctions between mouse and human.
7. **Transfer Learning:** Applying the trained mouse LeafletFA model to human data.


## 1. Data Processing (BAM to Junctions)

We utilized Snakemake pipelines to process raw data from several major single-cell atlases.

### Mouse Datasets

* **Allen Brain:** `LeafletFA_Submission2025/AllenInst/mouse_brain_dev_2021/dataprocessing/snakemake/run_snakemake.sh`
* **Tabula Muris Senis:** (Processed separately by age: 3m, 18m, 24m)
`LeafletFA_Submission2025/TabulaSenis/raw_data_processing/SS2_Snakemake`

### Human Datasets

* **Allen Brain (Lein Cortex):** `LeafletFA_Submission2025/AllenInst/lein-cortex-gru/snakemake/run_snakemake.sh`
* **Tabula Sapiens:** `LeafletFA_Submission2025/tabula_sapien/snakemake/snakemake/run_snakemake.sh`


## 2. ATSE Mapping and Filtering

After generating junction calls, we combine single-cell data to map **Alternative Transcript Structure Events (ATSE)** across data sources for each species.

### Processing Steps

1. **Batching:** Junction files are split into 100 batches for parallel processing.
2. **Junction Filtering:** Using the `JunctionReader` module from `LeafletFA-utils`.
* `min_intron`: 50 bp
* `max_intron`: 500,000 bp
* `min_cells`: 2
* `min_reads`: 10


3. **Final ATSE Mapping:** Merged junctions must meet a global threshold of **100 reads** and **10 cells**.
4. **Annotation:** Each junction is validated against a long-read based GTF, FASTA, and a GenomeDB to ensure valid splice site motifs and annotate exon-exon boundaries.

### Scripts

* **Mouse:** `LeafletFA_Submission2025/Mouse_Splicing_Foundation/ATSE_mapper`
* **Human:** `LeafletFA_Submission2025/Human_Splicing_Foundation/ATSE_mapper`


## 3. Anndata Preparation and Alignment

Mapped ATSEs and junction counts are integrated into **Anndata** objects and aligned with gene expression count objects, including scVI normalized expression values and latent space representations.

> **Note:** lncRNAs were excluded from this study.

| Species | Alignment Scripts |
| --- | --- |
| **Mouse** | `LeafletFA_Submission2025/Mouse_Splicing_Foundation/GeneExpression/01_prepare_expression_data.py` to `06_marker_gene_sanity_check.ipynb` |
| **Human** | `LeafletFA_Submission2025/Human_Splicing_Foundation/GeneExpression/01_load_raw_datasets.ipynb` to `09_marker_gene_sanity_check.ipynb` |


## 4. Mouse Model Training

The LeafletFA model is initially trained on the Mouse Splicing Foundation dataset.

* **Training Workflow:** `LeafletFA_Submission2025/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow`


## 5. Cross-Species Analysis and Transfer Learning

After establishing the mouse foundation, we compare splicing events across species and perform transfer learning.

### Conserved Junction Identification

To obtain the list of conserved splice junctions between mouse and human:

* **Script:** `LeafletFA_Submission2025/Multi_Species_Splicing_Foundation/joint_model_train/00_compare_ATSEs.ipynb`

### Human Transfer Learning

We apply the learned features from the mouse model onto the human dataset using transfer learning:

* **Script:** `LeafletFA_Submission2025/Multi_Species_Splicing_Foundation/joint_model_train/02_run_LeafletFA_transfer.ipynb`


## Data Availability

Processed data files, including ATSE mapping outputs and Anndata objects used in this analysis, are available on Zenodo:

**DOI:** [10.5281/zenodo.18158125](https://doi.org/10.5281/zenodo.18158125)

## Citation

If you use this code or these results in your research, please cite:

**Isaev, K. and Knowles, D. A. (2025). Aging-associated alternative splicing programs conserved between human and mouse tissues. bioRxiv.**

## Contact and Support

If you have any questions, please direct them to karin.isaev[@]gmail.com or submit an issue.