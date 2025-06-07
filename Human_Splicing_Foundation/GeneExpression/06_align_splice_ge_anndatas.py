# %% [markdown]
# ### Take in the raw splice_adata produced by ATSEmapper, clean up its metadata and read in corresponding GE based anndata file 
# 
# - ensure both anndatas contain the same cells and metadata for them in the SAME order 

# %%
import os
import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import anndata as ad
from scipy.sparse import coo_matrix, csr_matrix
from datetime import datetime
import anndata as ad
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.spatial.distance import cdist
import numpy as np
import sys
import os
import json
import numpy as np
import torch
import anndata as ad
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pyro 
import umap.umap_ as umap
import matplotlib.patches as mpatches
import scipy.sparse
import datetime
import sys
import random

# Output DIR 
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025"
print(f"Output directory: {output_dir}", flush=True)

# --- Splice data ---
input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250421/anndatas/merged_anndata.h5ad"
assert os.path.exists(input_file), f"Splice input file does not exist: {input_file}"

splice_adata = ad.read_h5ad(input_file)
assert splice_adata.shape[0] > 0, "Splice AnnData has zero cells"
assert splice_adata.shape[1] > 0, "Splice AnnData has zero features"

splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
print(f"The number of cells in the splice dataset is {splice_adata.shape[0]}", flush=True)

# --- ATSEs ---
ATSE_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-05-11_06-23-05.txt.gz"
assert os.path.exists(ATSE_file), f"ATSE file does not exist: {ATSE_file}"

atses = pd.read_csv(ATSE_file, sep="\t")
assert "event_id" in atses.columns, "'event_id' column missing from ATSE file"
assert len(atses) > 0, "ATSE file is empty"

print(f"The number of ATSEs in this dataset is {len(atses['event_id'].unique())}", flush=True)

# --- Gene expression data ---
ge_input = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/ts_ab_exons_combo_ge_adata_2025-06-06.h5ad"
assert os.path.exists(ge_input), f"Gene expression file does not exist: {ge_input}"

ge_adata = ad.read_h5ad(ge_input)
assert ge_adata.shape[0] > 0, "Gene expression AnnData has zero cells"
assert ge_adata.shape[1] > 0, "Gene expression AnnData has zero features"

ge_adata.obs.reset_index(drop=True, inplace=True)
ge_adata.obs["cell_id_index"] = ge_adata.obs.index 
print(f"The number of cells in the gene expression dataset is {ge_adata.shape[0]}", flush=True)

# %%
# Additional processing that needs to be done on the splicing data 
metadata = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv"
metadata = pd.read_csv(metadata, sep="\t")

#----------------------------------------
# Clean up cell_ids in both datasets 
#----------------------------------------

# Gene expression 
# 1. Only Tabula Sapiens cells
ts_like_cells = ge_adata.obs["cell_id_backup"].str.startswith("TSP")

# 2. Split on "_"
ge_parts = ge_adata.obs.loc[ts_like_cells, "cell_id_backup"].str.split("_")

# 3. Parse all fields
sample_id = ge_parts.str[0]
platform = ge_parts.str[1]
donor1 = ge_parts.str[2]        # L144742
plate1 = ge_parts.str[3]        # B1
donor2 = ge_parts.str[4]        # B133819
plate2 = ge_parts.str[5]        # D2
tissue = ge_parts.str[6]        # Liver
subtissue = ge_parts.str[7]     # NA
celltype = ge_parts.str[8]      # Hepatocyte

# 4. Build full cleaned ID
ge_adata.obs.loc[ts_like_cells, "cell_id_clean"] = (
    sample_id + "_donor_" + donor1 + "_" + plate1 + "_" + donor2 + "_" + plate2 + "_" + tissue + "_" + celltype
)

# 5. Fallback for non-TS cells
ge_adata.obs.loc[~ts_like_cells, "cell_id_clean"] = ge_adata.obs.loc[~ts_like_cells, "cell_id_backup"]

# Splicing 
# 1. Only Tabula Sapiens cells
ts_cells = splice_adata.obs["dataset"] == "tabula_sapiens"

# 2. Split after "per_"
per_parts = splice_adata.obs.loc[ts_cells, "cell_id"].str.split("per_").str[-1]
split_parts = per_parts.str.split("_")

# 3. Parse all fields
sample_id = split_parts.str[0]
platform = split_parts.str[1]
donor1 = split_parts.str[2]     # L144736
plate1 = split_parts.str[3]     # G2
donor2 = split_parts.str[4]     # B133819
plate2 = split_parts.str[5]     # M4
tissue = split_parts.str[6]     # Liver
subtissue = split_parts.str[7]  # NA
celltype = split_parts.str[8]   # Hepatocyte

# 4. Build full cleaned ID
splice_adata.obs.loc[ts_cells, "cell_id_clean"] = (
    sample_id + "_donor_" + donor1 + "_" + plate1 + "_" + donor2 + "_" + plate2 + "_" + tissue + "_" + celltype
)

# 5. Fallback for non-TS cells
splice_adata.obs.loc[~ts_cells, "cell_id_clean"] = splice_adata.obs.loc[~ts_cells, "cell_id"]

# Common cells 
common_cells = np.intersect1d(splice_adata.obs["cell_id_clean"], ge_adata.obs["cell_id_clean"])
print(len(common_cells))  # Should be the number of matched cells

# 1. Set obs_names to be the cleaned cell IDs
splice_adata.obs_names = splice_adata.obs["cell_id_clean"]
ge_adata.obs_names = ge_adata.obs["cell_id_clean"]

splice_adata = splice_adata[splice_adata.obs_names.isin(common_cells)].copy()
ge_adata = ge_adata[ge_adata.obs_names.isin(common_cells)].copy()
splice_adata = splice_adata[common_cells].copy()
ge_adata = ge_adata[common_cells].copy()
assert all(splice_adata.obs_names == ge_adata.obs_names), "Mismatch: cell order is wrong!"
print("Cells are now perfectly aligned.")

splice_adata.obs_names = pd.Index(map(str, range(splice_adata.n_obs)))
ge_adata.obs_names = pd.Index(map(str, range(ge_adata.n_obs)))

# Save in output_dir
import datetime
today = datetime.datetime.now().strftime("%Y-%m-%d")

# Save splice_adata
splice_adata_file = os.path.join(output_dir, f"splice_adata_matched_{today}.h5ad")
splice_adata.write_h5ad(splice_adata_file, compression="lzf")
print(f"Saved splice_adata to {splice_adata_file}")

# Save ge_adata
ge_adata_file = os.path.join(output_dir, f"ge_adata_matched_{today}.h5ad")
ge_adata.write_h5ad(ge_adata_file, compression="lzf")
print(f"Saved ge_adata to {ge_adata_file}")

# Submit script like this:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=200G --partition=cpu,dev,bigmem --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/06_align_splice_ge_anndatas.py"
