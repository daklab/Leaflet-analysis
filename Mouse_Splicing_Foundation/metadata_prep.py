print("="*80)
print("METADATA INTEGRATION PIPELINE")
print("Combining Tabula Muris Senis and Allen Brain Atlas metadata")
print("="*80)

import pandas as pd
import sys
from tqdm import tqdm
import numpy as np

# File paths
METADATA_PATH_TMS = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/metadata/tabula-muris-senis-full-metadata.csv"
METADATA_PATH_AB = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"

# Load Tabula Muris Senis metadata
print("\n>> Loading Tabula Muris Senis metadata...")
try:
    metadata_tms = pd.read_csv(METADATA_PATH_TMS, low_memory=False)
    print(f"   ✓ Loaded {len(metadata_tms)} entries")
except Exception as e:
    print(f"   ❌ Error loading TMS metadata: {e}")
    sys.exit(1)

# Load Allen Brain Atlas metadata
print("\n>> Loading Allen Brain Atlas metadata...")
try:
    metadata_ab = pd.read_csv(METADATA_PATH_AB, low_memory=False)
    print(f"   ✓ Loaded {len(metadata_ab)} entries")
except Exception as e:
    print(f"   ❌ Error loading Allen Brain metadata: {e}")
    sys.exit(1)

def format_cell_id(index, group):
    """Format cell ID based on age group with validation"""
    if not isinstance(index, str):
        print(f"Warning: Non-string index encountered: {index}, converting to string")
        index = str(index)
        
    if group == '3m':
        try:
            parts = index.replace('.', '-', 1).replace('_', '-', 1).split('.')
            if len(parts) < 2:
                print(f"Warning: Unexpected index format for 3m group: {index}")
                return index
            corrected_part = parts[1].replace('-', '_', 1)
            return parts[0] + '-' + corrected_part + '-1-1'
        except Exception as e:
            print(f"Error formatting cell ID for {index}: {e}")
            return index
    return index.split('.')[0]

# Process TMS metadata
print("\n>> Processing Tabula Muris Senis metadata...")
metadata_tms_subset = metadata_tms[metadata_tms['method'] == 'facs'].copy()
metadata_tms_subset['cell_id'] = metadata_tms_subset.apply(
    lambda row: format_cell_id(row['index'], row['age']), axis=1)
metadata_tms_subset = metadata_tms_subset[['cell_id', 'age', 'cell_ontology_class', 
                                 'mouse.id', 'sex', 'subtissue', 'tissue']]
print(f"   ✓ Processed {len(metadata_tms_subset)} FACS cells")

# Load additional Allen Brain metadata files
print("\n>> Loading supplementary Allen Brain metadata files...")
try:
    sample_group = pd.read_csv("/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/GSE185862_sample_group_mapping.csv.gz")
    sra_AB = pd.read_csv("/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/SraRunTable.txt")
    GSE_metadata = pd.read_csv("/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/GSE185862_metadata_ssv4.csv.gz", low_memory=False)
    sra_cell_ids = pd.read_csv("/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/sra_s3_links.txt", sep="\t", header=None)
    sra_cell_ids.columns = ['SRR', 'expected_R1_fastq']
    print(f"   ✓ Successfully loaded all supplementary files")
except Exception as e:
    print(f"   ❌ Error loading Allen Brain supplementary files: {e}")
    sys.exit(1)

# Process SRA file paths
print("\n>> Processing SRA file paths...")
# Extract the filename from the S3 path
sra_cell_ids["filename"] = sra_cell_ids["expected_R1_fastq"].str.extract(r"/([^/]+)$")[0]
# Clean suffix like `.fastq.gz.1` → `.fastq.gz`
sra_cell_ids["expected_R1_fastq"] = sra_cell_ids["filename"].str.replace(r"\.fastq.*", ".fastq.gz", regex=True)

# Check before merging
common_keys = set(GSE_metadata["expected_R1_fastq"]) & set(sra_cell_ids["expected_R1_fastq"])
print(f"   ✓ Found {len(common_keys)} common keys for merging out of {len(GSE_metadata)} GSE metadata rows")

# Merge metadata
print("\n>> Merging Allen Brain metadata sources...")
# Merge with indicator to check for drops
GSE_metadata = GSE_metadata.merge(
    sra_cell_ids, on="expected_R1_fastq", how='left', indicator=True
)
if (GSE_metadata["_merge"] != "both").any():
    print(f"   ⚠️ Warning: {(GSE_metadata['_merge'] != 'both').sum()} rows couldn't be merged")
GSE_metadata = GSE_metadata.drop(columns=["_merge"])

# Link SRA with GSE metadata
GSE_metadata["Run"] = GSE_metadata["SRR"]
sra_AB = sra_AB.merge(GSE_metadata, on="Run")
print(f"   ✓ Successfully merged Allen Brain metadata files")

# Create standardized Allen Brain metadata
print("\n>> Standardizing Allen Brain metadata...")
sra_AB["cell_id"] = sra_AB["Run"]
sra_AB["cell_ontology_class"] = sra_AB["cell_type_alias_label"]
sra_AB["tissue"] = sra_AB["sample_name"]
sra_AB["age"] = "2m" 
sra_AB["mouse.id"] = sra_AB["external_donor_name_label"] 
sra_AB["subtissue"] = sra_AB["region_label"] 
sra_AB["sex"] = sra_AB["donor_sex_label"]

# Create final Allen Brain subset
sra_AB_subset = sra_AB[["cell_id", "age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]]
print(f"   ✓ Created standardized metadata for {len(sra_AB_subset)} Allen Brain cells")

# Combine metadata
print("\n>> Combining TMS and Allen Brain metadata...")
metadata_combined = pd.concat([metadata_tms_subset, sra_AB_subset], ignore_index=True)
print(f"   ✓ Combined dataset contains {len(metadata_combined)} total cells")

# Perform data validation checks
print("\n>> Performing data validation checks...")

# Check for duplicate cell IDs in each dataset
tms_dupes = metadata_tms_subset.cell_id.duplicated().sum()
if tms_dupes > 0:
    print(f"   ⚠️ {tms_dupes} duplicate cell IDs found in TMS dataset")
else:
    print("   ✓ No duplicate cell IDs in TMS dataset")

ab_dupes = sra_AB_subset.cell_id.duplicated().sum()
if ab_dupes > 0:
    print(f"   ⚠️ {ab_dupes} duplicate cell IDs found in Allen Brain dataset")
else:
    print("   ✓ No duplicate cell IDs in Allen Brain dataset")

# Check for overlapping cell IDs between datasets
overlap = set(metadata_tms_subset.cell_id) & set(sra_AB_subset.cell_id)
if overlap:
    print(f"   ⚠️ {len(overlap)} cell IDs appear in both TMS and Allen Brain datasets")
    print(f"      Example: {list(overlap)[:5]}")
else:
    print("   ✓ No overlapping cell IDs between datasets")

# Check for duplicate cell IDs in combined dataset
dupes = metadata_combined.cell_id.duplicated().sum()
if dupes:
    print(f"   ⚠️ {dupes} duplicate cell IDs found in combined dataset")
else:
    print("   ✓ No duplicate cell IDs in combined dataset")

# Check for missing values
missing_vals = metadata_combined.isnull().sum()
if missing_vals.any():
    print("   ⚠️ Missing values detected:")
    for col in missing_vals[missing_vals > 0].index:
        print(f"      - {col}: {missing_vals[col]} missing values")
else:
    print("   ✓ No missing values detected")

# Display age distribution
print(f"   ✓ Age distribution: {dict(metadata_combined.age.value_counts())}")

# Save combined metadata
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/metadata_mouse_metadata_combined.csv"
metadata_combined.to_csv(output_path, index=False)
print(f"\n>> Successfully saved combined metadata to: {output_path}")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print(f"Combined metadata saved with {len(metadata_combined)} total cells")
print("="*80)