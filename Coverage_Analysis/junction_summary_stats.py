import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import pearsonr
from datetime import datetime
import glob

def read_single_junction_file_filt(junc_file, sequencing_type="smart_seq"):
    """
    Read a single junction file and return a dictionary of junctions and their cell-level counts.
    """
    dtypes = {
        0: str, 1: 'int32', 2: 'int32', 3: str, 4: 'int32', 5: str,
        6: 'int32', 7: 'int32', 8: str, 9: 'int32', 10: str, 11: str
    }

    try:
        juncs = pd.read_csv(junc_file, sep="\t", header=None, dtype=dtypes)
    except pd.errors.EmptyDataError:
        print(f"Empty or invalid file encountered: {junc_file}. Skipping.")
        return defaultdict(lambda: defaultdict(int))
    except Exception as e:
        print(f"Unexpected error reading {junc_file}: {e}. Skipping.")
        return defaultdict(lambda: defaultdict(int))

    col_names = [
        "chrom", "chromStart", "chromEnd", "name", "score", "strand",
        "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"
    ]
    if sequencing_type in ["smart_seq", "10x", "split-seq", "easysci"]:
        col_names += ["num_cells_wjunc", "cell_readcounts"]

    try:
        juncs.columns = col_names
    except ValueError:
        print(f"File {junc_file} has an unexpected number of columns. Skipping.")
        return defaultdict(lambda: defaultdict(int))

    try:
        juncs[['block_add_start', 'block_subtract_end']] = juncs["blockSizes"].str.extract(r'(\d+),(\d+)').astype(int)
        juncs["chromStart"] += juncs['block_add_start']
        juncs["chromEnd"] -= juncs['block_subtract_end']
        juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str) + '_' + juncs['strand']
    except KeyError as e:
        print(f"Missing expected column in file {junc_file}: {e}. Skipping.")
        return defaultdict(lambda: defaultdict(int))

    cell_junction_counts = defaultdict(lambda: defaultdict(int))

    for _, row in juncs.iterrows():
        junction_id = row['junction_id']

        if sequencing_type == "smart_seq":
            try:
                cell_id = row['cell_readcounts'].split(":")[0]
                read_count = row['score']
                cell_junction_counts[cell_id][junction_id] += read_count
            except Exception as e:
                print(f"Error processing row in file {junc_file}: {e}. Skipping row.")
        elif sequencing_type == "10x":
            try:
                for cell_info in row['cell_readcounts'].split(','):
                    cell_id, read_count = cell_info.split(":")
                    read_count = int(read_count)
                    cell_junction_counts[cell_id][junction_id] += read_count
            except Exception as e:
                print(f"Error processing row in file {junc_file}: {e}. Skipping row.")

    return cell_junction_counts

def visualize_distributions(cell_stats_df, output_dir):
    """
    Generate and save plots to visualize distributions of read counts and junction frequencies.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(cell_stats_df["junction_count"], bins=50, log=True, edgecolor="black")
    plt.title("Distribution of Number of Junctions Observed Per Cell")
    plt.xlabel("Number of Junctions")
    plt.ylabel("Frequency (log scale)")
    plt.savefig(os.path.join(output_dir, "junctions_per_cell_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(cell_stats_df["coverage"], bins=50, log=True, edgecolor="black")
    plt.title("Distribution of Cell Coverage (Total Reads)")
    plt.xlabel("Total Coverage (Read Counts)")
    plt.ylabel("Frequency (log scale)")
    plt.savefig(os.path.join(output_dir, "coverage_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cell_stats_df, x="coverage", y="junction_count", alpha=0.5)
    plt.title("Cell Coverage vs. Number of Junctions")
    plt.xlabel("Total Coverage (Read Counts)")
    plt.ylabel("Number of Junctions")
    plt.savefig(os.path.join(output_dir, "coverage_vs_junctions.png"))
    plt.close()

    correlation, p_value = pearsonr(cell_stats_df["coverage"], cell_stats_df["junction_count"])
    print(f"Correlation between coverage and junctions: {correlation:.2f} (p={p_value:.2e})")

def summarize_junction_data(file_list, sequencing_type, dataset_name, output_dir):

    """
    Summarize distributions of read counts, junction frequencies, and junctions per cell.
    """
    today = datetime.today().strftime('%Y-%m-%d')
    full_output_dir = os.path.join(output_dir, f"{dataset_name}_{today}")
    os.makedirs(full_output_dir, exist_ok=True)

    cell_junction_counts = defaultdict(lambda: defaultdict(int))

    print("Processing junction files...")
    for junc_file in tqdm(file_list):
        file_counts = read_single_junction_file_filt(junc_file, sequencing_type)
        for cell_id, counts in file_counts.items():
            for junction, count in counts.items():
                cell_junction_counts[cell_id][junction] += count

    # Cell-level statistics
    cell_junction_summaries = {
        "junction_count": [len(junctions) for junctions in cell_junction_counts.values()],
        "coverage": [sum(junctions.values()) for junctions in cell_junction_counts.values()]
    }
    cell_stats_df = pd.DataFrame(cell_junction_summaries)

    cell_stats_file = os.path.join(full_output_dir, "cell_stats.csv")
    cell_stats_df.to_csv(cell_stats_file, index=False)
    print(f"Cell-level statistics saved to {cell_stats_file}")

    # Junction-level statistics
    junction_counts = defaultdict(lambda: {"total_read_count": 0, "cell_count": 0})
    for cell_id, junctions in cell_junction_counts.items():
        for junction, count in junctions.items():
            junction_counts[junction]["total_read_count"] += count
            junction_counts[junction]["cell_count"] += 1

    junction_stats_df = pd.DataFrame.from_dict(junction_counts, orient="index")
    junction_stats_df.index.name = "junction_id"
    junction_stats_df.reset_index(inplace=True)

    junction_stats_file = os.path.join(full_output_dir, "junction_stats.csv")
    junction_stats_df.to_csv(junction_stats_file, index=False)
    print(f"Junction-level statistics saved to {junction_stats_file}")
    visualize_distributions(cell_stats_df, full_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize junction data across a dataset.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset being analyzed.")
    parser.add_argument("sequencing_type", type=str, choices=["smart_seq", "10x", "split-seq", "easysci"], help="Type of sequencing data.")
    parser.add_argument("--dataset_dir", type=str, help="Directory containing junction files.")
    parser.add_argument("--file_list", type=str, help="Path to a file containing a list of junction file paths (one per line).")
    parser.add_argument("--output_dir", type=str, default="./results", help="Root directory for saving outputs.")
    parser.add_argument("--suffix_junction_files", type=str, help="Suffix of junction files to be analyzed.", default="junctions_with_barcodes.bed")
    parser.add_argument("--glob_pattern", type=str, help="Glob pattern for retrieving junction files.", default=None)

    args = parser.parse_args()

    # Gather input parameters
    dataset_name = args.dataset_name
    sequencing_type = args.sequencing_type
    output_dir = args.output_dir

    # Get the list of files based on the input mode
    if args.file_list:
        # File list mode: Read paths from the provided file
        with open(args.file_list, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]
    elif args.glob_pattern:
        # Glob pattern mode: Use the provided glob pattern to retrieve files
        file_list = glob.glob(args.glob_pattern, recursive=True)
    elif args.dataset_dir:
        # Directory mode: Find files in the dataset directory with the given suffix
        file_list = [
            os.path.join(args.dataset_dir, f)
            for f in os.listdir(args.dataset_dir)
            if f.endswith(args.suffix_junction_files)
        ]
    else:
        raise ValueError("You must provide either --dataset_dir, --file_list, or --glob_pattern.")

    print(f"Number of files to process: {len(file_list)}")
    print(f"Processing junction files for {dataset_name} ({sequencing_type})...")

    # Summarize junction data
    summarize_junction_data(file_list, sequencing_type, dataset_name, output_dir)

# how to run? example... tabula sapien 
#script=/gpfs/commons/home/kisaev/Leaflet-analysis/junction_summary_stats.py
#dataset_name=TabulaSapien
#output_dir=/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/ATSEs
#python $script $dataset_name smart_seq --glob_pattern "/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/**/*.juncswbarcodes" --output_dir $output_dir
    

# how to run? example... allen brain 
#script=/gpfs/commons/home/kisaev/Leaflet-analysis/junction_summary_stats.py
#dataset_name=AllenBrainHuman
#output_dir=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/LeafletAnalysis/JAN2025/ATSEs
#python $script $dataset_name smart_seq --glob_pattern "/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet/*junctions_with_barcodes.bed" --output_dir $output_dir
    
#### sbatch --mem=200G --time=24:00:00 --job-name=AB_junc_stats --wrap="python $script $dataset_name smart_seq --glob_pattern '/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet/*junctions_with_barcodes.bed' --output_dir $output_dir"

# how to run? example... easysci RH 
#script=/gpfs/commons/home/kisaev/Leaflet-analysis/junction_summary_stats.py
#dataset_name=EasySciRH
#output_dir=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/
#### sbatch --mem=200G -p bigmem --time=24:00:00 --job-name=AB_junc_stats --wrap="python $script $dataset_name smart_seq --glob_pattern '/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/RH/*junctions_with_barcodes.bed' --output_dir $output_dir"
