#!/bin/bash
#
#SBATCH -N 1
#SBATCH -J EASYSCI_master
#SBATCH -c 1
#SBATCH -p cpu,bigmem,dev
#SBATCH --mem=16G
#SBATCH -t 5-00:00
#SBATCH --output=EASYSCI_master_%j.log

conda activate python3ENV

module purge
module unload htslib/1.9
module load star/2.7.10b-GCC-11.3.0
module load samtools

# Set METACELL_SUFFIX before submitting (e.g. _500, _100, _1000).
SUFFIX="${METACELL_SUFFIX:-}"
BASE_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: SpliceVI dir not found: $BASE_DIR" >&2
    echo "Set METACELL_SUFFIX (e.g. _500) before submitting." >&2
    exit 1
fi

# BAMs may be spread across multiple date subdirs (e.g. 20260228 + 20260301).
# Build combined_RH/ with real subdirs and file-level symlinks — os.walk won't
# follow symlinked directories, so cluster dirs must be real.
COMBINED_RH="$BASE_DIR/combined_RH"
mkdir -p "$COMBINED_RH"
for rh_dir in "$BASE_DIR"/*/RH/*/; do
    [[ -d "$rh_dir" ]] || continue
    cluster=$(basename "$rh_dir")
    bam="$rh_dir/${cluster}.bam"
    bai="$rh_dir/${cluster}.bam.bai"
    [[ -f "$bam" ]] || continue
    cluster_dir="$COMBINED_RH/$cluster"
    mkdir -p "$cluster_dir"
    [[ ! -e "$cluster_dir/${cluster}.bam" ]]     && ln -s "$bam" "$cluster_dir/${cluster}.bam"
    [[ -f "$bai" && ! -e "$cluster_dir/${cluster}.bam.bai" ]] && ln -s "$bai" "$cluster_dir/${cluster}.bam.bai"
done
N_CLUSTERS=$(ls "$COMBINED_RH" | wc -l)
echo "Combined RH dir: $COMBINED_RH ($N_CLUSTERS clusters)"

RESULTS_DIR="$BASE_DIR/junctions"
mkdir -p "$RESULTS_DIR"

DATE=$(date +%Y%m%d)
slurm_out="$BASE_DIR/slurm/$DATE"
mkdir -p "$slurm_out"

cd /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH

snakemake --keep-going -j 16 \
    --config bam_files="$COMBINED_RH" results_dir="$RESULTS_DIR" \
    --cluster-config cluster.json \
    --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out/slurm-%j.out --error=$slurm_out/slurm-%j.err" \
    --latency-wait 120 \
    --rerun-incomplete #--unlock

echo "Snakemake workflow submitted"

# Usage:
#   METACELL_SUFFIX=_500 sbatch run_snakemake.sh
#   METACELL_SUFFIX=_100 sbatch run_snakemake.sh
