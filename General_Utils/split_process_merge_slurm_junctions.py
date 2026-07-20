import os
import math
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

from atsemapper.atsemapper.junction_parser import JunctionReader


def split_file_list(input_file: str, chunks: int, output_dir: str):
    with open(input_file) as f:
        files = [line.strip() for line in f if line.strip()]

    chunk_size = math.ceil(len(files) / chunks)
    os.makedirs(output_dir, exist_ok=True)
    actual_chunks = math.ceil(len(files) / chunk_size)
    print(f"Will create {actual_chunks} chunks from {len(files)} files")

    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        chunk_id = i // chunk_size
        chunk_filename = f"{output_dir}/chunk_{chunk_id}.txt"
        print(f"  chunk {chunk_id}: {len(chunk)} files → {chunk_filename}")
        with open(chunk_filename, "w") as f:
            f.write("\n".join(chunk))

    return len(files), actual_chunks


def process_chunk(chunk_file: str, output_dir: str,
                  min_intron: int = 50, max_intron: int = 500000,
                  min_reads: int = 100, min_cells: int = 10,
                  batch_size: int = 10, num_workers: int = 4,
                  sequencing_type: str = "single_cell"):
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
    os.makedirs(output_dir, exist_ok=True)

    with open(chunk_file) as f:
        file_list = [line.strip() for line in f if line.strip()]
    print(f"Processing {len(file_list)} files from {chunk_file}")

    reader = JunctionReader(
        min_intron=min_intron,
        max_intron=max_intron,
        min_reads=min_reads,
        min_cells=min_cells,
        batch_size=batch_size,
        num_workers=num_workers,
        sequencing_type=sequencing_type,
    )

    junctions = reader.process_files(file_list)

    chunk_id = Path(chunk_file).stem
    output_file = f"{output_dir}/{chunk_id}_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(junctions, f)
    print(f"Saved {len(junctions)} junctions → {output_file}")


def merge_results(results_dir: str, output_file: str,
                  min_reads: int = 0, min_cells: int = 0):
    combined = {}
    results_files = sorted(Path(results_dir).glob("*_results.pkl"))
    print(f"Merging {len(results_files)} chunk result files...")

    for rf in results_files:
        with open(rf, "rb") as f:
            chunk = pickle.load(f)
        for j_id, j_data in chunk.items():
            if j_id not in combined:
                combined[j_id] = j_data.copy()
            else:
                combined[j_id]["cells"] += j_data["cells"]
                combined[j_id]["total_score"] += j_data["total_score"]

    before = len(combined)
    if min_reads > 0 or min_cells > 0:
        combined = {
            j_id: j_data for j_id, j_data in combined.items()
            if j_data["cells"] >= min_cells and j_data["total_score"] >= min_reads
        }
        print(f"After QC (min_cells={min_cells}, min_reads={min_reads}): "
              f"{before:,} → {len(combined):,} junctions")

    with open(output_file, "wb") as f:
        pickle.dump(combined, f)
    return len(combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "process", "merge"], required=True)
    parser.add_argument("--input-file")
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--chunk-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-output")
    # JunctionReader params (passed by 02_process.sh from config)
    parser.add_argument("--min-intron",  type=int, default=50)
    parser.add_argument("--max-intron",  type=int, default=500000)
    parser.add_argument("--min-reads",   type=int, default=100)
    parser.add_argument("--min-cells",   type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sequencing-type", default="single_cell")
    args = parser.parse_args()

    if args.mode == "split":
        if not args.input_file or not args.chunks:
            parser.error("--input-file and --chunks required for split mode")
        total, chunks = split_file_list(args.input_file, args.chunks, args.output_dir)
        print(f"Split {total} files into {chunks} chunks")

    elif args.mode == "process":
        if not args.chunk_file:
            parser.error("--chunk-file required for process mode")
        process_chunk(
            args.chunk_file, args.output_dir,
            min_intron=args.min_intron, max_intron=args.max_intron,
            min_reads=args.min_reads, min_cells=args.min_cells,
            batch_size=args.batch_size, num_workers=args.num_workers,
            sequencing_type=args.sequencing_type,
        )

    elif args.mode == "merge":
        if not args.merge_output:
            parser.error("--merge-output required for merge mode")
        n = merge_results(args.output_dir, args.merge_output,
                          min_reads=args.min_reads, min_cells=args.min_cells)
        print(f"Merged {n} unique junctions → {args.merge_output}")


if __name__ == "__main__":
    main()
