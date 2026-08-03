#!/usr/bin/env python
"""
Step 08-02c: Pairwise cell-type differential splicing with LeafCutter DS.

Supersedes 02b_run_leafcutter_ds_celltype_markers.py, which ran a multi-group LRT.
Two reasons for the change:

  1. LeafCutter2's own help says "currently only two groups, i.e. pairwise, supported".
     The multi-group form emits one omnibus p-value plus one-vs-baseline coefficients,
     which does not line up with LeafletFA's pairwise ΔPSI.
  2. The 2026-06-20 run was corrupted by the whitespace-parsing bug documented in
     lc_utils — group labels with spaces merged distinct cell types ("B CELL" and
     "T CELL" both became "CELL") and truncated sample names. Every groups file
     written here is read back through lc_utils.assert_groups_file_parses before any
     job is submitted.

Contrast levels
---------------
--contrast-level tissue_cell_type   (default)
    Pairs of tissue_cell_type WITHIN the same tissue. Tissue is held fixed by the
    label itself, so no tissue covariate is needed. This is the Tabula Muris Senis
    workhorse: each tissue_cell_type has one pseudobulk sample per donor, so groups
    reach 4-15 real biological replicates.

--contrast-level broad_cell_type
    Pairs of broad_cell_type, pooling tissues, with tissue as a covariate. Needed for
    Allen Brain: its pseudobulk key is tissue_cell_type x sex, which caps any single
    tissue_cell_type at 2 samples (one per sex). Pooling the 6 brain regions is the
    only way to reach a testable n — EXCITATORY vs INHIBITORY NEURON becomes 12 vs 12,
    paired across region x sex.

    Caveat for Methods: Allen replicates are region x sex strata, not animals, so the
    overdispersion LeafCutter estimates there is between-region variance rather than
    between-animal. It is anti-conservative. Tabula Muris Senis, keyed on real donors,
    should carry any calibration claim.

Contrasts are always formed WITHIN a dataset. A cross-dataset pair would be confounded
with protocol, nominal age (Allen's `age` is a hardcoded constant) and the replicate
definition itself, and none of that can be adjusted away.

Groups file format (whitespace-safe, all fields sanitised):
  sample_name  group  [covariate ...]

Output per pair (in output-dir/by_pair/<A>__vs__<B>/):
  counts_subset.tsv.gz
  groups.txt
  ds_results_cluster_significance.txt
  ds_results_effect_sizes.txt

Usage:
    # Tabula Muris Senis, within-tissue cell-type pairs
    python 02c_run_leafcutter_ds_pairwise_celltypes.py \\
        --counts-file <leafcutter_counts.tsv.gz> \\
        --metadata    <sample_metadata.csv> \\
        --output-dir  <dir> \\
        --dataset tabula_muris_senis --contrast-level tissue_cell_type

    # Allen Brain, EXC vs INH pooled across regions
    python 02c_run_leafcutter_ds_pairwise_celltypes.py \\
        ... --dataset allen_brain --contrast-level broad_cell_type

Writes every input file and validates it on --dry-run (the default); pass --submit to
actually queue the jobs.
"""

import argparse
import itertools
import os
import subprocess

import pandas as pd

from lc_utils import safe_name, assert_groups_file_parses, select_covariates

LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"

# LeafCutter's own error message: "we have only carefully checked the calibration of
# leafcutter p-values down to n=4 samples per group".
MIN_SAMPLES_PER_GROUP = 4

# Candidate covariates by contrast level. `age` is offered only where it is a real
# measurement — select_covariates drops it as constant on the Allen side anyway, but
# 01_pseudobulk_counts.py also records age_is_measured so this stays explicit.
COVARIATE_CANDIDATES = {
    "tissue_cell_type": ["sex", "age"],
    "broad_cell_type":  ["tissue", "sex", "age"],
}


def build_contrast_table(meta, level):
    """Reduce the sample table to (sample_name, group, covariates...) for this level."""
    m = meta.copy()
    m["group"] = m[level].map(safe_name)
    for col in ("tissue", "sex", "age"):
        if col in m.columns:
            m[col] = m[col].map(safe_name)
    return m


def eligible_pairs(m, level, min_samples):
    """Enumerate testable pairs, plus the ones skipped and why.

    At tissue_cell_type level, pairs are restricted to the same tissue — that is what
    makes the comparison within-tissue by construction. At broad_cell_type level,
    tissues are pooled and tissue enters as a covariate instead.
    """
    counts = m.groupby("group").size()
    pairs, skipped = [], []

    if level == "tissue_cell_type":
        blocks = [(t, sorted(sub["group"].unique())) for t, sub in m.groupby("tissue")]
    else:
        blocks = [("<pooled>", sorted(m["group"].unique()))]

    for block, groups in blocks:
        for a, b in itertools.combinations(groups, 2):
            na, nb = int(counts[a]), int(counts[b])
            if na < min_samples or nb < min_samples:
                skipped.append({"block": block, "group_1": a, "group_2": b,
                                "n_1": na, "n_2": nb,
                                "reason": f"fewer than {min_samples} samples per group"})
                continue
            pairs.append({"block": block, "group_1": a, "group_2": b, "n_1": na, "n_2": nb})
    return pairs, skipped


def write_pair_inputs(pair, m, counts, level, out_dir, min_samples):
    """Write counts_subset.tsv.gz + groups.txt for one contrast; validate before returning.

    Returns the manifest row, or raises if the groups file would not parse the way
    LeafCutter is going to read it.
    """
    a, b = pair["group_1"], pair["group_2"]
    sub = m[m["group"].isin([a, b])].copy()

    if level == "tissue_cell_type":
        # Both groups live in one tissue; keep the pair self-contained.
        sub = sub[sub["tissue"] == safe_name(pair["block"])] if pair["block"] != "<pooled>" else sub

    sample_names = [s for s in sub["sample_name"] if s in counts.columns]
    missing = len(sub) - len(sample_names)
    sub = sub[sub["sample_name"].isin(sample_names)]

    kept_cov, dropped_cov = select_covariates(
        sub, "group", COVARIATE_CANDIDATES[level]
    )

    pair_dir = os.path.join(out_dir, "by_pair", f"{a}__vs__{b}")
    os.makedirs(pair_dir, exist_ok=True)

    # Drop junctions with no reads anywhere in this subset. Safe for denominators — a
    # junction that is zero in every sample of the contrast contributes zero to every
    # cluster sum. It can leave a cluster with one surviving junction, which LeafCutter
    # then reports as "<=1 junction in cluster"; that is correct, not a defect.
    counts_sub = counts[sample_names]
    counts_sub = counts_sub[counts_sub.sum(axis=1) > 0]

    counts_path = os.path.join(pair_dir, "counts_subset.tsv.gz")
    counts_sub.to_csv(counts_path, sep="\t", compression="gzip", index_label="chrom")

    groups_path = os.path.join(pair_dir, "groups.txt")
    ordered = sub.set_index("sample_name").loc[sample_names].reset_index()
    with open(groups_path, "w") as f:
        for _, row in ordered.iterrows():
            fields = [row["sample_name"], row["group"]] + [row[c] for c in kept_cov]
            f.write("\t".join(str(x) for x in fields) + "\n")

    # The check whose absence corrupted the June run.
    assert_groups_file_parses(
        groups_path,
        counts_cols=counts_sub.columns,
        expected_labels={a, b},
        n_expected_cols=2 + len(kept_cov),
    )

    return {
        **pair,
        "n_samples": len(ordered),
        "n_donors_1": ordered.loc[ordered["group"] == a, "donor"].replace("", pd.NA).nunique(),
        "n_donors_2": ordered.loc[ordered["group"] == b, "donor"].replace("", pd.NA).nunique(),
        "n_junctions": len(counts_sub),
        "covariates_kept": ",".join(kept_cov) if kept_cov else "",
        "covariates_dropped": "; ".join(f"{k}: {v}" for k, v in dropped_cov.items()),
        "samples_not_in_counts": missing,
        "pair_dir": pair_dir,
        "status": "ready",
    }


def leafcutter_cmd(pair_dir, baseline, threads, min_samples):
    return (
        f"{LEAFCUTTER_DS} "
        f"{os.path.join(pair_dir, 'counts_subset.tsv.gz')} "
        f"{os.path.join(pair_dir, 'groups.txt')} "
        f"--output_prefix {os.path.join(pair_dir, 'ds_results')} "
        f"--baseline_group {baseline} "   # never leave this at the "Control" default
        f"--num_threads {threads} "
        f"--min_samples_per_intron {min_samples} "
        f"--min_samples_per_group {min_samples}"
    )


def submit_one(pair_dir, a, b, threads, min_samples):
    """One standalone job — used for a single --only-pair sanity run."""
    result = subprocess.run([
        "sbatch", "--mem=16G", "--partition=cpu", f"--cpus-per-task={threads}",
        f"--job-name=lcds_{a[:12]}_{b[:12]}",
        f"--output={os.path.join(pair_dir, 'leafcutter_ds_%j.out')}",
        f"--error={os.path.join(pair_dir, 'leafcutter_ds_%j.err')}",
        "--wrap", "export PYTHONUNBUFFERED=1; "
                  + leafcutter_cmd(pair_dir, a, threads, min_samples),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip()
    return result.stdout.strip().split()[-1], None


def submit_array(rows, out_dir, threads, min_samples, max_concurrent, tag):
    """One throttled SLURM array over all ready pairs.

    An array rather than N standalone jobs so `%max_concurrent` actually caps how many
    run at once — independent sbatch submissions have no such throttle.
    """
    list_path = os.path.join(out_dir, f"array_pairs_{tag}.tsv")
    with open(list_path, "w") as f:
        for r in rows:
            f.write(f"{r['pair_dir']}\t{r['group_1']}\n")

    log_dir = os.path.join(out_dir, "array_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Task i reads line i+1 of the list; keeps the array definition independent of the
    # (long, sanitised) pair names.
    wrap = (
        f'export PYTHONUNBUFFERED=1; '
        f'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {list_path}); '
        f'PAIR_DIR=$(echo "$LINE" | cut -f1); '
        f'BASELINE=$(echo "$LINE" | cut -f2); '
        f'echo "task $SLURM_ARRAY_TASK_ID -> $PAIR_DIR (baseline $BASELINE)"; '
        f'{LEAFCUTTER_DS} '
        f'"$PAIR_DIR/counts_subset.tsv.gz" "$PAIR_DIR/groups.txt" '
        f'--output_prefix "$PAIR_DIR/ds_results" '
        f'--baseline_group "$BASELINE" '
        f'--num_threads {threads} '
        f'--min_samples_per_intron {min_samples} '
        f'--min_samples_per_group {min_samples}'
    )

    result = subprocess.run([
        "sbatch", "--mem=16G", "--partition=cpu", f"--cpus-per-task={threads}",
        "--time=12:00:00",
        f"--array=0-{len(rows) - 1}%{max_concurrent}",
        f"--job-name=lcds_{tag}",
        f"--output={os.path.join(log_dir, f'{tag}_%A_%a.out')}",
        f"--error={os.path.join(log_dir, f'{tag}_%A_%a.err')}",
        "--wrap", wrap,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip()
    return result.stdout.strip().split()[-1], None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-file", required=True)
    ap.add_argument("--metadata",    required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--dataset",     required=True,
                    help="Restrict to one dataset; contrasts are never cross-dataset")
    ap.add_argument("--contrast-level", default="tissue_cell_type",
                    choices=["tissue_cell_type", "broad_cell_type"])
    ap.add_argument("--min-samples-per-group", type=int, default=MIN_SAMPLES_PER_GROUP)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--submit", action="store_true",
                    help="Actually queue the jobs (default is dry-run: write and validate only)")
    ap.add_argument("--only-pair", default=None,
                    help="Restrict to one contrast, given as '<group_1>__vs__<group_2>'. "
                         "Use for a single sanity run before launching the rest.")
    ap.add_argument("--max-concurrent", type=int, default=20,
                    help="Cap on simultaneously running array tasks (default: 20)")
    ap.add_argument("--redo-existing", action="store_true",
                    help="Re-run contrasts that already have results (default: skip them)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1] Loading sample metadata ...", flush=True)
    meta = pd.read_csv(args.metadata)
    meta = meta[meta["dataset"] == args.dataset].copy()
    if "donor" not in meta.columns:
        meta["donor"] = ""
    meta["donor"] = meta["donor"].fillna("")
    print(f"  {len(meta):,} pseudobulk samples in {args.dataset}", flush=True)

    m = build_contrast_table(meta, args.contrast_level)
    print(f"  contrast level: {args.contrast_level} "
          f"({m['group'].nunique()} distinct groups)", flush=True)

    print(f"\n[2] Enumerating pairs (>= {args.min_samples_per_group} samples per side) ...",
          flush=True)
    pairs, skipped = eligible_pairs(m, args.contrast_level, args.min_samples_per_group)
    print(f"  eligible: {len(pairs)}   skipped: {len(skipped)}", flush=True)

    if args.only_pair:
        want = args.only_pair
        pairs = [p for p in pairs if f"{p['group_1']}__vs__{p['group_2']}" == want]
        if not pairs:
            raise SystemExit(f"--only-pair {want!r} is not an eligible contrast. "
                             f"Run without --only-pair to list them.")
        print(f"  restricted to 1 contrast: {want}", flush=True)
        skipped = []          # a filtered run should not rewrite the full manifest
    if args.contrast_level == "tissue_cell_type":
        by_tissue = pd.Series([p["block"] for p in pairs]).value_counts()
        print(f"  tissues contributing pairs: {len(by_tissue)}", flush=True)
        print("   ", by_tissue.to_dict(), flush=True)

    if not pairs:
        print("\nNo eligible pairs — nothing to do.", flush=True)
        return

    print(f"\n[3] Loading counts matrix ...", flush=True)
    counts = pd.read_csv(args.counts_file, sep="\t", index_col=0, compression="gzip")
    print(f"  {counts.shape[0]:,} junctions x {counts.shape[1]} samples", flush=True)

    print(f"\n[4] Writing and validating per-pair inputs ...", flush=True)
    rows, failures = [], []
    for i, pair in enumerate(pairs, 1):
        try:
            rows.append(write_pair_inputs(pair, m, counts, args.contrast_level,
                                          args.output_dir, args.min_samples_per_group))
        except AssertionError as e:
            failures.append({**pair, "status": "INVALID", "error": str(e)})
            print(f"  [{i}/{len(pairs)}] FAILED {pair['group_1']} vs {pair['group_2']}",
                  flush=True)
            print(f"      {e}", flush=True)
            continue
        if i % 10 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] validated", flush=True)

    # Contrasts already finished (e.g. by an --only-pair sanity run, or a partial array)
    # keep their manifest row but drop out of the submission list.
    done = []
    if not args.redo_existing:
        still = []
        for r in rows:
            if os.path.exists(os.path.join(r["pair_dir"],
                                           "ds_results_cluster_significance.txt")):
                r["status"] = "already_done"
                done.append(r)
            else:
                still.append(r)
        rows = still

    print(f"\n  ready: {len(rows)}   already done: {len(done)}   invalid: {len(failures)}",
          flush=True)
    if failures:
        print("  Refusing to submit while any groups file is invalid.", flush=True)

    manifest = pd.concat([
        pd.DataFrame(rows),
        pd.DataFrame(done),
        pd.DataFrame(failures),
        pd.DataFrame([{**s, "status": "skipped"} for s in skipped]),
    ], ignore_index=True)
    manifest["dataset"] = args.dataset
    manifest["contrast_level"] = args.contrast_level
    tag = f"{args.dataset}_{args.contrast_level}"
    suffix = "_single" if args.only_pair else ""
    man_path = os.path.join(args.output_dir, f"pairwise_manifest_{tag}{suffix}.tsv")
    manifest.to_csv(man_path, sep="\t", index=False)
    print(f"  manifest → {man_path}", flush=True)

    if not rows:
        print("\nNothing left to run — every eligible contrast already has results.",
              flush=True)
        return

    if not args.submit:
        print(f"\n[5] DRY RUN — {len(rows)} jobs would be submitted. "
              f"Re-run with --submit to queue them.", flush=True)
        for r in rows[:3]:
            print(f"    {r['group_1']} vs {r['group_2']}  "
                  f"n={r['n_1']}/{r['n_2']}  cov=[{r['covariates_kept']}]  "
                  f"{r['n_junctions']:,} junctions", flush=True)
        if len(rows) > 3:
            print(f"    ... and {len(rows) - 3} more (see manifest)", flush=True)
        return

    if failures:
        print("\nAborting: fix the invalid groups files first.", flush=True)
        return

    if len(rows) == 1:
        r = rows[0]
        print(f"\n[5] Submitting 1 job: {r['group_1']} vs {r['group_2']} ...", flush=True)
        job_id, err = submit_one(r["pair_dir"], r["group_1"], r["group_2"],
                                 args.num_threads, args.min_samples_per_group)
        if job_id:
            print(f"  job {job_id}  ->  {r['pair_dir']}", flush=True)
        else:
            print(f"  ERROR: {err}", flush=True)
        return

    print(f"\n[5] Submitting array of {len(rows)} jobs "
          f"(max {args.max_concurrent} concurrent) ...", flush=True)
    job_id, err = submit_array(rows, args.output_dir, args.num_threads,
                               args.min_samples_per_group, args.max_concurrent, tag)
    if job_id:
        print(f"\n=== array job {job_id}: {len(rows)} tasks, "
              f"%{args.max_concurrent} throttle ===", flush=True)
        print(f"Results in: {os.path.join(args.output_dir, 'by_pair')}/", flush=True)
        print(f"Watch:  squeue -u $USER -j {job_id}", flush=True)
    else:
        print(f"  ERROR: {err}", flush=True)


if __name__ == "__main__":
    main()
