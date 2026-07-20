#!/usr/bin/env python
"""
Step 12a: Prepare one EasySci metacell size for LeafletFA model application.

Run separately for each metacell size (500, 1000, 2000, 4000).
SpliceVI_100 requires running ATSEmapper first before this script can be used.

What this script does:
  SPLICE:
    1. Load EasySci splice merged_anndata (mm39 junction coords)
    2. Map mm39 junction IDs → mm10 model junction IDs via liftover
    3. Build aligned cell_by_junction_matrix (N × 83,593) — zero-fill missing junctions
    4. RECOMPUTE cell_by_cluster_matrix from scratch using mm10 ATSE groupings
       (EasySci mm39 cluster values are DISCARDED — they use different ATSE groupings)
    5. Spot-check invariant: all junctions in observed ATSE have same ATSE total
    6. Compute junc_ratio using cluster sparsity (PSI=0 for zero-read in observed ATSE)

  GENE EXPRESSION:
    7. Load EasySci GE metacell anndata (metacell_RT_anndata_{size}.h5ad)
    8. Align to filtered mouse GE gene space (11,673 genes) by Ensembl gene_id
    9. Zero-fill genes absent from EasySci

  OUTPUT:
    - easysci_splice_model_ready_{size}_YYYY-MM-DD.h5ad
    - easysci_ge_model_ready_{size}_YYYY-MM-DD.h5ad
    Both aligned to same obs (intersection of splice ∩ GE metacells).

Usage:
    python 01_prep_easysci_for_model.py \\
        --easysci-splice-h5ad <SpliceVI_{size}/ATSEmap/.../merged_anndata.h5ad> \\
        --easysci-ge-h5ad    <metacell_RT_anndata_{size}.h5ad> \\
        --liftover-atse      <MOUSE_FOUNDATION_ATSE_FILE_..._lifted_mm39.txt.gz> \\
        --ref-splice-h5ad    <splice_adata_aligned_filtered_..._2026-06-13.h5ad> \\
        --ref-ge-h5ad        <ge_adata_aligned_filtered_..._2026-06-13.h5ad> \\
        --output-dir         <output_directory> \\
        --size               <500|1000|2000|4000>

SLURM:
    sbatch --mem=32G -p cpu \\
        --wrap "python pipelines/06_easysci/01_prep_easysci_for_model.py ..."
    Reference h5ads are read via h5py (var only) — no full AnnData load, minimal memory overhead.
"""

import argparse
import os
from datetime import datetime

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_h5ad_var(path):
    """Read only the var DataFrame from an h5ad file — avoids loading obs or data matrices."""
    with h5py.File(path, "r") as f:
        grp = f["var"]
        idx = grp["_index"][:].astype(str) if "_index" in grp else None
        cols = {}
        for key in grp.keys():
            if key == "_index":
                continue
            item = grp[key]
            if isinstance(item, h5py.Dataset):
                vals = item[:]
                cols[key] = vals.astype(str) if vals.dtype.kind in ("S", "O") else vals
            elif isinstance(item, h5py.Group) and "categories" in item:
                cats = item["categories"][:].astype(str)
                codes = item["codes"][:]
                cols[key] = pd.Categorical.from_codes(codes, categories=cats)
        df = pd.DataFrame(cols, index=idx)
        df.index.name = None
        return df


# ── cluster matrix recomputation ─────────────────────────────────────────────

def recompute_cluster_matrix(jc_csr, model_event_ids):
    """
    Recompute cell_by_cluster_matrix in mm10 ATSE space from mapped junction counts.

    Semantics (same as 09b_filter_aligned.py):
      - If any junction in mm10 ATSE A has reads in cell i → ALL junctions in A
        (including zero-read ones) get atse_total[i, A] stored in the cluster matrix.
      - Structural zeros (ATSE never observed) stay absent (not explicitly stored).
      - PSI = junction[i,j] / cluster[i,j]  →  0/total = 0 for zero-read junctions.
    """
    n_cells, J = jc_csr.shape
    unique_events, atse_idx = np.unique(model_event_ids, return_inverse=True)
    A = len(unique_events)

    # J × A indicator: which junctions belong to which ATSE
    M = sp.csr_matrix(
        (np.ones(J, dtype=np.float32), (np.arange(J), atse_idx)),
        shape=(J, A)
    )
    # atse_sums[i, a] = total reads across all mm10 junctions in ATSE a for cell i
    atse_sums = (jc_csr.astype(np.float32) @ M).tocsr()

    # Broadcast back: cluster[i, j] = atse_sums[i, atse_of_j] for all stored positions
    # Process in cell chunks to control memory
    CHUNK = 1000
    n_batches = (n_cells + CHUNK - 1) // CHUNK
    chunks = []
    for b, start in enumerate(range(0, n_cells, CHUNK)):
        end = min(start + CHUNK, n_cells)
        batch = atse_sums[start:end, :].tocsc()[:, atse_idx].tocsr().astype(np.int64)
        chunks.append(batch)
        if (b + 1) % 5 == 0 or b == 0 or b == n_batches - 1:
            print(f"  cluster recompute batch {b+1}/{n_batches}", flush=True)

    at_csr = sp.vstack(chunks, format="csr")
    return at_csr, unique_events, atse_idx


def spot_check_cluster(jc_csr, at_csr, model_event_ids, ref_var,
                       mm10_col_to_mm39, n_atse_check=6):
    """
    Verify ATSE cluster invariant and print human-readable diagnostics.

    For each sampled ATSE:
      - Shows gene name + number of junctions in ss2 reference
      - Shows mm10 junction IDs (model space) + original mm39 EasySci IDs
      - Samples cells that ACTUALLY have reads in this ATSE (not random zero cells)
      - Checks all junctions in observed ATSE have same ATSE total
    """
    event_to_juncs = {}
    for j, ev in enumerate(model_event_ids):
        event_to_juncs.setdefault(ev, []).append(j)

    # Build mm10 junction_id lookup from ref_var index
    mm10_junc_ids = ref_var["junction_id"].values if "junction_id" in ref_var.columns \
                    else ref_var.index.values

    # Sample ATSEs with varying junction counts
    jpa = {ev: len(js) for ev, js in event_to_juncs.items()}
    jpa_counts = np.array(list(jpa.values()))
    jpa_events = np.array(list(jpa.keys()))

    np.random.seed(42)
    sample_atses = []
    for target in [2, 3, 5, int(jpa_counts.max())]:
        candidates = jpa_events[jpa_counts == target]
        if len(candidates):
            sample_atses.append(np.random.choice(candidates))
    # Add 2 random multi-junction ATSEs
    multi = jpa_events[jpa_counts >= 2]
    for _ in range(2):
        sample_atses.append(np.random.choice(multi))
    sample_atses = list(dict.fromkeys(sample_atses))[:n_atse_check]

    errors = 0

    for atse in sample_atses:
        juncs = event_to_juncs[atse]  # mm10 column indices for this ATSE

        # Gene name from ref_var
        gene_name = ref_var.loc[ref_var["event_id"] == atse, "gene_name"].iloc[0] \
                    if "gene_name" in ref_var.columns else "?"

        # mm10 junction IDs for these columns
        mm10_ids_here = [mm10_junc_ids[j] for j in juncs]

        # Original mm39 junction IDs (or "zero-filled" if no EasySci coverage)
        mm39_ids_here = [mm10_col_to_mm39.get(j, "zero-filled") for j in juncs]

        n_covered = sum(1 for x in mm39_ids_here if x != "zero-filled")
        print(f"\n  ATSE: {atse}  gene={gene_name}  "
              f"({len(juncs)} junctions in ss2, {n_covered} with EasySci coverage)",
              flush=True)
        for jj, (mm10id, mm39id) in enumerate(zip(mm10_ids_here, mm39_ids_here)):
            cov_flag = "covered" if mm39id != "zero-filled" else "zero-filled"
            print(f"    j{jj}: mm10={mm10id}  mm39={mm39id}  [{cov_flag}]", flush=True)

        # Sample cells that actually have reads in this ATSE (ATSE total > 0)
        at_sub_all = np.asarray(at_csr[:, juncs].sum(axis=1)).ravel()
        nonzero_cells = np.where(at_sub_all > 0)[0]
        if len(nonzero_cells) == 0:
            print(f"    (no cells with reads for this ATSE — skipping cell check)", flush=True)
            continue
        probe_cells = np.random.choice(nonzero_cells, min(5, len(nonzero_cells)), replace=False)

        jc_sub = np.asarray(jc_csr[:, juncs][probe_cells, :].todense())
        at_sub = np.asarray(at_csr[:, juncs][probe_cells, :].todense())

        for k, ci in enumerate(probe_cells):
            jv = jc_sub[k]
            av = at_sub[k]
            atse_total = int(av.max())
            all_eq = bool(np.all(av == atse_total))
            status = "✓" if all_eq else "✗ MISMATCH"
            print(f"    cell {ci}: junc={jv}  atse={av}  total={atse_total}  {status}",
                  flush=True)
            if not all_eq:
                errors += 1

    assert errors == 0, f"CRITICAL: {errors} ATSE cluster invariant violations"
    print(f"\n  ✓ Spot-check passed ({len(sample_atses)} ATSEs, all with reads)",
          flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--easysci-splice-h5ad", required=True,
                        help="EasySci splice merged_anndata.h5ad (mm39 junction IDs)")
    parser.add_argument("--easysci-ge-h5ad", required=True,
                        help="EasySci GE metacell h5ad (metacell_RT_anndata_{size}.h5ad)")
    parser.add_argument("--liftover-atse", required=True,
                        help="mm39→mm10 liftover ATSE file (*_lifted_mm39.txt.gz)")
    parser.add_argument("--ref-splice-h5ad", required=True,
                        help="Filtered mouse splice h5ad — junction space + ATSE groupings reference")
    parser.add_argument("--ref-ge-h5ad", required=True,
                        help="Filtered mouse GE h5ad — gene space reference")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for splice and GE h5ads")
    parser.add_argument("--size", required=True,
                        help="Metacell size label (e.g. 500, 1000, 2000, 4000)")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.output_dir, exist_ok=True)
    out_splice = os.path.join(args.output_dir, f"easysci_splice_model_ready_{args.size}_{date_str}.h5ad")
    out_ge     = os.path.join(args.output_dir, f"easysci_ge_model_ready_{args.size}_{date_str}.h5ad")

    print(f"=== EasySci prep — size {args.size} ===", flush=True)
    print(f"  Date: {date_str}", flush=True)

    # ── 1. Load EasySci splice anndata ───────────────────────────────────────
    print("\n[1] Loading EasySci splice anndata...", flush=True)
    easysci_sp = ad.read_h5ad(args.easysci_splice_h5ad)
    N_sp = easysci_sp.n_obs
    print(f"  Shape: {easysci_sp.shape}", flush=True)
    print(f"  Layers: {list(easysci_sp.layers.keys())}", flush=True)
    print(f"  Obs: {list(easysci_sp.obs.columns)}", flush=True)

    # ── 2. Load liftover (mm39 → mm10) ───────────────────────────────────────
    print("\n[2] Loading liftover ATSE file...", flush=True)
    liftover = pd.read_csv(args.liftover_atse, sep="\t",
                           usecols=["junction_id", "mouse_foundation_junction_id"])
    liftover = liftover.drop_duplicates(subset="junction_id").reset_index(drop=True)
    liftover_dict = liftover.set_index("junction_id")["mouse_foundation_junction_id"].to_dict()
    print(f"  Liftover entries: {len(liftover_dict):,}", flush=True)

    # ── 3. Load reference splice var only (junction space + ATSE groupings) ──
    print("\n[3] Reading reference splice h5ad var...", flush=True)
    ref_sp_var = read_h5ad_var(args.ref_splice_h5ad)
    model_junc_ids  = ref_sp_var["junction_id"].values
    model_event_ids = ref_sp_var["event_id"].values
    model_junc_order = {jid: i for i, jid in enumerate(model_junc_ids)}
    J = len(model_junc_ids)
    n_atses = len(np.unique(model_event_ids))
    print(f"  Model: {J:,} junctions, {n_atses:,} ATSEs", flush=True)

    # ── 4. Map EasySci mm39 junction IDs → mm10 column indices ───────────────
    print("\n[4] Mapping mm39 → mm10 junction IDs...", flush=True)
    easysci_junc_ids = easysci_sp.var["junction_id"].values  # mm39

    mm10_ids  = np.array([liftover_dict.get(jid) for jid in easysci_junc_ids])
    model_col = np.array([
        model_junc_order.get(mid, -1) if mid is not None else -1
        for mid in mm10_ids
    ])
    n_mapped = (model_col >= 0).sum()
    print(f"  EasySci junctions mappable: {n_mapped:,} / {len(easysci_junc_ids):,} "
          f"({100*n_mapped/len(easysci_junc_ids):.1f}%)", flush=True)
    print(f"  Model junction coverage: {n_mapped:,} / {J:,} ({100*n_mapped/J:.1f}%)", flush=True)

    # ── 5. Build aligned cell_by_junction_matrix (N × J) ─────────────────────
    print("\n[5] Building aligned cell_by_junction_matrix...", flush=True)
    jc_raw = sp.csr_matrix(easysci_sp.layers["cell_by_junction_matrix"]).astype(np.int64)

    mappable_mask = model_col >= 0
    src_cols = np.where(mappable_mask)[0]
    dst_cols = model_col[src_cols]

    # Deduplicate: two mm39 junctions → same mm10 → keep higher-total-count one
    jc_totals = np.asarray(jc_raw[:, src_cols].sum(axis=0)).ravel()
    unique_dst, first_occ = {}, {}
    for i, (sc, dc) in enumerate(zip(src_cols, dst_cols)):
        if dc not in unique_dst or jc_totals[i] > jc_totals[first_occ[dc]]:
            unique_dst[dc] = sc
            first_occ[dc] = i

    final_src = np.array(list(unique_dst.values()))
    final_dst = np.array(list(unique_dst.keys()))
    sort_order = np.argsort(final_dst)
    final_src  = final_src[sort_order]
    final_dst  = final_dst[sort_order]
    print(f"  Unique mm10 junctions after dedup: {len(final_dst):,}", flush=True)

    jc_sub = jc_raw[:, final_src].tocoo()
    jc_aligned = sp.coo_matrix(
        (jc_sub.data, (jc_sub.row, final_dst[jc_sub.col])),
        shape=(N_sp, J)
    ).tocsr().astype(np.int64)
    print(f"  jc_aligned: {jc_aligned.shape}  nnz={jc_aligned.nnz:,}", flush=True)

    # Reverse mapping: mm10 column index → original mm39 junction ID (for spot-check)
    mm10_col_to_mm39 = {int(dst): easysci_junc_ids[int(src)]
                        for src, dst in zip(final_src, final_dst)}

    # ── 6. Recompute cluster matrix in mm10 ATSE space ───────────────────────
    print("\n[6] Recomputing cell_by_cluster_matrix in mm10 ATSE space...", flush=True)
    print("  (EasySci mm39 cluster values discarded — different ATSE groupings)", flush=True)
    at_aligned, unique_events, atse_idx = recompute_cluster_matrix(jc_aligned, model_event_ids)
    print(f"  at_aligned: {at_aligned.shape}  nnz={at_aligned.nnz:,}", flush=True)
    print(f"  nnz — junction: {jc_aligned.nnz:,}  cluster: {at_aligned.nnz:,}", flush=True)
    print(f"  (cluster nnz >= junction nnz is expected — zero-read junctions in "
          f"observed ATSEs get ATSE total stored)", flush=True)

    # ── 7. Orphan check: every junction entry must have non-zero ATSE total ──
    print("\n[7] Orphan check...", flush=True)
    jc_rows, jc_cols = jc_aligned.nonzero()
    at_at_junc = np.asarray(at_aligned[jc_rows, jc_cols]).ravel()
    orphan_nnz = int((at_at_junc == 0).sum())
    assert orphan_nnz == 0, (
        f"CRITICAL: {orphan_nnz:,} junction entries have zero ATSE total (PSI denominator=0)"
    )
    print(f"  ✓ All {jc_aligned.nnz:,} junction entries have non-zero ATSE total", flush=True)

    # ── 8. Spot-check ATSE cluster invariant ─────────────────────────────────
    print("\n[8] Spot-checking ATSE cluster invariant...", flush=True)
    spot_check_cluster(jc_aligned, at_aligned, model_event_ids,
                       ref_sp_var, mm10_col_to_mm39)

    # ── 9. Compute junc_ratio using cluster sparsity ─────────────────────────
    print("\n[9] Computing junc_ratio...", flush=True)
    at_coo = at_aligned.tocoo()
    jc_vals = np.asarray(jc_aligned[at_coo.row, at_coo.col]).ravel().astype(np.float32)
    at_vals = at_coo.data.astype(np.float32)
    ratio_data = jc_vals / at_vals
    junc_ratio = sp.coo_matrix(
        (ratio_data, (at_coo.row, at_coo.col)), shape=(N_sp, J)
    ).tocsr()
    assert float(junc_ratio.data.min()) >= 0, "CRITICAL: negative PSI"
    assert float(junc_ratio.data.max()) <= 1.0 + 1e-5, "CRITICAL: PSI > 1"
    print(f"  junc_ratio: nnz={junc_ratio.nnz:,}  "
          f"min={junc_ratio.data.min():.3f}  max={junc_ratio.data.max():.3f}  "
          f"mean(nz)={junc_ratio.data.mean():.3f}", flush=True)

    # ── 10. Build splice var + coverage flag + original mm39 junction IDs ───
    var_cols = ["junction_id", "event_id", "gene_id", "gene_name"]
    if "n_junctions_in_atse" in ref_sp_var.columns:
        var_cols.append("n_junctions_in_atse")
    new_sp_var = ref_sp_var[var_cols].copy()

    # Add original EasySci mm39 junction ID for each mm10 junction
    mm39_col_arr = np.array([mm10_col_to_mm39.get(j, "") for j in range(J)], dtype=object)
    new_sp_var["original_mm39_junction_id"] = mm39_col_arr

    covered = np.zeros(J, dtype=bool)
    covered[final_dst] = True
    new_sp_var["has_easysci_coverage"] = covered
    new_sp_var.index = new_sp_var["junction_id"].values
    print(f"\n  Junctions with EasySci coverage: {covered.sum():,} / {J:,}", flush=True)
    print(f"  Junctions zero-filled: {(~covered).sum():,}", flush=True)
    # Show a few examples of the mm10 ↔ mm39 mapping
    covered_idxs = np.where(covered)[0][:4]
    for ci in covered_idxs:
        print(f"    mm10={new_sp_var['junction_id'].iloc[ci]}  "
              f"mm39={new_sp_var['original_mm39_junction_id'].iloc[ci]}  "
              f"gene={new_sp_var['gene_name'].iloc[ci]}", flush=True)

    # ── 11. Set splice obs_names ──────────────────────────────────────────────
    easysci_sp.obs_names = easysci_sp.obs["Main_cluster_name_wkmeans"].values
    sp_obs = easysci_sp.obs.copy()

    # ── 12. Load EasySci GE metacell anndata ─────────────────────────────────
    print("\n[12] Loading EasySci GE metacell anndata...", flush=True)
    easysci_ge = ad.read_h5ad(args.easysci_ge_h5ad)
    easysci_ge.obs_names = easysci_ge.obs["Main_cluster_name_wkmeans"].values
    N_ge = easysci_ge.n_obs
    print(f"  GE shape: {easysci_ge.shape}", flush=True)
    print(f"  GE layers: {list(easysci_ge.layers.keys())}", flush=True)

    # ── 13. Read reference GE var only (gene space) ──────────────────────────
    print("\n[13] Reading reference GE h5ad var...", flush=True)
    ref_ge_var = read_h5ad_var(args.ref_ge_h5ad)
    G = len(ref_ge_var)
    print(f"  Reference gene space: {G:,} genes", flush=True)

    def strip_version(gid):
        s = str(gid)
        return s.split(".")[0] if "." in s else s

    if "gene_id" in easysci_ge.var.columns:
        ge_gene_ids = easysci_ge.var["gene_id"].astype(str).values
    else:
        ge_gene_ids = easysci_ge.var_names.astype(str).values

    ge_lookup = {}
    for col_idx, gid in enumerate(ge_gene_ids):
        ge_lookup[gid] = col_idx
        ge_lookup[strip_version(gid)] = col_idx

    if "gene_id" in ref_ge_var.columns:
        ref_gene_ids = ref_ge_var["gene_id"].astype(str).values
    else:
        ref_gene_ids = ref_ge_var.index.astype(str).values

    ge_src = []
    ge_dst = []
    for dst_j, ref_gid in enumerate(ref_gene_ids):
        src_j = ge_lookup.get(str(ref_gid)) or ge_lookup.get(strip_version(str(ref_gid)))
        if src_j is not None:
            ge_src.append(src_j)
            ge_dst.append(dst_j)

    ge_src = np.array(ge_src)
    ge_dst = np.array(ge_dst)
    pct_ge = 100 * len(ge_dst) / G
    print(f"  Genes matched: {len(ge_dst):,} / {G:,} ({pct_ge:.1f}%)", flush=True)
    print(f"  Genes zero-filled: {G - len(ge_dst):,}", flush=True)

    ref_gnames = ref_ge_var["gene_name"].values if "gene_name" in ref_ge_var.columns \
                 else ref_ge_var.index.values

    if pct_ge < 80:
        print(f"  *** WARNING: only {pct_ge:.1f}% of reference genes matched — "
              f"check gene_id format (Ensembl version stripping, gene symbol vs ID mismatch)",
              flush=True)
        unmatched_mask = np.ones(G, dtype=bool)
        unmatched_mask[ge_dst] = False
        print(f"  Example UNMATCHED reference genes (gene_name | gene_id):", flush=True)
        for idx in np.where(unmatched_mask)[0][:8]:
            print(f"    {ref_gnames[idx]:20s}  {ref_gene_ids[idx]}", flush=True)
        print(f"  Example EasySci gene_ids: {ge_gene_ids[:5].tolist()}", flush=True)
    else:
        print(f"  ✓ Gene coverage {pct_ge:.1f}% looks reasonable", flush=True)
        print(f"  Example matched genes (gene_name | ref gene_id → EasySci gene_id):", flush=True)
        for d, s in zip(ge_dst[:8], ge_src[:8]):
            print(f"    {ref_gnames[d]:20s}  {ref_gene_ids[d]} → {ge_gene_ids[s]}", flush=True)
        # Also show a few unmatched so we know what's missing
        unmatched_mask = np.ones(G, dtype=bool)
        unmatched_mask[ge_dst] = False
        unmatched_idxs = np.where(unmatched_mask)[0]
        if len(unmatched_idxs):
            print(f"  Example zero-filled genes (present in mouse ref, absent from EasySci):",
                  flush=True)
            for idx in unmatched_idxs[:5]:
                print(f"    {ref_gnames[idx]:20s}  {ref_gene_ids[idx]}", flush=True)

    # ── 14. Build aligned GE matrices ────────────────────────────────────────
    print("\n[14] Building aligned GE matrices...", flush=True)

    def align_ge_matrix(mat):
        mat = sp.csr_matrix(mat)
        sub = mat[:, ge_src].tocoo()
        return sp.coo_matrix(
            (sub.data, (sub.row, ge_dst[sub.col])),
            shape=(N_ge, G)
        ).tocsr()

    ge_X_aligned = align_ge_matrix(easysci_ge.X)
    ge_layers_aligned = {}
    for lname, layer in easysci_ge.layers.items():
        ge_layers_aligned[lname] = align_ge_matrix(layer)
        print(f"  GE layer '{lname}' aligned: nnz={ge_layers_aligned[lname].nnz:,}", flush=True)

    ge_coverage = np.zeros(G, dtype=bool)
    ge_coverage[ge_dst] = True

    ge_var_cols = [c for c in ["gene_id", "gene_name", "gene_type", "gene_biotype"]
                   if c in ref_ge_var.columns]
    new_ge_var = ref_ge_var[ge_var_cols].copy()
    new_ge_var["has_easysci_coverage"] = ge_coverage
    new_ge_var.index = ref_ge_var.index

    ge_obs = easysci_ge.obs.copy()

    # ── 15. Intersect obs (metacells present in both splice and GE) ──────────
    print("\n[15] Intersecting splice ∩ GE metacell obs...", flush=True)
    sp_names = set(easysci_sp.obs_names)
    ge_names = set(easysci_ge.obs_names)
    common_ordered = [x for x in easysci_ge.obs_names if x in sp_names]
    print(f"  Splice: {N_sp}  GE: {N_ge}  Common: {len(common_ordered)}", flush=True)

    sp_idx = [list(easysci_sp.obs_names).index(x) for x in common_ordered]
    ge_idx = [list(easysci_ge.obs_names).index(x) for x in common_ordered]

    # ── 16. Assemble and save splice output h5ad ─────────────────────────────
    print(f"\n[16] Saving splice → {out_splice}", flush=True)
    out_sp_adata = ad.AnnData(
        X    = junc_ratio[sp_idx, :],
        obs  = easysci_sp.obs.iloc[sp_idx].copy(),
        var  = new_sp_var,
        layers={
            "junc_ratio":              junc_ratio[sp_idx, :],
            "cell_by_junction_matrix": jc_aligned[sp_idx, :],
            "cell_by_cluster_matrix":  at_aligned[sp_idx, :],
        }
    )
    out_sp_adata.obs_names = [common_ordered[i] for i in range(len(common_ordered))]
    out_sp_adata.var_names = new_sp_var.index
    out_sp_adata.write_h5ad(out_splice, compression="gzip")
    print(f"  Done. Shape: {out_sp_adata.shape}", flush=True)

    # ── 17. Assemble and save GE output h5ad ─────────────────────────────────
    print(f"\n[17] Saving GE → {out_ge}", flush=True)
    ge_layers_sub = {lname: mat[ge_idx, :] for lname, mat in ge_layers_aligned.items()}
    out_ge_adata = ad.AnnData(
        X    = ge_X_aligned[ge_idx, :],
        obs  = easysci_ge.obs.iloc[ge_idx].copy(),
        var  = new_ge_var,
        layers=ge_layers_sub
    )
    out_ge_adata.obs_names = [common_ordered[i] for i in range(len(common_ordered))]
    out_ge_adata.var_names = new_ge_var.index
    out_ge_adata.write_h5ad(out_ge, compression="gzip")
    print(f"  Done. Shape: {out_ge_adata.shape}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Summary (size={args.size}) ===", flush=True)
    print(f"  Splice:   {out_sp_adata.shape}  "
          f"junction coverage: {covered.sum():,}/{J:,} ({100*covered.sum()/J:.1f}%)", flush=True)
    print(f"  GE:       {out_ge_adata.shape}  "
          f"gene coverage: {len(ge_dst):,}/{G:,} ({100*len(ge_dst)/G:.1f}%)", flush=True)
    print(f"  Outputs:", flush=True)
    print(f"    Splice: {out_splice}", flush=True)
    print(f"    GE:     {out_ge}", flush=True)


if __name__ == "__main__":
    main()
