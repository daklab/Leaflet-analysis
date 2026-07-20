"""
Compare splice and GE anndata objects: Oct 2025 (102025) vs May 2026 (20260526).
- All coverage stats computed on ALL cells using scipy sparse ops (no sampling)
- Inspects obs_names/var_names formats + all .var columns to find right join keys
- Checks that shared cells have identical GE values in both versions
Output: text summary + PDF plots
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

OLD_SPLICE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/aligned_splicing_data_20251003_152725.h5ad"
OLD_GE     = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/aligned_gene_expression_data_20251003_152725.h5ad"
NEW_SPLICE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/splice_adata_aligned_2026-05-27.h5ad"
NEW_GE     = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/ge_adata_aligned_2026-05-27.h5ad"

OUT_PREFIX = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/version_comparison_102025_vs_20260526"

LABELS = {"old": "Oct2025 (102025)", "new": "May2026 (20260526)"}


# ── loading ────────────────────────────────────────────────────────────────────

def load_adata(path, label):
    print(f"  Loading {label} ...")
    adata = ad.read_h5ad(path)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    print(f"    shape={adata.shape}  nnz={adata.X.nnz:,}  dtype={adata.X.dtype}")
    return adata


# ── structure inspection ───────────────────────────────────────────────────────

def describe_structure(adata, label, lines):
    lines.append(f"\n--- {label} ---")

    # obs_names sample
    lines.append(f"  obs_names ({adata.obs_names.dtype}), first 5:")
    for v in adata.obs_names[:5]:
        lines.append(f"    {v!r}")

    # var_names sample
    lines.append(f"  var_names ({adata.var_names.dtype}), first 5:")
    for v in adata.var_names[:5]:
        lines.append(f"    {v!r}")

    # all .var columns with dtype and first 3 values
    lines.append(f"  .var columns ({len(adata.var.columns)}):")
    if len(adata.var.columns) == 0:
        lines.append("    (none)")
    for col in adata.var.columns:
        sample = list(adata.var[col].iloc[:3].astype(str))
        lines.append(f"    {col} ({adata.var[col].dtype}): {sample}")

    # all .obs columns with dtype and first 3 values
    lines.append(f"  .obs columns ({len(adata.obs.columns)}), sample values:")
    for col in adata.obs.columns:
        sample = list(adata.obs[col].iloc[:3].astype(str))
        lines.append(f"    {col} ({adata.obs[col].dtype}): {sample}")


# ── coverage (all cells, fully sparse) ────────────────────────────────────────

def sparse_coverage(adata):
    X = adata.X.tocsr()
    counts = np.asarray(X.sum(axis=1)).ravel()
    feats  = X.getnnz(axis=1)
    return counts, feats


# ── helpers ────────────────────────────────────────────────────────────────────

def percentile_summary(arr, label):
    pcts = [5, 10, 25, 50, 75, 90, 95]
    vals = np.nanpercentile(arr, pcts)
    tag  = "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, vals))
    return f"  {label}:  mean={arr.mean():.0f}  {tag}"


def find_col(obs_or_var, candidates):
    return next((c for c in candidates if c in obs_or_var.columns), None)


def cell_type_col(obs):
    return find_col(obs, ["cell_type", "cell_type_general", "celltype",
                          "broad_cell_type", "leiden", "louvain"])


def dataset_col(obs):
    return find_col(obs, ["dataset", "source", "batch", "study"])


# ── find best join key ─────────────────────────────────────────────────────────

def find_best_key(a, b, domain, lines):
    """
    Try obs_names/var_names first, then each column in .obs/.var,
    returning (key_a, key_b, shared_values) for the pair with most overlap.
    domain: 'obs' or 'var'
    """
    def values(adata, col):
        if col == "__index__":
            return set(adata.obs_names if domain == "obs" else adata.var_names)
        df = adata.obs if domain == "obs" else adata.var
        return set(df[col].astype(str))

    # Gather candidate column names from both objects
    df_a = a.obs if domain == "obs" else a.var
    df_b = b.obs if domain == "obs" else b.var
    cols_a = ["__index__"] + list(df_a.columns)
    cols_b = ["__index__"] + list(df_b.columns)

    best_n, best = 0, None
    for ca in cols_a:
        va = values(a, ca)
        for cb in cols_b:
            vb = values(b, cb)
            n  = len(va & vb)
            if n > best_n:
                best_n = n
                best = (ca, cb, sorted(va & vb))

    if best is None or best_n == 0:
        lines.append(f"  [{domain}] No overlap found across any column pair.")
        return None, None, []

    ca, cb, shared = best
    lines.append(f"  [{domain}] Best join: OLD.{ca} x NEW.{cb}  =>  {best_n:,} shared values")
    lines.append(f"    OLD.{ca} sample: {list(values(a, ca))[:3]}")
    lines.append(f"    NEW.{cb} sample: {list(values(b, cb))[:3]}")
    return ca, cb, shared


def get_col_values(adata, col, domain):
    if col == "__index__":
        return (adata.obs_names if domain == "obs" else adata.var_names).tolist()
    df = adata.obs if domain == "obs" else adata.var
    return df[col].astype(str).tolist()


# ── GE value consistency check ─────────────────────────────────────────────────

def extract_barcode(cell_id):
    """Extract the B0XXXXX barcode component shared across cell ID formats.
    OLD format: A10_B000120  -> B000120
    NEW format: A1-B000126-3_39_F-1-1 -> B000126
    Falls back to the full string if no B0XXXXX pattern found.
    """
    import re
    m = re.search(r'(B\d+)', str(cell_id))
    return m.group(1) if m else str(cell_id)


def compare_ge_values(old_ge, new_ge, lines):
    lines.append("\n" + "=" * 60)
    lines.append("GE VALUE CONSISTENCY (shared cells × shared genes)")
    lines.append("=" * 60)

    # ── gene join: gene_name column vs var_names ───────────────────────────
    old_gene_names = old_ge.var["gene_name"].astype(str).tolist()
    new_gene_names = new_ge.var_names.tolist()
    shared_genes_set = set(old_gene_names) & set(new_gene_names)
    lines.append(f"\n  Gene join: OLD.var['gene_name'] x NEW.var_names")
    lines.append(f"  Shared genes: {len(shared_genes_set):,} / old={old_ge.n_vars:,} / new={new_ge.n_vars:,}")

    # ── cell join: extract barcode from cell_id in both ───────────────────
    old_cell_id_col = "cell_id" if "cell_id" in old_ge.obs.columns else old_ge.obs_names.name
    new_cell_id_col = "cell_id" if "cell_id" in new_ge.obs.columns else new_ge.obs_names.name

    old_barcodes = pd.Series(
        [extract_barcode(v) for v in old_ge.obs[old_cell_id_col]],
        index=range(old_ge.n_obs)
    )
    new_barcodes = pd.Series(
        [extract_barcode(v) for v in new_ge.obs[new_cell_id_col]],
        index=range(new_ge.n_obs)
    )
    from collections import Counter
    old_counts = Counter(old_barcodes)
    new_counts = Counter(new_barcodes)
    all_shared = set(old_barcodes) & set(new_barcodes)
    # Keep only barcodes that appear exactly once in both — duplicates are ambiguous
    shared_barcodes = {b for b in all_shared if old_counts[b] == 1 and new_counts[b] == 1}
    n_dup_old = sum(1 for b in all_shared if old_counts[b] > 1)
    n_dup_new = sum(1 for b in all_shared if new_counts[b] > 1)

    lines.append(f"\n  Cell join: barcode extracted from cell_id (B0XXXXX pattern)")
    lines.append(f"  Sample OLD barcodes: {list(old_barcodes[:3])}")
    lines.append(f"  Sample NEW barcodes: {list(new_barcodes[:3])}")
    lines.append(f"  Shared barcodes (any): {len(all_shared):,} / old={old_ge.n_obs:,} / new={new_ge.n_obs:,}")
    lines.append(f"  Duplicated in OLD: {n_dup_old:,}  Duplicated in NEW: {n_dup_new:,} (excluded from comparison)")
    lines.append(f"  Unambiguous shared cells: {len(shared_barcodes):,}")

    if not shared_barcodes or not shared_genes_set:
        lines.append("  Cannot compare GE values — no unambiguous shared cells or genes.")
        return

    # Build positional indices ordered by shared barcode so rows align between old and new
    shared_barcodes_sorted = sorted(shared_barcodes)
    shared_genes_sorted    = sorted(shared_genes_set)
    barcode_rank = {b: i for i, b in enumerate(shared_barcodes_sorted)}
    gene_rank    = {g: i for i, g in enumerate(shared_genes_sorted)}

    old_cell_idx = sorted(
        [i for i, b in enumerate(old_barcodes) if b in shared_barcodes],
        key=lambda i: barcode_rank[old_barcodes[i]]
    )
    new_cell_idx = sorted(
        [i for i, b in enumerate(new_barcodes) if b in shared_barcodes],
        key=lambda i: barcode_rank[new_barcodes[i]]
    )
    assert len(old_cell_idx) == len(new_cell_idx), \
        f"Cell index mismatch after dedup: old={len(old_cell_idx)} new={len(new_cell_idx)}"
    old_gene_idx = sorted(
        [i for i, g in enumerate(old_gene_names) if g in shared_genes_set],
        key=lambda i: gene_rank[old_gene_names[i]]
    )
    new_gene_idx = sorted(
        [i for i, g in enumerate(new_gene_names) if g in shared_genes_set],
        key=lambda i: gene_rank[new_gene_names[i]]
    )

    print(f"  Subsetting GE: {len(old_cell_idx):,} cells x {len(old_gene_idx):,} genes ...")
    old_X = old_ge.X.tocsr()[old_cell_idx, :][:, old_gene_idx]
    new_X = new_ge.X.tocsr()[new_cell_idx, :][:, new_gene_idx]

    print("  Computing sparse diff ...")
    diff = old_X - new_X
    if diff.nnz == 0:
        lines.append("  RESULT: all shared cells have IDENTICAL GE values in both versions.")
    else:
        max_diff   = np.abs(diff.data).max()
        total_elem = len(old_cell_idx) * len(old_gene_idx)
        lines.append(f"  RESULT: {diff.nnz:,} / {total_elem:,} elements differ")
        lines.append(f"          max |old - new| = {max_diff:.6g}")
        lines.append("          |diff| percentiles: " +
                     "  ".join(f"p{p}={v:.4g}"
                                for p, v in zip([1, 5, 25, 50, 75, 95, 99],
                                                np.percentile(np.abs(diff.data),
                                                              [1, 5, 25, 50, 75, 95, 99]))))


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_hist_pair(ax, old_vals, new_vals, xlabel, title):
    q99 = max(np.quantile(old_vals, 0.99), np.quantile(new_vals, 0.99))
    bins = np.linspace(0, q99, 80)
    ax.hist(old_vals, bins=bins, alpha=0.6, label=LABELS["old"], density=True)
    ax.hist(new_vals, bins=bins, alpha=0.6, label=LABELS["new"], density=True)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    lines = []

    print("=== Loading splice objects ===")
    old_sp = load_adata(OLD_SPLICE, "old_splice")
    new_sp = load_adata(NEW_SPLICE, "new_splice")

    print("\n=== Loading GE objects ===")
    old_ge = load_adata(OLD_GE, "old_ge")
    new_ge = load_adata(NEW_GE, "new_ge")

    # ── shapes ────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("SHAPES")
    lines.append("=" * 60)
    lines.append(f"Splice  OLD: {old_sp.n_obs:,} cells  x  {old_sp.n_vars:,} junctions")
    lines.append(f"Splice  NEW: {new_sp.n_obs:,} cells  x  {new_sp.n_vars:,} junctions")
    lines.append(f"GE      OLD: {old_ge.n_obs:,} cells  x  {old_ge.n_vars:,} genes")
    lines.append(f"GE      NEW: {new_ge.n_obs:,} cells  x  {new_ge.n_vars:,} genes")

    # ── structure: obs_names, var_names, all .var/.obs columns + samples ──────
    lines.append("\n" + "=" * 60)
    lines.append("STRUCTURE (obs_names / var_names formats + all .var columns)")
    lines.append("=" * 60)
    for label, adata in [("OLD splice", old_sp), ("NEW splice", new_sp),
                          ("OLD GE",     old_ge), ("NEW GE",     new_ge)]:
        describe_structure(adata, label, lines)

    # ── junction overlap (try all var column combos) ───────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("JUNCTION OVERLAP (best column pair)")
    lines.append("=" * 60)
    find_best_key(old_sp, new_sp, "var", lines)

    # ── gene overlap ──────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("GENE OVERLAP (best column pair)")
    lines.append("=" * 60)
    find_best_key(old_ge, new_ge, "var", lines)

    # ── cell overlap (splice) ─────────────────────────────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("CELL OVERLAP (splice, best column pair)")
    lines.append("=" * 60)
    find_best_key(old_sp, new_sp, "obs", lines)

    # ── dataset composition ───────────────────────────────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("DATASET COMPOSITION")
    lines.append("=" * 60)
    for lbl, adata in [("OLD splice", old_sp), ("NEW splice", new_sp)]:
        dcol = dataset_col(adata.obs)
        if dcol:
            vc = adata.obs[dcol].value_counts()
            lines.append(f"\n  {lbl} ({dcol}):")
            for k, v in vc.items():
                lines.append(f"    {k}: {v:,}  ({100*v/adata.n_obs:.1f}%)")

    # ── cell type composition ─────────────────────────────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("CELL TYPE COMPOSITION (splice)")
    lines.append("=" * 60)
    ct_old_col = cell_type_col(old_sp.obs)
    ct_new_col = cell_type_col(new_sp.obs)
    ct_df = None
    if ct_old_col and ct_new_col:
        # Normalise case for comparison
        old_ct = old_sp.obs[ct_old_col].str.upper().value_counts().rename("old")
        new_ct = new_sp.obs[ct_new_col].str.upper().value_counts().rename("new")
        ct_df  = pd.concat([old_ct, new_ct], axis=1).fillna(0).astype(int)
        ct_df["delta"] = ct_df["new"] - ct_df["old"]
        lines.append(f"  (old col={ct_old_col}, new col={ct_new_col}, both uppercased for comparison)")
        lines.append(ct_df.sort_values("delta").to_string())

    # ── coverage ──────────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 60)
    lines.append("COVERAGE DISTRIBUTIONS (all cells, sparse matrix ops)")
    lines.append("=" * 60)

    print("\n=== Splice coverage (sparse) ===")
    old_sp_counts, old_sp_feats = sparse_coverage(old_sp)
    new_sp_counts, new_sp_feats = sparse_coverage(new_sp)
    lines.append("\nSplice total reads per cell:")
    lines.append(percentile_summary(old_sp_counts, LABELS["old"]))
    lines.append(percentile_summary(new_sp_counts, LABELS["new"]))
    lines.append("\nSplice junctions detected per cell:")
    lines.append(percentile_summary(old_sp_feats, LABELS["old"]))
    lines.append(percentile_summary(new_sp_feats, LABELS["new"]))

    print("\n=== GE coverage (sparse) ===")
    old_ge_counts, old_ge_feats = sparse_coverage(old_ge)
    new_ge_counts, new_ge_feats = sparse_coverage(new_ge)
    lines.append("\nGE total UMI counts per cell:")
    lines.append(percentile_summary(old_ge_counts, LABELS["old"]))
    lines.append(percentile_summary(new_ge_counts, LABELS["new"]))
    lines.append("\nGE genes detected per cell:")
    lines.append(percentile_summary(old_ge_feats, LABELS["old"]))
    lines.append(percentile_summary(new_ge_feats, LABELS["new"]))

    # ── GE value consistency ──────────────────────────────────────────────────
    print("\n=== GE value consistency check ===")
    compare_ge_values(old_ge, new_ge, lines)

    # ── write text ────────────────────────────────────────────────────────────
    txt_path = OUT_PREFIX + ".txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nText summary → {txt_path}")
    print("\n".join(lines))

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating PDF ...")
    pdf_path = OUT_PREFIX + ".pdf"
    with PdfPages(pdf_path) as pdf:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Splice coverage per cell (all cells)", fontsize=11)
        plot_hist_pair(axes[0], old_sp_counts, new_sp_counts, "Total junction reads", "Junction reads / cell")
        plot_hist_pair(axes[1], old_sp_feats,  new_sp_feats,  "Junctions detected",  "Junctions detected / cell")
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("GE coverage per cell (all cells)", fontsize=11)
        plot_hist_pair(axes[0], old_ge_counts, new_ge_counts, "Total UMI counts", "UMI counts / cell")
        plot_hist_pair(axes[1], old_ge_feats,  new_ge_feats,  "Genes detected",   "Genes detected / cell")
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        if ct_df is not None:
            top_n   = 40
            plot_df = pd.concat([ct_df.sort_values("delta").head(top_n // 2),
                                  ct_df.sort_values("delta").tail(top_n // 2)])
            colors  = ["#d62728" if x < 0 else "#1f77b4" for x in plot_df["delta"]]
            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.28)))
            ax.barh(plot_df.index, plot_df["delta"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Delta cell count (new − old)")
            ax.set_title(f"Cell type changes (top/bottom {top_n//2})", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"PDF → {pdf_path}")


if __name__ == "__main__":
    main()
