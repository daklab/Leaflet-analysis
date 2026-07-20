#!/usr/bin/env python
"""
test_warmstart.py — validate that the pi_init guide fix eliminates ELBO spikes at batch
transitions and that pi differentiates across batches.

Mirrors the EXACT flow of run_training.py:
  from_anndata() → train() → get_all_variables() → extract psi/pi/phi → pass to next batch

Usage (CPU, ~5-10 min):
    PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
    TRAIN=$WD/MODEL_INPUT/20260526/train_splice_model_ready_weighted_centered_2026-06-14.h5ad
    $PYTHON tests/test_warmstart.py $TRAIN

Two things to check in output:
  1. pi_spread should GROW across batches (pi differentiating)
  2. ELBO at START of batch N+1 should be close to END of batch N (warm start)
     If it spikes back up to the initial level → fix didn't take
"""

import argparse
import sys
import numpy as np
import anndata as ad
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa import LeafletFA
import leafletfa.waypoints as wayp

OUT_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


DEFAULT_CELL_TYPES = ["EXCITATORY NEURON", "INHIBITORY NEURON", "FIBROBLAST"]


def load_subset(h5ad_path, n_cells=2000, seed=42, cell_types=None):
    print(f"Loading subset of {n_cells} cells...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    rng = np.random.default_rng(seed)
    if cell_types and "broad_cell_type" in adata.obs.columns:
        bct = adata.obs["broad_cell_type"].astype(str)
        per_type = n_cells // len(cell_types)
        idx = []
        for ct in cell_types:
            ct_idx = np.where(bct == ct)[0]
            chosen = rng.choice(ct_idx, size=min(per_type, len(ct_idx)), replace=False)
            idx.extend(chosen.tolist())
            print(f"  {ct}: {len(chosen)} cells")
        idx = np.array(sorted(idx))
    elif "broad_cell_type" in adata.obs.columns:
        bct = adata.obs["broad_cell_type"].astype(str)
        unique = bct.unique()
        per_type = max(1, n_cells // len(unique))
        idx = []
        for ct in unique:
            ct_idx = np.where(bct == ct)[0]
            idx.extend(rng.choice(ct_idx, size=min(per_type, len(ct_idx)), replace=False).tolist())
        idx = np.array(sorted(idx[:n_cells]))
    else:
        idx = np.sort(rng.choice(len(adata), size=n_cells, replace=False))
    subset = adata[idx].to_memory()
    adata.file.close()
    print(f"  {subset.shape[0]} cells × {subset.shape[1]} junctions")
    return subset


def plot_phi_pca(phi_matrix, cell_type_labels, waypoint_indices_in_phi, out_path, K):
    """PCA of learned phi (cell factor activities), coloured by cell type."""
    from sklearn.decomposition import PCA
    if phi_matrix.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(phi_matrix)
        xlabel = f"Phi-PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"Phi-PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        coords = phi_matrix
        xlabel = "Factor 1"; ylabel = "Factor 2"

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_types = sorted(set(cell_type_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_types))
    for i, ct in enumerate(unique_types):
        mask = np.array(cell_type_labels) == ct
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.5, color=cmap(i), label=ct)
    if waypoint_indices_in_phi is not None and len(waypoint_indices_in_phi):
        ax.scatter(coords[waypoint_indices_in_phi, 0], coords[waypoint_indices_in_phi, 1],
                   s=120, marker="*", color="red", edgecolors="black",
                   linewidths=0.5, zorder=5, label="Waypoints")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"PCA of learned Φ (K={K}) — compare to raw data PCA")
    ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Phi PCA plot saved: {out_path}")


def plot_pca_waypoints(X_pca, waypoint_indices, cell_type_labels, out_path):
    """PCA scatter coloured by cell type; waypoint cells shown as red stars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_types = sorted(set(cell_type_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_types))
    for i, ct in enumerate(unique_types):
        mask = np.array(cell_type_labels) == ct
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=8, alpha=0.5, color=cmap(i), label=ct)
    # overlay waypoints as red stars
    ax.scatter(X_pca[waypoint_indices, 0], X_pca[waypoint_indices, 1],
               s=120, marker="*", color="red", edgecolors="black",
               linewidths=0.5, zorder=5, label="Waypoints")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("PCA — waypoints used for PSI initialisation")
    ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Waypoint PCA plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_h5ad")
    parser.add_argument("--n-cells",      type=int, default=2000)
    parser.add_argument("--k",            type=int, default=10)
    parser.add_argument("--epochs-first", type=int, default=50,
                        help="SVI epochs for first batch (waypoint init)")
    parser.add_argument("--epochs-later", type=int, default=30,
                        help="SVI epochs for batches 2+")
    parser.add_argument("--n-batches",    type=int, default=4)
    parser.add_argument("--batch-size",   type=int, default=512)
    parser.add_argument("--cell-types",   type=str, default=None,
                        help="Comma-separated broad_cell_type values to restrict to "
                             f"(default: {','.join(DEFAULT_CELL_TYPES)})")
    args = parser.parse_args()

    K = args.k
    cell_types = [c.strip() for c in args.cell_types.split(",")] if args.cell_types else DEFAULT_CELL_TYPES
    print(f"Cell types: {cell_types}")
    adata = load_subset(args.train_h5ad, n_cells=args.n_cells, cell_types=cell_types)
    n_cells = len(adata)

    # Split cells into batches (like run_training.py per-pass permutation)
    cell_indices = np.random.permutation(n_cells)
    batches = []
    for b in range(args.n_batches):
        start = b * (n_cells // args.n_batches)
        end   = min((b+1) * (n_cells // args.n_batches), n_cells)
        batches.append(np.sort(cell_indices[start:end]))

    print(f"\nK={K}, {n_cells} cells, {args.n_batches} batches, "
          f"epochs: first={args.epochs_first} later={args.epochs_later}\n")

    # ── Compute waypoints on the FULL dataset before batching ──────────────────
    # Waypoints on just the first batch give PSI that overfits to those cells;
    # using the full dataset gives a global PSI that initialises every batch well.
    from sklearn.decomposition import TruncatedSVD
    print("Computing waypoints on full dataset...")
    n_pca = min(K, n_cells - 1, adata.shape[1] - 1)
    metacell_size = min(30, n_cells // K)

    if "junc_ratio_centered" in adata.layers:
        data_for_pca = adata.layers["junc_ratio_centered"]
    elif "junc_ratio" in adata.layers:
        data_for_pca = adata.layers["junc_ratio"]
    else:
        data_for_pca = adata.X

    svd = TruncatedSVD(n_components=n_pca, random_state=42)
    U = svd.fit_transform(data_for_pca)
    X_pca_full = U * svd.singular_values_          # (n_cells, n_pca)

    waypoints = wayp.max_min_sampling(X_pca_full, K, num_components=n_pca, seed=42)
    metacell_dict = wayp.assign_nearest_cells(waypoints, X_pca_full, num_nearest=metacell_size)

    if "junc_ratio" in adata.layers:
        rho_hat_full = adata.layers["junc_ratio"]
    else:
        rho_hat_full = adata.X

    psi_inits, phi_inits = wayp.generate_initializations(
        rho_hat_full, {K: waypoints}, {K: metacell_dict}, epsilon=0.001
    )
    # global_psi_wp: (J, K) — stored in varm without transpose, model.py transposes on load
    global_psi_wp = psi_inits[0] if not isinstance(psi_inits[0], torch.Tensor) else psi_inits[0].cpu().numpy()
    # global_phi_wp: (n_cells, K) — slice per batch
    global_phi_wp = phi_inits[0] if not isinstance(phi_inits[0], torch.Tensor) else phi_inits[0].cpu().numpy()
    print(f"  PSI init shape: {global_psi_wp.shape},  PHI init shape: {global_phi_wp.shape}")

    # PCA plot with waypoints highlighted
    ct_labels = adata.obs["broad_cell_type"].astype(str).tolist() if "broad_cell_type" in adata.obs.columns else ["unknown"] * n_cells
    plot_pca_waypoints(X_pca_full, waypoints, ct_labels,
                       OUT_DIR / f"test_warmstart_pca_waypoints_K{K}.png")
    print()

    # Global state
    global_psi       = None
    global_pi        = None
    global_alpha_pi  = None
    global_dir_conc  = None
    all_cell_assignments = np.zeros((n_cells, K))

    batch_best_elbos = []
    pi_per_batch     = []

    for b, batch_cell_indices in enumerate(batches):
        is_first = (b == 0)
        batch_adata = adata[batch_cell_indices].copy()
        num_epochs  = args.epochs_first if is_first else args.epochs_later

        print(f"=== Batch {b+1}/{args.n_batches}  ({len(batch_cell_indices)} cells, "
              f"{num_epochs} epochs) ===")

        # ── PSI init: waypoint PSI for batch 1, learned PSI for later batches ──
        if is_first:
            batch_adata.varm[f'psi_init_{K}_waypoints'] = global_psi_wp
            batch_adata.obsm[f'phi_init_{K}_waypoints'] = global_phi_wp[batch_cell_indices]
        else:
            # global_psi is (K, J) from model — store as (J, K), model.py transposes on load
            batch_adata.varm[f'psi_init_{K}_waypoints'] = global_psi.T

            # Use learned phi if this cell was already trained; otherwise fall back to
            # global waypoint phi (NOT random — that's what caused the 2× ELBO jump).
            learned = all_cell_assignments[batch_cell_indices]
            if learned.sum() > 0:
                batch_adata.obsm[f'phi_init_{K}_waypoints'] = learned
            else:
                batch_adata.obsm[f'phi_init_{K}_waypoints'] = global_phi_wp[batch_cell_indices]

        # ── Instantiate LeafletFA (mirrors run_training.py LeafletFA() call) ──
        pi_tensor       = torch.tensor(global_pi,      dtype=torch.float32) if global_pi      is not None else None
        alpha_pi_tensor = torch.tensor(global_alpha_pi,dtype=torch.float32) if global_alpha_pi is not None else None
        dir_conc_tensor = torch.tensor([global_dir_conc], dtype=torch.float32, device=device) if global_dir_conc is not None else None

        model = LeafletFA(
            adata           = batch_adata,
            K               = K,
            waypoints_use   = True,          # always True after first batch (same as run_training.py line 320)
            pi_init         = pi_tensor,
            alpha_pi_init   = alpha_pi_tensor,
            delta_fixed     = dir_conc_tensor,
            num_epochs      = num_epochs,
            ELBO_num_particles = 2,
            num_samples     = 10,
            lr              = 0.001,
            gamma           = 0.1,
            min_delta       = 1,
            patience        = 20,
            print_epochs    = 10,
            log_wandb       = False,
            enable_cpu_optimization = False,
        )

        # ── EXACT run_training.py sequence ──
        model.from_anndata()           # line 339 — extract sparse tensors from adata
        if device.type == 'cuda':
            model.initialize_triton_mask()
        model.train(num_initializations=1)   # line 345
        model.get_all_variables()      # line 348 — populates model.pi, model.psi, model.assign_post

        # Debug: show what keys are in the summary stats
        if hasattr(model, 'latent_results'):
            ss_keys = list(model.latent_results[model.best_init]['summary_stats'].keys())
            print(f"  summary_stats keys: {ss_keys}")

        # ── Extract global state (mirrors run_training.py lines 350-376) ──
        if hasattr(model, 'psi'):
            global_psi = model.psi if isinstance(model.psi, np.ndarray) else model.psi.cpu().numpy()
        if hasattr(model, 'pi'):
            global_pi = model.pi if isinstance(model.pi, np.ndarray) else model.pi.cpu().numpy()
        if hasattr(model, 'alpha_pi'):
            global_alpha_pi = model.alpha_pi if isinstance(model.alpha_pi, np.ndarray) else model.alpha_pi.cpu().numpy()
        if hasattr(model, 'dir_conc'):
            val = model.dir_conc
            global_dir_conc = val.cpu().item() if isinstance(val, torch.Tensor) else float(val)
        if hasattr(model, 'assign_post'):
            ap = model.assign_post
            if hasattr(ap, 'detach'):   # torch tensor
                ap = ap.detach().cpu().numpy()
            elif not isinstance(ap, np.ndarray):
                ap = np.array(ap)
            print(f"  assign_post: shape={ap.shape}, sum={ap.sum():.2f}")
            all_cell_assignments[batch_cell_indices] = ap
        else:
            print(f"  WARNING: model has no assign_post — phi plot will be skipped")

        best_elbo = model.best_elbo if hasattr(model, 'best_elbo') else None
        batch_best_elbos.append(float(best_elbo) if best_elbo is not None else np.nan)
        pi_per_batch.append(global_pi.copy())

        spread = global_pi.max() - global_pi.min()
        print(f"  best_elbo: {best_elbo:.4e}")
        print(f"  pi: {np.round(global_pi, 4)}")
        print(f"  pi_spread: {spread:.4f}  {'← DIFFERENTIATING ✓' if spread > 0.02 else '← still flat'}")
        print()

        del model, batch_adata

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. π per factor across batches
    ax = axes[0]
    for b_idx, pi in enumerate(pi_per_batch):
        ax.plot(range(K), pi, marker='o', label=f"Batch {b_idx+1}")
    ax.axhline(1/K, color='gray', linestyle='--', label=f"Uniform 1/{K}")
    ax.set_xlabel("Factor"); ax.set_ylabel("π weight")
    ax.set_title("π per factor\n(should diverge from dashed line)")
    ax.legend(fontsize=7)

    # 2. π spread per batch
    ax2 = axes[1]
    spreads = [pi.max() - pi.min() for pi in pi_per_batch]
    ax2.bar(range(1, len(spreads)+1), spreads, color="steelblue")
    ax2.set_xlabel("Batch"); ax2.set_ylabel("max π − min π")
    ax2.set_title("π spread per batch\n(should increase)")

    # 3. ELBO per batch (continuity check)
    ax3 = axes[2]
    ax3.plot(range(1, len(batch_best_elbos)+1), batch_best_elbos, marker='o', color='steelblue')
    ax3.set_xlabel("Batch"); ax3.set_ylabel("Best ELBO")
    ax3.set_title("Best ELBO per batch\n(should trend down, not spike)")

    plt.suptitle(f"Warm-start test | K={K}, {args.n_cells} cells", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / f"test_warmstart_K{K}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")

    # ── Phi PCA plot (compare learned factor space to raw data PCA) ──────────
    print(f"all_cell_assignments total sum = {all_cell_assignments.sum():.2f}")
    if all_cell_assignments.sum() > 0:
        plot_phi_pca(
            all_cell_assignments, ct_labels,
            waypoints,                            # same waypoint cell indices as raw PCA
            OUT_DIR / f"test_warmstart_phi_pca_K{K}.png",
            K,
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    final_spread = spreads[-1]
    print("\n=== VERDICT ===")
    if final_spread > 0.02:
        print(f"✓  π IS differentiating (final spread={final_spread:.4f})")
        print("   Safe to resubmit training runs.")
    else:
        print(f"✗  π still near-uniform (final spread={final_spread:.4f})")
        print("   Fix did not work — check that LeafletFA package was reinstalled.")


if __name__ == "__main__":
    main()
