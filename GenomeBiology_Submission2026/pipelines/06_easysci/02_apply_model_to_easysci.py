#!/usr/bin/env python
"""
Step 12b: Apply a trained LeafletFA model to prepared EasySci metacell data.

Takes the output of 01_prep_easysci_for_model.py (EasySci aligned to model
junction space) and runs LeafletFA with the PSI matrix fixed to the trained
values, inferring only the cell factor assignments (phi / assign_post).

Usage:
    python 02_apply_model_to_easysci.py \\
        --easysci-h5ad <easysci_model_ready_*.h5ad> \\
        --model-run-dir <LEAFLETFA_MODELS/2026-06-01/run_N/> \\
        --output-dir <output/easysci_leafletfa/>

    The script writes:
      <output_dir>/easysci_leafletfa_run_N.h5ad   — EasySci adata with obsm
      <output_dir>/easysci_leafletfa_run_N_assign_post.csv
      <output_dir>/easysci_leafletfa_run_N_pi.csv

SLURM (GPU recommended for speed, CPU works too):
    sbatch --mem=32G -p gpu --gres=gpu:1 \\
        --wrap "python pipelines/06_easysci/02_apply_model_to_easysci.py ..."

    # Or CPU:
    sbatch --mem=32G -p cpu \\
        --wrap "python pipelines/06_easysci/02_apply_model_to_easysci.py ..."
"""

import argparse
import os
import sys
import gzip
import pickle

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_tensor_type(
    "torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor"
)
torch.manual_seed(0)
np.random.seed(0)

# LeafletFA installed at /gpfs/commons/home/kisaev/LeafletFA
from leafletfa import LeafletFA


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(run_dir):
    pkl_path = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    data = {}
    with gzip.open(pkl_path, "rb") as f:
        while True:
            try:
                data.update(pickle.load(f))
            except EOFError:
                break
    return data


def load_run_summary(run_dir):
    path = os.path.join(run_dir, "run_summary.csv")
    return pd.read_csv(path).iloc[0].to_dict()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--easysci-h5ad", required=True,
                        help="Prepared EasySci h5ad from 01_prep_easysci_for_model.py")
    parser.add_argument("--model-run-dir", required=True,
                        help="Path to a single run directory (e.g. run_0/) "
                             "containing leafletfa_model.pkl.gz and run_summary.csv")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write output files")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.model_run_dir.rstrip("/"))

    # ── Load EasySci data ─────────────────────────────────────────────────────
    print("Loading EasySci h5ad...")
    easysci = ad.read_h5ad(args.easysci_h5ad)
    print(f"  Shape: {easysci.shape}")
    print(f"  Layers: {list(easysci.layers.keys())}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model_run_dir} ...")
    model_data = load_model(args.model_run_dir)
    summary    = load_run_summary(args.model_run_dir)

    K   = int(model_data["K"])
    psi = model_data["psi"].astype(np.float32)   # (K × J_model)
    pi  = model_data["pi"].astype(np.float32)
    alpha_pi = float(model_data["alpha_pi"])
    dir_conc = float(model_data["dir_conc"])
    junc_specific_prior = bool(model_data.get("junc_specific_prior", True))

    print(f"  K={K}, psi shape={psi.shape}")
    print(f"  param_id={summary['param_id']}, waypoints={summary['waypoints_use']}, "
          f"prior={junc_specific_prior}, batch={summary['batch_size']}")

    # Degenerate model check
    if psi.max() < 1e-6:
        print("ERROR: psi is all-zero (training collapse). Exiting.")
        sys.exit(1)

    # ── Confirm junction alignment ─────────────────────────────────────────────
    J_model = psi.shape[1]
    J_easysci = easysci.n_vars
    if J_easysci != J_model:
        print(f"ERROR: EasySci has {J_easysci} junctions, model psi has {J_model} columns.")
        print("Run 01_prep_easysci_for_model.py with the same train_h5ad used for training.")
        sys.exit(1)
    print(f"  Junction alignment confirmed: {J_model:,} junctions")

    # ── Run LeafletFA with fixed PSI ──────────────────────────────────────────
    print(f"\nInitializing LeafletFA (K={K}, fixed PSI) ...")
    leaflet_model = LeafletFA(
        adata=easysci,
        K=K,
        fixed_psi=torch.tensor(psi),
        pi_init=torch.tensor(pi),
        alpha_pi_init=torch.tensor(alpha_pi),
        junc_specific_prior=junc_specific_prior,
        waypoints_use=False,           # EasySci is 547 cells — no waypoints needed
        input_conc_prior=float("inf"),
        delta_fixed=torch.tensor(dir_conc),
        num_epochs=args.num_epochs,
        print_epochs=5,
        ELBO_num_particles=10,
        lr=args.lr,
        gamma=args.gamma,
        min_delta=10,
        num_samples=100,
        patience=args.patience,
        output_dir=args.output_dir,
        log_wandb=False,
    )

    print("Extracting sparse tensors from anndata...")
    leaflet_model.from_anndata()

    print("Initializing triton mask...")
    leaflet_model.initialize_triton_mask()

    print("Training (inferring cell factor assignments) ...")
    leaflet_model.train(num_initializations=1)

    print("Extracting results...")
    leaflet_model.get_all_variables()

    # ── Attach results to adata ────────────────────────────────────────────────
    assign_post = leaflet_model.assign_post        # (N × K)
    pi_learned  = leaflet_model.pi                 # (K,)

    easysci.obsm[f"X_leafletFA_K{K}_{run_name}"] = assign_post

    # ── Save results ──────────────────────────────────────────────────────────
    # Convert COO layers to CSR before saving
    for layer_name in list(easysci.layers.keys()):
        if sp.issparse(easysci.layers[layer_name]):
            easysci.layers[layer_name] = easysci.layers[layer_name].tocsr()
    if sp.issparse(easysci.X):
        easysci.X = easysci.X.tocsr()

    h5ad_path = os.path.join(args.output_dir, f"easysci_leafletfa_{run_name}.h5ad")
    easysci.write_h5ad(h5ad_path, compression="gzip")
    print(f"Saved anndata → {h5ad_path}")

    # Save assign_post as CSV for downstream use (factor activities per metacell)
    assign_df = pd.DataFrame(
        assign_post,
        index=easysci.obs_names,
        columns=[f"factor_{i}" for i in range(K)]
    )
    assign_df = pd.concat([easysci.obs[["Main_cluster_name", "broad_cell_type", "Type"]],
                            assign_df], axis=1)
    csv_path = os.path.join(args.output_dir, f"easysci_leafletfa_{run_name}_assign_post.csv")
    assign_df.to_csv(csv_path)
    print(f"Saved assign_post → {csv_path}")

    # Save PI
    pi_df = pd.DataFrame({"factor": [f"factor_{i}" for i in range(K)],
                           "pi": pi_learned})
    pi_path = os.path.join(args.output_dir, f"easysci_leafletfa_{run_name}_pi.csv")
    pi_df.to_csv(pi_path, index=False)
    print(f"Saved pi → {pi_path}")

    print(f"\nDone. Run: {run_name}, K={K}")


if __name__ == "__main__":
    main()
