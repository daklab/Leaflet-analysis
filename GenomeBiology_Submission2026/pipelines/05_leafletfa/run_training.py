#!/usr/bin/env python
"""
LeafletFA Model Training Script with Multi-pass Mini-batch Training
Optimized for GPU with large datasets.

Uses the installed leafletfa package at /gpfs/commons/home/kisaev/LeafletFA
(pip install -e /gpfs/commons/home/kisaev/LeafletFA).

Called by 02_submit_jobs.py as:
    python run_training.py <param_id> <model_dir> <anndata.h5ad>

param_id maps to a row in model_dir/parameter_combinations.json.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import json
import pickle
import lzma
import wandb
import gzip
import gc
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
import pyro
from pyro.infer.autoguide import AutoGuideList, AutoDiagonalNormal
from pyro import poutine
from pyro.infer.autoguide.initialization import init_to_value

import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
# torch.set_default_tensor_type is deprecated in torch 2.x and is no longer needed: LeafletFA now
# builds all of its priors and init tensors on its own device, so the guide params land on the GPU
# without a global default. (Previously they were created on the CPU -- a 23x slowdown.)

# Seed is set from the params below (after they are loaded) so that the 3 random-init restarts the
# Methods promises ("fit three times, keep the best ELBO") are ACTUALLY different runs. It used to be
# hardcoded to 0, so extra seeds would have produced identical fits.

sns.set_theme()
sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Use installed leafletfa package (pip install -e /gpfs/commons/home/kisaev/LeafletFA)
from leafletfa import LeafletFA
import leafletfa.waypoints as wayp
import leafletfa.utils as utils


def _to_np(x):
    """Model outputs can be CUDA tensors. np.asarray() on one raises

        TypeError: can't convert cuda:0 device type tensor to numpy

    which is exactly how three atlas tasks died AFTER 3.5 h of successful training, at the final
    sanity check, before anything was written to disk. psis_loc/psis_scale come straight off the
    guide and are on the GPU; psi/assign_post/pi are already numpy by this point because they are
    rebuilt from the chunk-to-chunk globals. Never call np.asarray() on a model attribute directly.
    """
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# Get arguments from command line
param_id = int(sys.argv[1])
base_output_dir = sys.argv[2]
ATSE_anndata_file = sys.argv[3]

print(f"Loading parameter set {param_id}...")
print(f"Base output directory: {base_output_dir}")
print(f"Anndata file: {ATSE_anndata_file}")

param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "r") as f:
    param_list = json.load(f)
params = param_list[param_id]

params["input_conc"] = None if params["input_conc"] is None else torch.tensor(np.inf)

# Per-run seed: this is what makes the 3 random-init restarts genuinely distinct runs.
SEED = int(params.get("seed", 0))
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"seed = {SEED}   arm = {params.get('arm', '?')}")

output_dir = os.path.join(base_output_dir, f"run_{param_id}")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved in {output_dir}")

today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Starting run at: {today}")
today_date = datetime.datetime.now().strftime("%Y-%m-%d")

wandb.init(
    project=f"LeafletFA-MouseFoundation_{today_date}",
    config=params,
    name=f"run_{param_id}",
    dir=output_dir,
    group="MouseFoundation",
    notes=f"Parameter set {param_id}, K={params['K']}, waypoints={params['waypoints_use']}, multi-pass minibatch"
)

wandb.config.update({
    "param_id": param_id,
    "data_source": "MouseFoundation",
    "anndata_file": ATSE_anndata_file,
})

print(f"Loading Anndata file: {ATSE_anndata_file}")
adata = ad.read_h5ad(ATSE_anndata_file)
print(f"Anndata file loaded successfully.")
print(adata.obs.head())
print(f"Anndata object contains {adata.n_obs} cells and {adata.n_vars} junctions.")

# =============================================================================
# Filter junctions by max junction count per ATSE (if specified)
# =============================================================================

MAX_JUNCTIONS = params.get("max_junctions", 3)

if "num_junctions" in adata.var.columns and MAX_JUNCTIONS is not None:
    print(f"\nFiltering to ATSEs with <= {MAX_JUNCTIONS} junctions...")
    original_shape = adata.shape
    adata = adata[:, adata.var["num_junctions"] <= MAX_JUNCTIONS].copy()
    adata.var["junction_id_index"] = np.arange(adata.shape[1])
    print(f"  {original_shape} → {adata.shape} ({adata.shape[1]/original_shape[1]*100:.1f}% retained)")
    wandb.log({
        "original_n_junctions": original_shape[1],
        "filtered_n_junctions": adata.shape[1],
        "max_junctions_threshold": MAX_JUNCTIONS,
        "junction_retention_rate": adata.shape[1]/original_shape[1],
    })
else:
    print("Skipping junction count filter (num_junctions column not found or max_junctions not set)")

wandb.log({
    "dataset_cells": adata.shape[0],
    "dataset_junctions": adata.shape[1],
    "initial_learning_rate": params["lr"],
    "gamma_decay": params["gamma"],
    "initial_K": params["K"],
    "delta_fixed": params["delta_fixed"],
})

# =============================================================================
# Multi-pass Mini-batch Training Configuration
# =============================================================================

BATCH_SIZE = params.get("batch_size", 5000)
NUM_PASSES = params.get("num_passes", 10)
NUM_EPOCHS_FIRST = params.get("num_epochs_first", 50)
NUM_EPOCHS_LATER = params.get("num_epochs_later", 20)

n_cells = adata.n_obs
n_junctions = adata.n_vars
n_batches_per_pass = int(np.ceil(n_cells / BATCH_SIZE))
total_batches = n_batches_per_pass * NUM_PASSES

print("\n" + "="*60)
print("MULTI-PASS MINI-BATCH TRAINING MODE")
print(f"  Device: {device}")
print(f"  Total cells: {n_cells:,}")
print(f"  Total junctions: {n_junctions:,}")
print(f"  Batch size: {BATCH_SIZE:,}")
print(f"  Batches per pass: {n_batches_per_pass}")
print(f"  Number of passes: {NUM_PASSES}")
print(f"  Total batches: {total_batches}")
print(f"  Waypoints init: {params.get('waypoints_use', False)}")
print("="*60)

# =============================================================================
# Initialize storage for global parameters
# =============================================================================

global_psi = None
global_psi_loc = None
global_psi_scale = None
global_pi = None
global_alpha_pi = None
global_dir_conc = params.get("delta_fixed", None)
global_bb_conc = None
global_carry = {}   # every other global, carried chunk-to-chunk (see capture below)
_GLOBAL_FIRST = {}  # first-pass value of each global, for drift tracking
_GLOBAL_PREV  = {}  # previous-pass value of each global
global_a = None
global_b = None
global_a_shape = None
global_a_rate = None
global_b_shape = None
global_b_rate = None

all_cell_assignments = np.zeros((n_cells, params["K"]))
cell_assignment_counts = np.zeros(n_cells)

all_batch_results = []
pass_results = []
pi_evolution = []

# =============================================================================
# Pre-compute waypoints on FULL dataset (if enabled)
# Must happen before the training loop so every batch gets the same init.
# =============================================================================

global_psi_wp = None
global_phi_wp = None
waypoint_indices_global = None
X_pca_full = None
psi_init_ref = None   # waypoint PSI before any SVI, used for delta metrics
psi_prev_pass = None  # PSI at end of previous pass, used for pass-to-pass convergence

if params.get("waypoints_use", False):
    print("\n*** Pre-computing waypoints on full dataset ***")

    _K    = params["K"]
    _N_WP = params.get("n_waypoints", _K)

    # PC count is FIXED at 10, independent of K, per the Methods ("we extracted ten PCs"). It used to
    # be min(K, 50), so the waypoint space silently changed dimensionality with K (K=5 -> 5 PCs,
    # K=50 -> 50) and only K=10 accidentally matched the paper. A fixed budget also keeps the
    # waypoint geometry comparable across the K sweep, which is the point of sweeping K.
    _N_PCA = min(params.get("n_pca_components", 10), n_cells - 1, n_junctions - 1)
    _N_DIM = _N_PCA

    # 50 nearest neighbours per waypoint, per the Methods. Was 30.
    _METACELL_SIZE = min(params.get("metacell_size", 50), n_cells // _N_WP)

    if "junc_ratio_centered" in adata.layers:
        _data_pca = adata.layers["junc_ratio_centered"]
    elif "junc_ratio" in adata.layers:
        _data_pca = adata.layers["junc_ratio"]
    else:
        _data_pca = adata.X

    print(f"  PCA on full dataset ({n_cells} cells, {_N_PCA} components)...")
    _svd = TruncatedSVD(n_components=_N_PCA, random_state=42)
    # TruncatedSVD.fit_transform ALREADY returns U * Sigma. The old code multiplied by
    # singular_values_ a second time, giving U * Sigma^2 -- so max-min sampling and the nearest-
    # neighbour assignment ran Euclidean distances in a space where each PC was weighted by sigma^2
    # instead of sigma. That collapses waypoint selection toward PC1 (on a toy check the PC1:PC5
    # weighting went from 3.6x to 12.7x) -- the exact opposite of "chosen to capture major subspaces
    # of variation".
    X_pca_full = _svd.fit_transform(_data_pca)

    # Unsupervised max-min sampling over the PC space of the junction usage ratios.
    # Deliberately NOT stratified by cell type: the paper claims LeafletFA recovers splicing
    # programs "without prior cell-type labels", so labels must not enter through the
    # initialization of the very model whose programs are then shown to recover cell types.
    print(f"  Finding {_N_WP} waypoints (unsupervised global max-min, no cell-type labels)...")
    waypoint_indices_global = wayp.max_min_sampling(X_pca_full, _N_WP, num_components=_N_DIM, seed=42)
    _metacell_dict          = wayp.assign_nearest_cells(waypoint_indices_global, X_pca_full, num_nearest=_METACELL_SIZE)

    _rho = adata.layers["junc_ratio"] if "junc_ratio" in adata.layers else adata.X

    # Coverage mask: 1 where the junction's ATSE HAS READS, so an observed PSI of 0.0 counts as data
    # rather than being silently dropped as "missing". Without it the psi init drops 58.6% of stored
    # values (the exact zeros) and is biased ~2.4x high (mean 0.735 vs the true 0.305); ~52% of the
    # remaining values are exactly 1.0, which maps to a saturated logit (+15.94, gradient 1.2e-7) and
    # freezes about half the psi matrix for the entire run.
    if "psi_mask" in adata.layers:
        _mask = adata.layers["psi_mask"]
    else:
        _mask = adata.layers["cell_by_cluster_matrix"] > 0

    _psi_inits, _phi_inits = wayp.generate_initializations(
        _rho, {_N_WP: waypoint_indices_global}, {_N_WP: _metacell_dict},
        mask=_mask, epsilon=0.001
    )

    global_psi_wp = _psi_inits[0] if not isinstance(_psi_inits[0], torch.Tensor) else _psi_inits[0].cpu().numpy()
    global_phi_wp = _phi_inits[0] if not isinstance(_phi_inits[0], torch.Tensor) else _phi_inits[0].cpu().numpy()
    print(f"  PSI init: {global_psi_wp.shape},  PHI init: {global_phi_wp.shape}")
    # Store (K, J) reference for delta metrics — never updated after this point
    psi_init_ref = global_psi_wp.T.copy()  # (K, J)

    # ── Waypoint PCA plot (upload to wandb + save locally) ──────────────────
    try:
        _ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                        if c in adata.obs.columns), None)
        _ct_labels = adata.obs[_ct_col].astype(str).tolist() if _ct_col else ["unknown"] * n_cells
        _unique_ct = sorted(set(_ct_labels))
        _cmap = plt.colormaps.get_cmap("tab20" if len(_unique_ct) > 10 else "tab10")
        fig, ax = plt.subplots(figsize=(10, 8))
        for _i, _ct in enumerate(_unique_ct):
            _mask = np.array(_ct_labels) == _ct
            ax.scatter(X_pca_full[_mask, 0], X_pca_full[_mask, 1],
                       s=4, alpha=0.4, color=_cmap(_i / max(len(_unique_ct) - 1, 1)), label=_ct)
        ax.scatter(X_pca_full[waypoint_indices_global, 0], X_pca_full[waypoint_indices_global, 1],
                   s=150, marker="*", color="red", edgecolors="black", linewidths=0.5, zorder=5, label="Waypoints")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title(f"Waypoint PCA | K={_K}, {n_cells:,} cells")
        ax.legend(markerscale=2, fontsize=6, loc="best", ncol=2)
        fig.tight_layout()
        _wp_plot = os.path.join(output_dir, f"waypoint_pca_K{_K}.png")
        fig.savefig(_wp_plot, dpi=150)
        plt.close(fig)
        wandb.log({"waypoint_pca": wandb.Image(_wp_plot)})
        print(f"  ✓ Waypoint PCA plot → {_wp_plot}")
    except Exception as _e:
        print(f"  Warning: waypoint PCA plot failed: {_e}")

# =============================================================================
# Multi-pass Training Loop
# =============================================================================

batch_counter = 0

for pass_idx in range(NUM_PASSES):
    print(f"\n{'='*60}")
    print(f"PASS {pass_idx + 1}/{NUM_PASSES}")
    print(f"{'='*60}")

    cell_indices = np.random.permutation(n_cells)
    pass_elbo = []

    for batch_in_pass in range(n_batches_per_pass):
        batch_counter += 1

        print(f"\n{'-'*50}")
        print(f"Pass {pass_idx + 1}, Batch {batch_in_pass + 1}/{n_batches_per_pass} (Overall: {batch_counter}/{total_batches})")
        print(f"{'-'*50}")

        start_idx = batch_in_pass * BATCH_SIZE
        end_idx = min((batch_in_pass + 1) * BATCH_SIZE, n_cells)
        batch_cell_indices = cell_indices[start_idx:end_idx]
        batch_cell_indices = np.sort(batch_cell_indices)

        print(f"Processing {len(batch_cell_indices)} cells")
        batch_adata = adata[batch_cell_indices, :].copy()

        # =====================================================================
        # First batch: waypoint initialization (if enabled)
        # =====================================================================

        if batch_counter == 1 and params.get("waypoints_use", False):
            # global_psi_wp / global_phi_wp were computed on the FULL dataset before
            # the training loop — just slice the relevant rows for this batch.
            batch_adata.varm[f'psi_init_{params["K"]}_waypoints'] = global_psi_wp
            batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = global_phi_wp[batch_cell_indices]
            print("✓ Waypoint init applied from pre-computed full-dataset waypoints")

        # =====================================================================
        # Initialize LeafletFA model for this batch
        # =====================================================================

        # Rescale the LOCAL terms (likelihood + the phi prior) by N_total/batch_size, leaving the
        # global KL (psi, pi, a, b) at weight 1. Without this each chunk optimizes the ELBO of a
        # BATCH_SIZE-cell dataset -- the full global prior gets charged against one batch's
        # likelihood, over-weighting it by N/B and shrinking the K factors toward each other.
        batch_obs_scale = n_cells / len(batch_cell_indices)

        if batch_counter == 1:
            print("Initializing model from scratch (first batch ever)...")

            leaflet_model = LeafletFA(
                adata=batch_adata,
                K=params["K"],
                junc_specific_prior=params["junc_specific_prior"],
                waypoints_use=params.get("waypoints_use", False),
                input_conc_prior=params.get("input_conc", None),
                delta_fixed=params.get("delta_fixed", None),
                obs_scale=batch_obs_scale,
                num_epochs=NUM_EPOCHS_FIRST,
                print_epochs=5,
                ELBO_num_particles=params["ELBO_num_particles"],
                lr=params["lr"],
                gamma=params["gamma"],
                min_delta=params["min_delta"],
                min_delta_rel=params.get("min_delta_rel", 1e-5),
                num_samples=params["num_samples"],
                patience=params["patience"],
                output_dir=output_dir,
                log_wandb=True,
                enable_cpu_optimization=False
            )

        else:
            print("Initializing model with previous batch parameters...")

            batch_adata.varm[f'psi_init_{params["K"]}_waypoints'] = global_psi.T

            if all_cell_assignments[batch_cell_indices].sum() > 0:
                # Cells seen in a previous pass — use their learned phi
                batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = all_cell_assignments[batch_cell_indices]
            elif global_phi_wp is not None:
                # Pass 1, first time seeing these cells — use waypoint phi (better than random)
                batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = global_phi_wp[batch_cell_indices]
            else:
                batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = np.random.dirichlet(
                    np.ones(params["K"]), size=len(batch_cell_indices)
                )

            leaflet_model = LeafletFA(
                adata=batch_adata,
                K=params["K"],
                junc_specific_prior=params["junc_specific_prior"],
                waypoints_use=True,
                input_conc_prior=params.get("input_conc", None),
                # Honour the CONFIGURED delta only. Previously the learned delta from batch 1 was
                # fed back in here as `delta_fixed`, which stopped it being sampled at all — so
                # delta was estimated from the first 5,000 cells and then frozen for the remaining
                # ~185 chunks. That was a side-effect of get_pi() overwriting self.dir_conc, not a
                # modelling choice. With obs_scale set, delta's per-chunk posterior is already an
                # unbiased estimate of its full-data posterior, so just let it be learned.
                delta_fixed=params.get("delta_fixed", None),
                pi_init=torch.tensor(global_pi, dtype=torch.float32) if global_pi is not None else None,
                alpha_pi_init=torch.tensor(global_alpha_pi, dtype=torch.float32) if global_alpha_pi is not None else None,
                global_init=global_carry or None,   # bb_conc, a, b, a_shape/a_rate/b_shape/b_rate
                obs_scale=batch_obs_scale,
                num_epochs=NUM_EPOCHS_LATER,
                print_epochs=5,
                ELBO_num_particles=params["ELBO_num_particles"],
                lr=params["lr"] * (0.95 ** pass_idx),
                gamma=params["gamma"],
                min_delta=params["min_delta"],
                min_delta_rel=params.get("min_delta_rel", 1e-5),
                num_samples=params["num_samples"],
                patience=params["patience"],
                output_dir=output_dir,
                log_wandb=True,
                enable_cpu_optimization=False
            )

        print("Extracting sparse tensors...")
        leaflet_model.from_anndata()

        # (initialize_triton_mask() used to be required on CUDA. It built a dense C x J mask that
        # the Triton kernel never actually read, and is now a no-op — LeafletFA gathers rho only at
        # the observed positions on both devices.)

        print(f"Training batch {batch_counter}...")
        # NB: do NOT pass psi_scale_init here. Warm-starting q(psi)'s WIDTH across chunks was tried
        # and it makes the posterior WORSE, not better -- measured against a full-batch reference:
        #     full-batch (the truth)      : mean q(psi) width 0.173
        #     chunked, scale reset        : 0.265   (1.5x too wide)
        #     chunked, scale carried fwd  : 0.411   (2.4x too wide)  <- compounds the widening
        # Each chunk's optimisation widens from its narrow start, so carrying that width forward
        # compounds rather than accumulates. The width is ~1.5x too wide either way, which makes the
        # DS/FDR CONSERVATIVE (fewer calls), not anti-conservative.
        #
        # The real fix is architectural: ONE persistent guide + optimizer across all minibatches with
        # pyro.plate(subsample=...), so q(psi) is a single set of params refined by every batch. The
        # per-chunk teardown is what prevents q(psi) accumulating evidence at all. See METHODS.md 5.1.
        leaflet_model.train(num_initializations=1, seeds=[SEED + batch_counter])

        print("Extracting learned parameters...")
        leaflet_model.get_all_variables()

        # Refuse to carry a collapsed model forward: if the K programs are identical, warm-starting
        # the next chunk from them just propagates the collapse.
        _h = getattr(leaflet_model, "health", None)
        if _h is not None:
            wandb.log({"batch": batch_counter,
                       "psi_contrast": _h["psi_contrast"],
                       "phi_spread": _h["phi_spread"],
                       "phi_perplexity": _h["phi_perplexity"],
                       "phi_perplexity_prior": _h["phi_perplexity_prior"],
                       "health_ok": int(_h["healthy"])})
            if not _h["healthy"]:
                print(f"  !! HEALTH CHECK FAILED on batch {batch_counter}: {_h['problems']}")

        if hasattr(leaflet_model, 'psi'):
            global_psi = leaflet_model.psi if isinstance(leaflet_model.psi, np.ndarray) else leaflet_model.psi.cpu().numpy()
        if hasattr(leaflet_model, 'pi'):
            global_pi = leaflet_model.pi if isinstance(leaflet_model.pi, np.ndarray) else leaflet_model.pi.cpu().numpy()
        if hasattr(leaflet_model, 'alpha_pi'):
            global_alpha_pi = leaflet_model.alpha_pi if isinstance(leaflet_model.alpha_pi, np.ndarray) else leaflet_model.alpha_pi.cpu().numpy()
        # Read the POSTERIOR delta from dir_conc_post. `leaflet_model.dir_conc` is the *input*
        # (None means "learn it"); get_pi() used to overwrite it with the learned value, which is
        # how delta silently ended up frozen at the first chunk's estimate for the whole run.
        _dc = getattr(leaflet_model, 'dir_conc_post', None)
        if _dc is not None:
            global_dir_conc = _dc.item() if isinstance(_dc, torch.Tensor) else float(_dc)
        if hasattr(leaflet_model, 'bb_conc'):
            global_bb_conc = leaflet_model.bb_conc
        # CARRY EVERY GLOBAL chunk-to-chunk. Previously only psi/phi/pi/alpha_pi were carried, so
        # bb_conc, a, b and the four Beta hyperparameters were re-initialised from their PRIORS on
        # every one of ~240 chunks and given only num_epochs_later steps on ONE batch to move --
        # they never accumulated evidence over the full dataset. The tell on the mouse atlas: saved
        # bb_conc was bit-identical across waypoint/random and junction/global-prior arms whenever
        # the seed matched (0.337/2.053/1.373 for seeds 0/1/2 at K=10) while psi genuinely differed,
        # i.e. it never moved off its seed-determined init. That also means bb_conc could NOT be
        # used as a convergence diagnostic. Seeded via LeafletFA(global_init=...) on the next chunk.
        # CARRY ONLY bb_conc -- measured. Carrying the per-junction a/b compounds instead of
        # accumulating: on a 6,000-cell atlas subset it drove the psi prior to Beta(204,6) (mean
        # 0.97), pushed 42% of psi above 0.9, halved psi_contrast (0.115 -> 0.059) and collapsed phi
        # commitment (0.47 -> 0.14). a/b/dir_conc remain LAST-CHUNK estimates; fix that post hoc with
        # psi frozen, not by carrying.
        for _g in ("bb_conc",):
            _v = getattr(leaflet_model, "dir_conc_post", None) if _g == "dir_conc" \
                 else getattr(leaflet_model, _g, None)
            if _v is None or isinstance(_v, str):      # "infinity" when conc is FIXED, not latent
                continue
            global_carry[_g] = (_v.detach().cpu().numpy()
                                if isinstance(_v, torch.Tensor) else np.asarray(_v))
        if hasattr(leaflet_model, 'psis_loc') and leaflet_model.psis_loc is not None:
            global_psi_loc = leaflet_model.psis_loc
            # This is now FED BACK into the next chunk (see train(psi_scale_init=...) above), not
            # just stored for the final save. Previously it was extracted and silently discarded.
            global_psi_scale = leaflet_model.psis_scale
        if hasattr(leaflet_model, 'a'):
            global_a = leaflet_model.a
            global_b = leaflet_model.b
            global_a_shape = leaflet_model.a_shape
            global_a_rate = leaflet_model.a_rate
            global_b_shape = leaflet_model.b_shape
            global_b_rate = leaflet_model.b_rate

        if hasattr(leaflet_model, 'assign_post'):
            _ap = leaflet_model.assign_post
            if hasattr(_ap, 'detach'):
                _ap = _ap.detach().cpu().numpy()
            elif not isinstance(_ap, np.ndarray):
                _ap = np.array(_ap)
            all_cell_assignments[batch_cell_indices, :] = _ap
            cell_assignment_counts[batch_cell_indices] += 1

        batch_elbo = leaflet_model.best_elbo if hasattr(leaflet_model, 'best_elbo') else None

        # Divide by n_cells, NOT by len(batch).
        # obs_scale = n_cells/len(batch) is already applied to the likelihood and the phi prior, so
        # the loss is ALREADY on the full-dataset scale: loss ~= n_cells * per_cell_ELBO + global_KL.
        # Dividing by len(batch) therefore inflated it by n_cells/len(batch) (16.2x here) -- and the
        # LAST batch of each pass is short (1,634 cells, not 8,192), so its obs_scale is 81 rather
        # than 16 and dividing by its own size made it look ~5x larger than every other batch. That
        # corrupted pass_avg_elbo, pass_min/max_elbo and every wandb curve.
        batch_elbo_per_cell = batch_elbo / n_cells if batch_elbo is not None else None
        pass_elbo.append(batch_elbo_per_cell)

        # The health check is the ONLY thing that detects a symmetric collapse: a fit whose loss
        # curve looks fine and whose "dead programs" count is 0 while all K programs are identical.
        _health = getattr(leaflet_model, "health", None)

        all_batch_results.append({
            'pass': pass_idx + 1,
            'batch_in_pass': batch_in_pass + 1,
            'overall_batch': batch_counter,
            'batch_size': len(batch_cell_indices),
            'elbo': batch_elbo,
            'elbo_per_cell': batch_elbo_per_cell,
            'learning_rate': params["lr"] * (0.95 ** pass_idx)
        })

        print(f"Batch {batch_counter} complete. ELBO: {batch_elbo:.4e}  ({batch_elbo_per_cell:.4e} per cell)")

        wandb.log({
            "batch": batch_counter,
            "pass": pass_idx + 1,
            "batch_elbo": batch_elbo,
            "batch_elbo_per_cell": batch_elbo_per_cell,
            "learning_rate": params["lr"] * (0.95 ** pass_idx)
        })

        del batch_adata, leaflet_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_pass_elbo = np.mean([e for e in pass_elbo if e is not None])
    pass_results.append({
        'pass': pass_idx + 1,
        'avg_elbo': avg_pass_elbo,
        'min_elbo': np.min([e for e in pass_elbo if e is not None]),
        'max_elbo': np.max([e for e in pass_elbo if e is not None])
    })

    if global_pi is not None:
        pi_evolution.append({
            'pass': pass_idx + 1,
            'pi': global_pi.copy(),
            'avg_elbo': avg_pass_elbo
        })
        print(f"Pi after pass {pass_idx + 1}: {global_pi[:5]}... (first 5)")

    print(f"\nPass {pass_idx + 1} complete. Average ELBO per cell: {avg_pass_elbo:.4e}")
    print(f"Cells seen: min={cell_assignment_counts.min()}, max={cell_assignment_counts.max()}, mean={cell_assignment_counts.mean():.1f}")

    wandb.log({
        "pass": pass_idx + 1,
        "pass_avg_elbo_per_cell": avg_pass_elbo,
        "pass_min_elbo_per_cell": pass_results[-1]['min_elbo'],
        "pass_max_elbo_per_cell": pass_results[-1]['max_elbo'],
    })

    # PSI delta metrics vs waypoint init and vs previous pass
    if global_psi is not None:
        psi_f = global_psi  # (K, J)

        psi_log = {"pass": pass_idx + 1}
        # ── GLOBAL DRIFT TRACKING ────────────────────────────────────────────────────────────
        # Catches the failure mode that hid bb_conc for months: a global that is never carried
        # across chunks silently sits at its seed-determined initialisation while everything else
        # learns. Invisible in the loss and in psi_contrast.
        #
        # NB read from the global_* accumulators (captured at the end of each chunk), NOT from
        # leaflet_model -- that name is not in scope here, which is exactly how the first version of
        # this block failed silently inside its own try/except.
        try:
            # EVERY latent, in one table -- including psi and phi, the two that matter most.
            _now = {"psi": global_psi, "phi": all_cell_assignments,
                    "pi": global_pi, "alpha_pi": global_alpha_pi,
                    "bb_conc": global_bb_conc, "dir_conc": global_dir_conc,
                    "a": global_a, "b": global_b,
                    "a_shape": global_a_shape, "a_rate": global_a_rate,
                    "b_shape": global_b_shape, "b_rate": global_b_rate}
            for _g, _v in _now.items():
                if _v is None or isinstance(_v, str):
                    continue
                _v = _v.detach().cpu().numpy() if isinstance(_v, torch.Tensor) else np.asarray(_v)
                _v = np.asarray(_v, dtype=float).ravel()
                if _v.size == 0 or not np.isfinite(_v).all():
                    continue
                _scale = float(np.max(np.abs(_v))) + 1e-12
                _first = _GLOBAL_FIRST.setdefault(_g, _v.copy())
                _prev  = _GLOBAL_PREV.get(_g, _v)
                d_first = float(np.max(np.abs(_v - _first)) / _scale)
                d_prev  = float(np.max(np.abs(_v - _prev))  / _scale)
                psi_log[f"drift_from_first_{_g}"] = d_first
                psi_log[f"drift_from_prev_{_g}"]  = d_prev
                psi_log[f"value_{_g}"] = float(_v.mean())
                _GLOBAL_PREV[_g] = _v.copy()
                _flag = "  <-- STUCK AT INIT?" if (pass_idx >= 2 and d_first < 1e-8) else ""
                _extra = ""
                if _g == "psi":     # the DS effect size lives here
                    _p = np.asarray(global_psi, dtype=float)
                    _extra = f"  psi_contrast={_p.std(axis=0).mean():.4f}"
                elif _g == "phi":   # do cells still commit, or has phi gone uniform?
                    _f = np.asarray(all_cell_assignments, dtype=float)
                    _f = _f / (_f.sum(1, keepdims=True) + 1e-12)
                    _extra = f"  per-cell max={np.median(_f.max(1)):.3f} (uniform={1.0/_f.shape[1]:.3f})"
                print(f"  [global] {_g:<9} mean={_v.mean():>12.4f}  "
                      f"drift_vs_pass1={d_first:.3e}  vs_prev={d_prev:.3e}{_extra}{_flag}", flush=True)
        except Exception as _e:
            print(f"  [global] drift tracking failed: {type(_e).__name__}: {_e}", flush=True)



        # --- program separation (psi contrast) ---
        # Spread across the K programs at each junction, averaged over junctions. This is the
        # single number that tells you whether the factors ever pulled apart: all K programs
        # start at nearly the same psi, so a run that never escapes that symmetric point keeps
        # a contrast of ~0.003 forever. A run that separates lands around ~0.1. The dead-program
        # and never-dominant counts do NOT catch this -- they report everything as healthy while
        # every program is a copy of the others.
        psi_log["psi_contrast"] = float(psi_f.std(axis=0).mean())
        print(f"  PSI contrast (program separation): {psi_log['psi_contrast']:.4f}"
              f"{'   <-- SUSPICIOUS: programs may not have separated' if psi_log['psi_contrast'] < 0.01 else ''}")

        # --- vs original waypoint init ---
        if psi_init_ref is not None:
            delta = np.abs(psi_f - psi_init_ref)                         # (K, J)
            mac_per_junction = delta.mean(axis=0)                        # (J,)

            # 1. Overall mean absolute change
            psi_log["psi_mac_vs_init"] = float(delta.mean())

            # 2. Per-junction MAC summary (describes the distribution)
            psi_log["psi_mac_per_junc_median"] = float(np.median(mac_per_junction))
            psi_log["psi_mac_per_junc_p90"]    = float(np.percentile(mac_per_junction, 90))
            psi_log["psi_mac_per_junc_hist"]   = wandb.Histogram(mac_per_junction)

            # 3. Per-factor Pearson r (mean and min across factors)
            r_per_factor = [pearsonr(psi_f[k, :], psi_init_ref[k, :])[0]
                            for k in range(psi_f.shape[0])]
            psi_log["psi_pearson_r_mean"] = float(np.mean(r_per_factor))
            psi_log["psi_pearson_r_min"]  = float(np.min(r_per_factor))

            # 4. Fraction of factor-junction pairs with |delta| > 0.1 PSI units
            psi_log["psi_frac_delta_gt_0.1"] = float((delta > 0.1).mean())

            print(f"  PSI delta vs init — MAC: {psi_log['psi_mac_vs_init']:.4f}, "
                  f"mean r: {psi_log['psi_pearson_r_mean']:.3f}, "
                  f"frac>0.1: {psi_log['psi_frac_delta_gt_0.1']:.3f}")

        # --- pass-to-pass convergence ---
        if psi_prev_pass is not None:
            psi_log["psi_mac_pass_to_pass"] = float(np.abs(psi_f - psi_prev_pass).mean())
            print(f"  PSI pass-to-pass MAC: {psi_log['psi_mac_pass_to_pass']:.4f}")

        wandb.log(psi_log)
        psi_prev_pass = psi_f.copy()

    # PHI snapshot after all cells have been seen once (end of pass 1)
    if pass_idx == 0 and all_cell_assignments.sum() > 0:
        try:
            _phi_p1 = all_cell_assignments.astype(np.float32)
            _adata_p1 = ad.AnnData(np.zeros((n_cells, 1)))
            _adata_p1.obsm['X_phi'] = _phi_p1
            sc.pp.neighbors(_adata_p1, use_rep='X_phi', n_neighbors=30, random_state=42)
            sc.tl.umap(_adata_p1, random_state=42)
            _umap_p1 = _adata_p1.obsm['X_umap']

            _ct_col_p1 = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                               if c in adata.obs.columns), None)
            _ct_labels_p1 = adata.obs[_ct_col_p1].astype(str).tolist() if _ct_col_p1 else ["unknown"] * n_cells
            _unique_p1 = sorted(set(_ct_labels_p1))
            _cmap_p1 = plt.colormaps.get_cmap("tab20" if len(_unique_p1) > 10 else "tab10")

            fig, ax = plt.subplots(figsize=(10, 8))
            for _i, _ct in enumerate(_unique_p1):
                _mask = np.array(_ct_labels_p1) == _ct
                ax.scatter(_umap_p1[_mask, 0], _umap_p1[_mask, 1],
                           s=3, alpha=0.4, color=_cmap_p1(_i / max(len(_unique_p1) - 1, 1)), label=_ct)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_title(f"PHI UMAP after Pass 1 | K={params['K']}, {n_cells:,} cells")
            ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
            fig.tight_layout()
            _p1_plot = os.path.join(output_dir, f"phi_umap_pass1_K{params['K']}.png")
            fig.savefig(_p1_plot, dpi=150)
            plt.close(fig)
            wandb.log({"phi_umap_pass1": wandb.Image(_p1_plot)})
            print(f"✓ PHI UMAP after pass 1 → {_p1_plot}")
        except Exception as _e:
            print(f"Warning: pass-1 phi UMAP failed: {_e}")

# =============================================================================
# Final Model Assembly
# =============================================================================

print("\n" + "="*60)
print("TRAINING COMPLETE - ASSEMBLING FINAL MODEL")
print("="*60)

print(f"Cell coverage: {(cell_assignment_counts > 0).sum()}/{n_cells} cells updated")
print(f"Average updates per cell: {cell_assignment_counts.mean():.1f}")


class FinalModel:
    pass


leaflet_model = FinalModel()
leaflet_model.psi = global_psi
leaflet_model.psi_learned = global_psi
leaflet_model.assign_post = all_cell_assignments
leaflet_model.pi = global_pi
leaflet_model.alpha_pi = global_alpha_pi
leaflet_model.dir_conc = global_dir_conc
leaflet_model.bb_conc = global_bb_conc
leaflet_model.K = params["K"]
leaflet_model.best_elbo = np.mean([r['elbo'] for r in all_batch_results if r['elbo'] is not None])
leaflet_model.junc_specific_prior = params["junc_specific_prior"]
leaflet_model.input_conc_prior = params.get("input_conc", None)
leaflet_model.gamma = params["gamma"]
leaflet_model.ELBO_num_particles = params["ELBO_num_particles"]

if global_psi_loc is not None:
    leaflet_model.psis_loc = global_psi_loc
    leaflet_model.psis_scale = global_psi_scale
if global_a is not None:
    leaflet_model.a = global_a
    leaflet_model.b = global_b
    leaflet_model.a_shape = global_a_shape
    leaflet_model.a_rate = global_a_rate
    leaflet_model.b_shape = global_b_shape
    leaflet_model.b_rate = global_b_rate

print("\nPruning factors...")
original_K = leaflet_model.K
if leaflet_model.pi is not None:
    keep_factors = leaflet_model.pi > 0.01
    if keep_factors.sum() < original_K:
        leaflet_model.K = int(keep_factors.sum())
        # EVERY factor-indexed array must be pruned together, or the saved arrays silently
        # misalign. psi/assign_post/pi used to be pruned while psis_loc and psis_scale were NOT --
        # and all four were pickled, so the saved psis_loc had original_K rows against psi's
        # pruned_K. psis_loc/psis_scale are exactly what the differential-splicing analysis samples
        # from, so anything pairing them with psi was reading the wrong programs.
        leaflet_model.psi = leaflet_model.psi[keep_factors, :]
        leaflet_model.psi_learned = leaflet_model.psi
        leaflet_model.assign_post = leaflet_model.assign_post[:, keep_factors]
        leaflet_model.assign_post = leaflet_model.assign_post / leaflet_model.assign_post.sum(axis=1, keepdims=True)
        leaflet_model.pi = leaflet_model.pi[keep_factors]
        leaflet_model.pi = leaflet_model.pi / leaflet_model.pi.sum()
        for _attr in ("psis_loc", "psis_scale"):          # (K, J) -- factor-indexed
            _v = getattr(leaflet_model, _attr, None)
            if _v is not None:
                setattr(leaflet_model, _attr, _to_np(_v)[keep_factors, :])
        # a, b, a_shape, ... are per-JUNCTION (shape [J]) or scalars: they carry no factor axis and
        # must NOT be pruned.
        print(f"Pruned K from {original_K} to {leaflet_model.K}")
    else:
        print(f"No pruning needed, keeping all {original_K} factors")

new_K = leaflet_model.K

# Sanity: every factor-indexed array must now agree on K.
_shapes = {"psi": leaflet_model.psi.shape[0],
           "assign_post": leaflet_model.assign_post.shape[1],
           "pi": leaflet_model.pi.shape[0]}
for _attr in ("psis_loc", "psis_scale"):
    _v = getattr(leaflet_model, _attr, None)
    if _v is not None:
        _shapes[_attr] = _to_np(_v).shape[0]
assert len(set(_shapes.values())) == 1, f"factor-axis mismatch after pruning: {_shapes}"

# =============================================================================
# Save Results
# =============================================================================

wandb.log({
    "final_K": new_K,
    "original_K": original_K,
    "final_elbo": leaflet_model.best_elbo,
    "num_passes": NUM_PASSES,
    "total_batches": total_batches,
    "cells_coverage": (cell_assignment_counts > 0).sum() / n_cells,
    "avg_updates_per_cell": cell_assignment_counts.mean(),
})

if pi_evolution and len(pi_evolution) > 1:
    pi_changes = [
        np.linalg.norm(pi_evolution[i]['pi'] - pi_evolution[i-1]['pi'])
        for i in range(1, len(pi_evolution))
    ]
    wandb.log({
        "pi_final_stability": pi_changes[-1],
        "pi_avg_change": np.mean(pi_changes),
        "pi_convergence_ratio": pi_changes[-1] / pi_changes[0] if pi_changes[0] > 0 else 1,
    })
    print(f"\nPi stability: final change={pi_changes[-1]:.6f}, avg={np.mean(pi_changes):.6f}")

if hasattr(leaflet_model, 'pi') and leaflet_model.pi is not None:
    PI_df = pd.DataFrame({"PI": leaflet_model.pi, "Factor": np.arange(len(leaflet_model.pi))})
    PI_df = PI_df.sort_values("PI", ascending=False)
    print(f"\nFactor probabilities (top 10):\n{PI_df.head(10)}")
    PI_df.to_csv(os.path.join(output_dir, "factor_assignment_probabilities.csv"), index=False)

pd.DataFrame(all_batch_results).to_csv(os.path.join(output_dir, "batch_training_results.csv"), index=False)
pd.DataFrame(pass_results).to_csv(os.path.join(output_dir, "pass_summary_results.csv"), index=False)

if pi_evolution:
    pi_evolution_data = [
        {'pass': s['pass'], 'factor': k, 'pi_value': s['pi'][k], 'avg_elbo': s['avg_elbo']}
        for s in pi_evolution for k in range(len(s['pi']))
    ]
    pi_df = pd.DataFrame(pi_evolution_data)
    pi_df.to_csv(os.path.join(output_dir, "pi_evolution.csv"), index=False)

    plt.figure(figsize=(12, 8))
    for k in range(min(10, params["K"])):
        fd = pi_df[pi_df['factor'] == k]
        plt.plot(fd['pass'], fd['pi_value'], marker='o', label=f'Factor {k}', linewidth=2)
    plt.xlabel('Pass')
    plt.ylabel('Pi')
    plt.title('Factor Probabilities Across Training Passes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_evolution_plot.png"), dpi=100, bbox_inches='tight')
    plt.close()

    pi_matrix = np.zeros((NUM_PASSES, params["K"]))
    for s in pi_evolution:
        pi_matrix[s['pass']-1, :] = s['pi']
    plt.figure(figsize=(14, 8))
    sns.heatmap(pi_matrix.T, cmap='coolwarm',
                xticklabels=range(1, NUM_PASSES+1),
                yticklabels=range(params["K"]),
                cbar_kws={'label': 'Pi'})
    plt.xlabel('Pass')
    plt.ylabel('Factor')
    plt.title('Pi Heatmap Across Training')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_evolution_heatmap.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Saved pi evolution plots")

# ── Phi PCA plot: compare learned factor space to raw data PCA ────────────
if leaflet_model.assign_post is not None and leaflet_model.assign_post.sum() > 0:
    try:
        from sklearn.decomposition import PCA as _PCA
        _phi = leaflet_model.assign_post   # (n_cells, pruned_K)
        if _phi.shape[1] > 2:
            _pca_phi = _PCA(n_components=2, random_state=42)
            _coords  = _pca_phi.fit_transform(_phi)
            _xlabel  = f"Phi-PC 1 ({_pca_phi.explained_variance_ratio_[0]*100:.1f}%)"
            _ylabel  = f"Phi-PC 2 ({_pca_phi.explained_variance_ratio_[1]*100:.1f}%)"
        else:
            _coords = _phi
            _xlabel, _ylabel = "Factor 1", "Factor 2"

        _ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                        if c in adata.obs.columns), None)
        _ct_labels = adata.obs[_ct_col].astype(str).tolist() if _ct_col else ["unknown"] * n_cells
        _unique_ct = sorted(set(_ct_labels))
        _cmap = plt.colormaps.get_cmap("tab20" if len(_unique_ct) > 10 else "tab10")

        fig, ax = plt.subplots(figsize=(10, 8))
        for _i, _ct in enumerate(_unique_ct):
            _mask = np.array(_ct_labels) == _ct
            ax.scatter(_coords[_mask, 0], _coords[_mask, 1],
                       s=4, alpha=0.4, color=_cmap(_i / max(len(_unique_ct) - 1, 1)), label=_ct)
        if waypoint_indices_global is not None:
            ax.scatter(_coords[waypoint_indices_global, 0], _coords[waypoint_indices_global, 1],
                       s=150, marker="*", color="red", edgecolors="black", linewidths=0.5, zorder=5, label="Waypoints")
        ax.set_xlabel(_xlabel); ax.set_ylabel(_ylabel)
        ax.set_title(f"PCA of learned Φ | K={new_K} (from {original_K}), {n_cells:,} cells")
        ax.legend(markerscale=3, fontsize=6, loc="best", ncol=2)
        fig.tight_layout()
        _phi_plot = os.path.join(output_dir, f"phi_pca_K{new_K}.png")
        fig.savefig(_phi_plot, dpi=150)
        plt.close(fig)
        wandb.log({"phi_pca": wandb.Image(_phi_plot)})
        print(f"✓ Phi PCA plot → {_phi_plot}")
    except Exception as _e:
        print(f"Warning: phi PCA plot failed: {_e}")

# ── Phi UMAP: UMAP of the learned latent space ───────────────────────────────
if leaflet_model.assign_post is not None and leaflet_model.assign_post.sum() > 0:
    try:
        _phi = leaflet_model.assign_post.astype(np.float32)
        _adata_phi = ad.AnnData(np.zeros((n_cells, 1)))
        _adata_phi.obsm['X_phi'] = _phi
        sc.pp.neighbors(_adata_phi, use_rep='X_phi', n_neighbors=30, random_state=42)
        sc.tl.umap(_adata_phi, random_state=42)
        _umap_phi_coords = _adata_phi.obsm['X_umap']

        _ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                        if c in adata.obs.columns), None)
        _ct_labels = adata.obs[_ct_col].astype(str).tolist() if _ct_col else ["unknown"] * n_cells
        _unique_ct = sorted(set(_ct_labels))
        _cmap = plt.colormaps.get_cmap("tab20" if len(_unique_ct) > 10 else "tab10")

        fig, ax = plt.subplots(figsize=(10, 8))
        for _i, _ct in enumerate(_unique_ct):
            _mask = np.array(_ct_labels) == _ct
            ax.scatter(_umap_phi_coords[_mask, 0], _umap_phi_coords[_mask, 1],
                       s=3, alpha=0.4, color=_cmap(_i / max(len(_unique_ct) - 1, 1)), label=_ct)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP of Learned Φ | K={new_K}, {n_cells:,} cells")
        ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        _phi_umap_plot = os.path.join(output_dir, f"phi_umap_K{new_K}.png")
        fig.savefig(_phi_umap_plot, dpi=150)
        plt.close(fig)
        wandb.log({"phi_umap": wandb.Image(_phi_umap_plot)})
        print(f"✓ Phi UMAP → {_phi_umap_plot}")
    except Exception as _e:
        print(f"Warning: phi UMAP failed: {_e}")

# ── Raw data UMAP (splicing PCA-based) ───────────────────────────────────────
# Uses the same X_pca_full computed during waypoint init (n_pca_components = K),
# so the raw UMAP and phi UMAP are directly comparable in the same dimensionality.
try:
    if X_pca_full is None:
        _data_pca_viz = adata.layers["junc_ratio_centered"] if "junc_ratio_centered" in adata.layers \
            else adata.layers.get("junc_ratio", adata.X)
        _n_viz = min(20, n_cells - 1, n_junctions - 1)
        _svd_viz = TruncatedSVD(n_components=_n_viz, random_state=42)
        X_pca_full = _svd_viz.fit_transform(_data_pca_viz)
        print(f"  Computed PCA ({_n_viz} components) for raw UMAP")

    _adata_raw = ad.AnnData(np.zeros((n_cells, 1)))
    _adata_raw.obsm['X_pca'] = X_pca_full.astype(np.float32)
    sc.pp.neighbors(_adata_raw, use_rep='X_pca', n_neighbors=30, random_state=42)
    sc.tl.umap(_adata_raw, random_state=42)
    _umap_raw_coords = _adata_raw.obsm['X_umap']

    _ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                    if c in adata.obs.columns), None)
    _ct_labels = adata.obs[_ct_col].astype(str).tolist() if _ct_col else ["unknown"] * n_cells
    _unique_ct = sorted(set(_ct_labels))
    _cmap = plt.colormaps.get_cmap("tab20" if len(_unique_ct) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(10, 8))
    for _i, _ct in enumerate(_unique_ct):
        _mask = np.array(_ct_labels) == _ct
        ax.scatter(_umap_raw_coords[_mask, 0], _umap_raw_coords[_mask, 1],
                   s=3, alpha=0.4, color=_cmap(_i / max(len(_unique_ct) - 1, 1)), label=_ct)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Raw Splicing Data UMAP (PCA) | {n_cells:,} cells")
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    _raw_umap_plot = os.path.join(output_dir, "raw_data_umap.png")
    fig.savefig(_raw_umap_plot, dpi=150)
    plt.close(fig)
    wandb.log({"raw_data_umap": wandb.Image(_raw_umap_plot)})
    print(f"✓ Raw data UMAP → {_raw_umap_plot}")
except Exception as _e:
    print(f"Warning: raw data UMAP failed: {_e}")

# ── Pi barplot: learned factor weights ───────────────────────────────────────
if leaflet_model.pi is not None:
    try:
        _pi_sorted = np.sort(leaflet_model.pi)[::-1]
        fig, ax = plt.subplots(figsize=(max(6, new_K * 0.8), 4))
        ax.bar(range(new_K), _pi_sorted,
               color=sns.color_palette("husl", new_K))
        ax.set_xlabel("Factor (sorted by weight)")
        ax.set_ylabel("Pi (mixing weight)")
        ax.set_title(f"Learned Factor Weights | K={new_K} (from {original_K})")
        ax.set_xticks(range(new_K))
        ax.set_xticklabels([f"F{i}" for i in range(new_K)], fontsize=8)
        fig.tight_layout()
        _pi_plot = os.path.join(output_dir, f"pi_barplot_K{new_K}.png")
        fig.savefig(_pi_plot, dpi=150)
        plt.close(fig)
        wandb.log({"pi_barplot": wandb.Image(_pi_plot)})
        print(f"✓ Pi barplot → {_pi_plot}")
    except Exception as _e:
        print(f"Warning: pi barplot failed: {_e}")

# ── Model health: is this fit interpretable at all? ──────────────────────────────────────────
# psi_contrast (programs differ from each other), phi_spread (cells differ from each other), and
# whether early stopping fired in the first 25% of epochs. A collapsed fit -- all K programs
# identical to each other -- still produces a good-looking ELBO, a full pi barplot and zero dead
# programs, so this has to be recorded per run rather than eyeballed.
try:
    health = leaflet_model.health_check()
    wandb.log({f"health/{k}": v for k, v in health.items() if isinstance(v, (int, float, bool))})
    if not health["healthy"]:
        print("\n!! MODEL FAILED HEALTH CHECK — do not trust this run's programs:")
        for _p in health["problems"]:
            print(f"   - {_p}")
except Exception as _e:
    print(f"Warning: health_check failed: {_e}")
    health = {"psi_contrast": np.nan, "phi_spread": np.nan,
              "stopped_early": None, "healthy": None}

results_df = pd.DataFrame([{
    "param_id": param_id,
    "K": params["K"],
    "pruned_K": new_K,
    "junc_specific_prior": params["junc_specific_prior"],
    "dir_conc": params["delta_fixed"],
    "waypoints_use": params["waypoints_use"],
    "psi_contrast": health["psi_contrast"],
    "phi_spread": health["phi_spread"],
    "stopped_early": health["stopped_early"],
    "healthy": health["healthy"],
    "best_elbo": leaflet_model.best_elbo,
    "input_conc": leaflet_model.bb_conc,
    "num_epochs_first": NUM_EPOCHS_FIRST,
    "num_epochs_later": NUM_EPOCHS_LATER,
    "num_passes": NUM_PASSES,
    "batch_size": BATCH_SIZE,
    "total_batches": total_batches,
    "lr": params["lr"],
    "gamma": params["gamma"],
    "num_samples": params["num_samples"],
    "ELBO_num_particles": params["ELBO_num_particles"],
    "cells_coverage": (cell_assignment_counts > 0).sum() / n_cells,
    "avg_updates_per_cell": cell_assignment_counts.mean(),
}])
results_df.to_csv(os.path.join(output_dir, "run_summary.csv"), index=False)
print(f"Saved run summary → {output_dir}/run_summary.csv")

model_file = os.path.join(output_dir, "leafletfa_model.pkl.gz")
only_things_we_need = [
    'ELBO_num_particles', 'K', 'assign_post', 'best_elbo',
    'gamma', 'input_conc_prior', 'junc_specific_prior',
    'pi', 'psi', 'psi_learned', 'dir_conc', 'bb_conc', 'alpha_pi'
]
if hasattr(leaflet_model, 'psis_loc'):
    only_things_we_need.extend(['psis_loc', 'psis_scale'])
if hasattr(leaflet_model, 'a'):
    only_things_we_need.extend(['a', 'a_rate', 'a_shape', 'b', 'b_rate', 'b_shape'])

with gzip.open(model_file, "wb") as f:
    print(f"Saving model → {model_file}...")
    pickle.dump({
        "_meta": {
            "param_id": param_id,
            "data_shape": adata.shape,
            "waypoints_used": params.get("waypoints_use", False),
            "training_mode": "multi_pass_minibatch",
            "num_passes": NUM_PASSES,
            "batch_size": BATCH_SIZE,
            "total_batches": total_batches,
            "training_date": today,
            "best_elbo": leaflet_model.best_elbo,
            "original_K": original_K,
            "pruned_K": new_K,
        }
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attr_name in only_things_we_need:
        if hasattr(leaflet_model, attr_name):
            val = getattr(leaflet_model, attr_name)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                pickle.dump({attr_name: val}, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Model saved. Size: {os.path.getsize(model_file) / (1024**2):.2f} MB")

wandb.log({"training_complete": True, "final_K": new_K, "model_saved": True})
wandb.finish()

print("\n" + "="*60)
print("RUN COMPLETE")
print(f"  Device: {device}")
print(f"  Total passes: {NUM_PASSES}")
print(f"  Total batches: {total_batches}")
print(f"  Final K: {new_K} (pruned from {original_K})")
print(f"  Final ELBO: {leaflet_model.best_elbo:.4e}")
print("="*60)
