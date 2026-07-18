#!/usr/bin/env python
"""
Step 10b: Generate LeafletFA hyperparameter grid and save to JSON.

Reads the LeafletFA config block from the dataset YAML and writes
parameter_combinations.json into the model output directory.

Usage:
    python pipelines/05_leafletfa/01_generate_params.py config/mouse_smartseq2.yaml
"""

import sys
import os
import itertools
import json
import datetime
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_generate_params.py <config.yaml>")
        sys.exit(1)

    cfg     = load_config(sys.argv[1])
    lfa_cfg = cfg.get("leafletfa", {})
    today   = datetime.datetime.now().strftime("%Y-%m-%d")

    out_dir = os.path.join(lfa_cfg["model_output_dir"], today)
    os.makedirs(out_dir, exist_ok=True)

    # Read from config or fall back to sensible defaults
    n_factors         = lfa_cfg.get("n_factors",                  20)
    if isinstance(n_factors, int):
        n_factors = [n_factors]   # allow single value or list in config
    lr_default        = lfa_cfg.get("learning_rate",               0.01)
    lrs               = lfa_cfg.get("param_grid_learning_rates",   [lr_default])
    if isinstance(lrs, (int, float)):
        lrs = [lrs]
    batch_sizes       = lfa_cfg.get("param_grid_batch_sizes",      [2048, 4096, 8192])
    num_passes        = lfa_cfg.get("param_grid_num_passes",       [5, 10, 20])
    if isinstance(num_passes, int):
        num_passes = [num_passes]
    max_junctions     = lfa_cfg.get("param_grid_max_junctions",    [5])
    waypoints_options = lfa_cfg.get("param_grid_waypoints_use",    [True, False])
    num_inits         = lfa_cfg.get("num_inits",                   3)

    input_conc_vals = lfa_cfg.get("param_grid_input_conc", [None])
    # null  → None  → Beta-Binomial (model learns concentration)
    # "inf" → str   → Binomial (run_training.py checks `is None`, so non-None → torch.tensor(inf))
    if not isinstance(input_conc_vals, list):
        input_conc_vals = [input_conc_vals]

    param_grid = {
        "input_conc":         input_conc_vals,
        "junc_specific_prior":[True, False],
        "delta_fixed":        [None],
        "K":                  n_factors,
        "waypoints_use":      waypoints_options,
        "batch_size":         batch_sizes,
        "num_passes":         num_passes,
        # FEWER epochs per chunk is BOTH faster AND gives a better posterior. Counterintuitive, but
        # measured (LeafletFA/METHODS.md 11.4c, experiments/qpsi_width_under_chunking.py): each chunk
        # sees different cells and disagrees with the others, so q(psi)'s width inflates to cover the
        # between-chunk disagreement -- and training each chunk HARDER makes it pull HARDER:
        #     60 epochs/chunk  -> q(psi) width 1.53x the full-batch reference
        #    200 epochs/chunk  -> 1.95x
        #    600 epochs/chunk  -> 2.15x
        # The psi POINT ESTIMATE is fine either way (r = 0.986-0.993 vs full-batch 0.997); it is only
        # the posterior WIDTH that inflates, which makes the DS/FDR conservative. So: cap the epochs.
        "num_epochs_first":   [150],
        "num_epochs_later":   [60],
        "ELBO_num_particles": [5],
        "num_samples":        [100],
        # gamma is the TOTAL LR decay over the run (lrd = gamma ** (1/num_epochs)), so gamma=0.01
        # decayed the LR 100x -- killing it well before the K programs broke out of the symmetric
        # init. gamma=0.5 decays only 2x. Measured on the Figure 2 simulation: lr=.001/gamma=.01
        # recovers nothing (psi_r ~ 0, agreement = chance), lr=.01/gamma=.5 recovers (psi_r=0.68).
        "gamma":              [0.5],
        # Early stopping: min_delta is ABSOLUTE and the ELBO here is ~1e6-1e9, so min_delta=100 is
        # a relative tolerance of ~1e-7 -- numerically inert. The rule degenerated to "stop after
        # `patience` epochs with no new global best", and SVI noise tripped it almost immediately
        # (one run stopped at epoch 54 of 500). Use the scale-free min_delta_rel and a patience
        # that cannot fire on noise alone.
        "min_delta":          [100],
        "min_delta_rel":      [1e-5],
        "lr":                 lrs,
        "patience":           [50],
        "max_junctions":      max_junctions,
    }

    # ── CORE + ABLATIONS, not a blind full factorial ──────────────────────────────────────────
    #
    # A full cross of every axis wastes runs on questions the Figure-2 simulation already answered,
    # and leaves out the one thing the Methods actually claims. So we build the list explicitly:
    #
    #   CORE       K x waypoints_use x seed
    #              The two things we genuinely need: (a) SELECT K on held-out data, and (b) COMPARE
    #              random init against waypoint init. (b) matters a lot now: with the corrected
    #              schedule (lr=0.05, gamma=0.5) random init recovers simulated ground truth to
    #              psi_r=0.99 / 100% program agreement, so "programs are recovered from random init;
    #              waypoints just converge faster" is a far stronger claim than "we needed waypoints".
    #
    #   SEEDS      The Methods says the random-init model "is fit three times, and the one with the
    #              best ELBO was selected". That was NEVER done -- run_training.py called
    #              train(num_initializations=1). It is done here, as separate runs with different
    #              seeds (3 restarts of the WHOLE run, which is what the sentence means -- NOT 3
    #              restarts per chunk, which is what num_initializations would give). Waypoint init
    #              is deterministic given the data, so it gets one seed.
    #
    #   PSI PRIOR  A CORE AXIS, not an ablation -- promoted 2026-07-13 after it was finally tested.
    #              `psi_kj ~ Beta(a_j, b_j)` indexes a_j/b_j by JUNCTION ONLY, so the SAME Beta prior
    #              is shared across all K factors at a junction. That is literally a shrinkage prior
    #              pulling the K programs TOWARD EACH OTHER -- i.e. toward the symmetric saddle that
    #              is this model's main failure mode (METHODS.md 11.1). A global Beta(a,b) has no
    #              per-junction pull. Measured on simulated data with known truth:
    #                 junction-specific  psi_r +0.395 / +0.158 / +0.386   (seeds 0/1/2)
    #                 global             psi_r +0.994 / +0.994 / +0.993
    #              The global prior is not just better, it REMOVES the seed variance -- and the ELBO
    #              prefers it too (5.07e4 vs 6.5e4-4.2e5), so best-ELBO selection would find it.
    #              The paper uses junction-specific and it was NEVER tested. Cross it; let held-out
    #              data on the atlas decide.
    #
    #   ABLATION   Likelihood (Beta-Binomial vs Binomial) is settled by the simulation, so we confirm
    #              it ONCE at the reference K rather than crossing it with everything:
    #                 truth=Binomial, model=BB   -> psi_r 0.984, agreement 99.9%, AUROC 0.998
    #                 truth=Binomial, model=Bin  -> psi_r 0.758, agreement 45.1%, AUROC 0.914
    #              The Beta-Binomial wins EVEN WHEN THE TRUTH IS BINOMIAL. Beta-Binomial is the default.
    #
    # Every run is trained on the 70% split and scored on the held-out 30% (07c / 04*). SELECT ON
    # HELD-OUT METRICS, NOT ON THE ELBO.

    K_REF   = lfa_cfg.get("reference_k", 20)         # the K the ablations are run at
    SEEDS   = list(range(int(num_inits)))            # 3 restarts, per the Methods

    def make(K, waypoints, seed, input_conc, junc_prior, tag, lr=None):
        p = dict(
            input_conc=input_conc, junc_specific_prior=junc_prior, delta_fixed=None,
            K=K, waypoints_use=waypoints, seed=seed, arm=tag,
            # MEASURED: psi reaches its final distance from init in PASS 1 and then does not move.
            # run_14 psi_mac_vs_init across 20 passes: 0.469 0.482 0.488 0.486 0.476 0.499 ... 0.470
            # -- flat at 0.478 +- 0.015 with no trend, while pass-to-pass movement stays ~0.12. That
            # is DIFFUSION, not convergence: psi wanders on a shell without getting further from its
            # start. Independent seeds agree 0.86 and independent inits agree 0.95, so the wandering
            # is in directions that do not change the recovered programs. The per-cell ELBO likewise
            # bottoms out at pass 2 (7062) and drifts slightly UP to pass 20 (7963).
            # => passes beyond ~2-3 buy nothing. 8 keeps margin at 2.5x less GPU than 20.
            batch_size=batch_sizes[0], num_passes=num_passes[0],
            num_epochs_first=150, num_epochs_later=60,
            ELBO_num_particles=5, num_samples=100,
            gamma=0.5, min_delta=100, min_delta_rel=1e-5, patience=50,
            lr=(lrs[0] if lr is None else lr), max_junctions=max_junctions[0],
        )
        if waypoints:
            p["n_waypoints"]      = K
            # TEN PCs, fixed, per the Methods -- NOT min(K, 50). It used to scale with K, so the
            # waypoint space silently changed dimension across the K sweep and only K=10 matched the
            # paper. A fixed budget also keeps the waypoint geometry comparable across K.
            p["n_pca_components"] = 10
            p["n_dim_components"] = 10
            # 50 nearest neighbours, per the Methods. Was 30.
            p["metacell_size"]    = 50
        p["num_inits"] = num_inits
        return p

    BB, GLOBAL_PRIOR = None, False       # input_conc=None -> Beta-Binomial; junc_prior default True
    param_list = []

    # CORE: K x lr x init x seed  -- BOTH init arms get the SAME number of seeds.
    #
    # This matters. Giving the random arm 3 restarts (keep the best) and the waypoint arm only 1
    # would bias the comparison IN FAVOUR of random init -- which is exactly the conclusion we are
    # hoping for, and therefore the one we can least afford to rig. Equal seeds or the comparison is
    # worthless.
    #
    # NB the waypoint INITIALISATION is deterministic (max_min_sampling has a hardcoded seed=42, and
    # initialize_psi_phi takes a MEAN), but the guide and the SVI sampling still vary with the run
    # seed, so the waypoint arm has genuine run-to-run variance too.
    #
    # lr is crossed because 0.02 and 0.05 are BOTH strong on the simulation (psi_r 0.97 / 0.98) and
    # the atlas is a different regime (sparser, 20x more cells). Let the held-out data pick.
    #
    # junc_prior is crossed for the reason in the header: the junction-specific prior is a shrinkage
    # prior toward the symmetric saddle, and on simulated data the global prior beats it 0.99 vs 0.31
    # AND removes the seed variance. That is too large an effect to settle on one simulation, and too
    # large to leave as an untested default. Both arms, all seeds.
    for K in n_factors:
        for lr in lrs:
            for jp, ptag in ((True, "juncprior"), (False, "globalprior")):
                for wp, itag in ((True, "waypoints"), (False, "random")):
                    for sd in SEEDS:
                        param_list.append(make(K, wp, sd, BB, jp, f"core_{itag}_{ptag}", lr=lr))

    # ABLATION -- likelihood: Binomial instead of Beta-Binomial, at K_REF, both init arms
    for wp in (True, False):
        param_list.append(make(K_REF, wp, 0, "inf", True, "ablation_binomial"))

    param_file = os.path.join(out_dir, "parameter_combinations.json")
    with open(param_file, "w") as f:
        json.dump(param_list, f, indent=2)

    csv_file = os.path.join(out_dir, "parameter_combinations.csv")
    pd.DataFrame(param_list).to_csv(csv_file, index=False)

    n_waypoint = sum(1 for p in param_list if p["waypoints_use"])
    n_random   = len(param_list) - n_waypoint
    print(f"Generated {len(param_list)} parameter sets ({n_waypoint} with waypoints, {n_random} random init)")
    print(f"  → {len(param_list)} SLURM jobs (one per param set; num_inits stored in JSON but not used by run_training.py)")
    print(f"  → {param_file}")
    print(f"Output directory: {out_dir}")
    print(f"\nTo submit jobs:")
    print(f"  python pipelines/05_leafletfa/02_submit_jobs.py {sys.argv[1]} {out_dir} <anndata.h5ad>")


if __name__ == "__main__":
    main()
