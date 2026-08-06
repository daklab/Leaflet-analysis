#!/usr/bin/env python
"""
Fbln2 worked example: how the program decomposition produces its predicted PSI, and why the
predicted effect is small when the observed one is large.

This is the explanatory figure for the low-rank ceiling. It shows, for one ATSE:

  a  psi -- how each of the K=30 programs uses each of the 3 Fbln2 junctions
  b  phi-bar -- how much each cell type uses each program, and the difference between them
  c  the resulting predicted PSI per cell type, against what LeafCutter measured
  d  program 0 across ALL cell types -- testing whether the program the two heart types SHARE
     is itself a heart-identity program, i.e. carries signal in a different comparison

Panel d exists because of a specific objection: Heart endothelial and fibroblast both load
heavily on program 0, so it cancels in their contrast -- but that does not mean program 0 is
uninformative. If it separates heart from everything else, it is doing real work in a
comparison LeafCutter would have to be re-run to make.

Usage:
    python build_fbln2_program_decomposition.py
"""

import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import leaflet_theme as T  # noqa: E402

import argparse
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
_ap = argparse.ArgumentParser()
_ap.add_argument("--task", type=int, default=7, help="mb_run_<task>; 6/7/8=K30, 3/4/5=K20, 0/1/2=K10")
_ap.add_argument("--fit", default=None,
                 help="explicit fit.npz. Use this for the DM+simplex fits, whose psi is stored as "
                      "psi_mean rather than psi. Without it the OLD BetaBinomial atlas fits are "
                      "used, where psi does not sum to 1 within an ATSE (86%% valid) -- which is "
                      "precisely the defect panel c is sensitive to.")
_ap.add_argument("--tag", default=None, help="suffix for the output filename")
_ARGS, _ = _ap.parse_known_args()
FIT = _ARGS.fit or f"{WD}/LEAFLETFA_MODELS/2026-07-26_atlas_FULL/mb_run_{_ARGS.task}/fit.npz"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "FIGURE3_calibration", "reference_atse_Fbln2.json")
G1, G2 = "Heart_ENDOTHELIAL_CELL", "Heart_FIBROBLAST"
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cat(name, which):
    with h5py.File(H5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in n["categories"][:]])
            return pd.Categorical.from_codes(n["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def main():
    T.apply_theme()
    z = np.load(FIT)
    # DM+simplex fits store the point estimate as psi_mean; the released BetaBinomial fits as psi.
    phi = z["phi"]
    psi = z["psi"] if "psi" in z.files else z["psi_mean"]
    K = psi.shape[0]

    tct = np.array([safe(x) for x in cat("tissue_cell_type", "obs")])
    jid = cat("junction_id", "var")
    jpos = {k: i for i, k in enumerate(jid)}

    ref = json.load(open(REF))
    juncs = [(r["junction_id"], r["role"], r["delta_psi"],
              r["psi_Heart_ENDOTHELIAL_CELL"], r["psi_Heart_FIBROBLAST"])
             for r in ref["junctions"]]
    idx = [jpos[j[0]] for j in juncs]
    roles = [j[1] for j in juncs]

    A = phi[tct == G1].mean(0)
    B = phi[tct == G2].mean(0)
    d = A - B

    fig = plt.figure(figsize=T.figsize("double", 5.6))
    gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.32,
                          left=0.10, right=0.98, top=0.93, bottom=0.09,
                          height_ratios=[1.0, 1.0, 1.15])

    # ── a: psi for the three junctions across programs ───────────────────────
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(K)
    w = 0.27
    cols = [T.PALETTE["teal"], T.PALETTE["ochre"], T.PALETTE["indigo"]]
    for i, (jj, role) in enumerate(zip(idx, roles)):
        ax.bar(x + (i - 1) * w, psi[:, jj], w, color=cols[i], label=role)
    st = 1 if K <= 20 else 2
    ax.set_xticks(x[::st]); ax.set_xticklabels(x[::st])
    ax.set_xlabel("splicing program")
    ax.set_ylabel(r"$\psi$ — program's" "\nuse of the junction")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    T.despine(ax)
    T.panel_label(ax, "a", dx=-0.07)

    # ── b: phi-bar per cell type, and the difference ─────────────────────────
    ax = fig.add_subplot(gs[1, :])
    ax.bar(x - 0.2, A, 0.4, color=T.PALETTE["teal"], label=G1.replace("_", " ").title())
    ax.bar(x + 0.2, B, 0.4, color=T.PALETTE["ochre"], label=G2.replace("_", " ").title())
    ax.plot(x, np.abs(d), "o", ms=2.6, color=T.INK, label=r"|difference|")
    st = 1 if K <= 20 else 2
    ax.set_xticks(x[::st]); ax.set_xticklabels(x[::st])
    ax.set_xlabel("splicing program")
    ax.set_ylabel(r"$\bar{\phi}$ — cell type's" "\nuse of the program")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.text(0.99, 0.72,
            f"total differing mass $\\|\\Delta\\bar\\phi\\|_1$ = {np.abs(d).sum():.2f}\n"
            f"→ ceiling on any junction = {np.abs(d).sum()/2:.2f}",
            transform=ax.transAxes, ha="right", fontsize=T.PT["annot"])
    T.despine(ax)
    T.panel_label(ax, "b", dx=-0.07)

    # ── c: predicted vs measured PSI ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    pred_A, pred_B = A @ psi, B @ psi
    xs = np.arange(len(idx))
    ax.bar(xs - 0.33, [pred_A[j] for j in idx], 0.20, color=T.PALETTE["teal"],
           label="endothelial, model")
    ax.bar(xs - 0.11, [j[3] for j in juncs], 0.20, color=T.PALETTE["teal"], alpha=0.45,
           label="endothelial, LeafCutter")
    ax.bar(xs + 0.11, [pred_B[j] for j in idx], 0.20, color=T.PALETTE["ochre"],
           label="fibroblast, model")
    ax.bar(xs + 0.33, [j[4] for j in juncs], 0.20, color=T.PALETTE["ochre"], alpha=0.45,
           label="fibroblast, LeafCutter")
    ax.set_xticks(xs); ax.set_xticklabels(roles, fontsize=T.PT["tick"] - 1)
    ax.set_ylabel("PSI")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=T.PT["legend"] - 1, loc="upper right")
    T.despine(ax)
    T.panel_label(ax, "c", dx=-0.18)

    # ── d: is the SHARED program a heart-identity program? ───────────────────
    ax = fig.add_subplot(gs[2, 1])
    top_shared = int(np.argmax(np.minimum(A, B)))     # program both types use most
    means = pd.Series({c: phi[tct == c, top_shared].mean() for c in np.unique(tct)})
    means = means.sort_values(ascending=False)
    is_heart = np.array([c.startswith("Heart") for c in means.index])
    ax.bar(np.arange(len(means))[~is_heart], means.values[~is_heart], 1.0,
           color=T.PALETTE["grey"], label="other cell types")
    ax.bar(np.arange(len(means))[is_heart], means.values[is_heart], 1.0,
           color=T.PALETTE["rose"], label="heart cell types")
    ax.set_xlabel(f"cell types, ranked ({len(means)} total)")
    ax.set_ylabel(f"mean use of program {top_shared}")
    hr = np.flatnonzero(is_heart) + 1
    ax.set_title(f"program {top_shared}: shared by both heart types\n"
                 f"heart ranks {', '.join(map(str, hr))} of {len(means)}",
                 fontsize=T.PT["annot"], pad=4)
    ax.legend(frameon=False, fontsize=T.PT["legend"] - 1)
    T.despine(ax)
    T.panel_label(ax, "d", dx=-0.18)

    _rj = os.path.join(os.path.dirname(FIT), "result.json")
    _m = json.load(open(_rj)) if os.path.exists(_rj) else {}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"FIG3_supp_Fbln2_program_decomposition_"
                       f"K{_m.get('K', psi.shape[0])}_seed{_m.get('seed', 0)}"
                       f"{('_' + _ARGS.tag) if _ARGS.tag else ''}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved -> {out}.pdf / .png\n")

    print(f"|dphi|_1 = {np.abs(d).sum():.4f}  -> ceiling {np.abs(d).sum()/2:.4f}")
    print(f"largest |beta| over all 79,768 junctions = {np.abs(d @ psi).max():.4f}")
    print(f"\nmost-shared program = {top_shared} "
          f"(endo {A[top_shared]:.3f}, fibro {B[top_shared]:.3f})")
    print(f"  heart cell types rank {list(hr)} of {len(means)} by use of it")
    print(f"  mean in heart {means.values[is_heart].mean():.3f} vs "
          f"{means.values[~is_heart].mean():.3f} elsewhere")
    for (jn, role, dp, pa, pb), j in zip(juncs, idx):
        print(f"  {role:11s} model {pred_A[j]:.3f}/{pred_B[j]:.3f}  "
              f"LeafCutter {pa:.3f}/{pb:.3f}")


if __name__ == "__main__":
    main()
