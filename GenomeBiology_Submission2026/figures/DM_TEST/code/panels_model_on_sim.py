#!/usr/bin/env python
"""Does the fitted model reproduce the differential splicing that was simulated?

Run: python panels_model_on_sim.py --fit <fit_*.npz> --sim <sim.h5ad> --out <dir>

The QC panels (`01_qc_simulation.py`) checked the DATA before fitting. These check the MODEL
after fitting, and the question is the one that matters biologically: for ATSEs designed to be
differentially spliced across programs, does the recovered junction usage separate the programs
the way the truth says it should -- and for ATSEs designed NOT to be differential, does it
correctly show no separation?

  a  three TRUE DIFFERENTIAL ATSEs: recovered psi per junction, one bar per program, with the
     true psi drawn as an open marker. Programs should separate, and should separate the way
     the truth does.
  b  three TRUE NULL ATSEs, same layout. Programs should sit on top of each other. Any spread
     here is estimation noise, and it is exactly the noise that makes TV positively biased
     under the null (see A38) -- so this panel shows the calibration problem directly.
  c  the whole distribution: recovered max-min psi across programs, differential vs null. The
     overlap between these two histograms IS the achievable discrimination; the fixed
     delta = 0.10 threshold is drawn on it.
  d  recovered vs true psi, all programs and junctions, coloured by whether the ATSE is
     differential. Attenuation shows as points falling below the diagonal.
  e  within-ATSE sums: exactly 1 under the simplex. The property the whole parameterisation
     exists to guarantee.

Labels are 1-indexed (program 1..K). Colours from the project palette.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/"
                   "figures/_common")
import leaflet_theme as th  # noqa: E402


def load_sim(path):
    f = h5py.File(path, "r")

    def cat(node):
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return c[node["codes"][:]]
        return np.asarray(node[:])

    d = dict(ev=cat(f["var"]["event_id"]).astype(str),
             role=cat(f["var"]["junction_role"]).astype(str),
             delta=np.asarray(f["uns"]["atse_delta_psi"]),
             psi_skip=np.asarray(f["uns"]["true_psi_skip"]),
             atse_ids=np.array([x.decode() if isinstance(x, bytes) else x
                                for x in f["uns"]["atse_ids"][:]]).astype(str))
    f.close()
    return d


def true_psi_per_junction(psi_skip, role, inv_ev, n_atse):
    """The simulator builds J3 = psi, J1 = J2 = (1 - psi)/2."""
    K = psi_skip.shape[0]
    out = np.zeros((K, len(role)))
    is_skip = role == "J3"
    out[:, is_skip] = psi_skip[:, inv_ev[is_skip]]
    out[:, ~is_skip] = (1 - psi_skip[:, inv_ev[~is_skip]]) / 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    z = np.load(args.fit, allow_pickle=True)
    psi = z["psi_mean"]
    lik, par = str(z["likelihood"]), str(z["psi_parameterization"])
    seed = int(z["seed"])
    s = load_sim(args.sim)
    uq_ev, inv_ev = np.unique(s["ev"], return_inverse=True)
    n_atse, K = len(uq_ev), psi.shape[0]
    cols = [th.PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive")][:K]

    order = {a: i for i, a in enumerate(s["atse_ids"])}
    truth_idx = np.array([order[a] for a in uq_ev])
    delta = s["delta"][truth_idx]
    is_null = delta <= args.null_tol
    tpsi = true_psi_per_junction(s["psi_skip"][:, truth_idx], s["role"], inv_ev, n_atse)

    # recovered spread across programs, per ATSE (max-min of the skip junction)
    skip = np.array([np.flatnonzero(inv_ev == e)[
        np.flatnonzero(s["role"][inv_ev == e] == "J3")[0]] for e in range(n_atse)])
    rec_spread = psi[:, skip].max(0) - psi[:, skip].min(0)

    print(f"[1] {os.path.basename(args.fit)}")
    print(f"    {lik} + {par}, seed {seed}, K={K}, {n_atse:,} ATSEs "
          f"({int((~is_null).sum()):,} differential)")

    fig = plt.figure(figsize=th.figsize("double", h=5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.35)

    # ── a / b: example ATSEs, differential and null ────────────────────────────────────
    rng = np.random.default_rng(0)
    diff_ex = np.argsort(-delta)[:args.n_examples]
    null_pool = np.flatnonzero(is_null)
    null_ex = rng.choice(null_pool, size=min(args.n_examples, len(null_pool)), replace=False)

    for row, (exs, lab) in enumerate([(diff_ex, "designed DIFFERENTIAL"),
                                      (null_ex, "designed NULL")]):
        for col, e in enumerate(exs):
            ax = fig.add_subplot(gs[row, col])
            cj = np.flatnonzero(inv_ev == e)
            roles = s["role"][cj]
            w = 0.8 / K
            for i in range(K):
                x = np.arange(len(cj)) + i * w - 0.4 + w / 2
                ax.bar(x, psi[i, cj], w, color=cols[i], linewidth=0,
                       label=f"program {i+1}" if (row == 0 and col == 0) else None)
                ax.scatter(x, tpsi[i, cj], s=9, facecolors="none",
                           edgecolors=th.INK, linewidths=0.6, zorder=3)
            ax.set_xticks(np.arange(len(cj)))
            ax.set_xticklabels(roles, fontsize=th.PT["tick"])
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel("recovered $\\psi$")
            ax.set_title(f"{lab}\ntrue $\\Delta\\psi$ = {delta[e]:.2f}, "
                         f"recovered {rec_spread[e]:.2f}", fontsize=th.PT["annot"])
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
                th.panel_label(ax, "a")
            if row == 1 and col == 0:
                th.panel_label(ax, "b")
    fig.text(0.5, 0.965, "open circles = true $\\psi$;  bars = recovered",
             ha="center", fontsize=th.PT["annot"], color=th.INK)
    # the sim tag MUST be in the filename: without it the two effect sizes write to the same
    # path and silently overwrite each other. Dots become 'p' because leaflet_theme.save() uses
    # Path.with_suffix(), which truncates a stem at its last dot.
    simtag = os.path.basename(args.sim).replace(".h5ad", "").replace("sim_heart_K5_", "") \
                                       .replace("_frac0.3_seed0", "").replace(".", "p")
    stem = f"{simtag}_{lik[:2]}{par[:2]}_seed{seed}"
    th.save(fig, os.path.join(args.out, f"model_examples_{stem}"))
    plt.close(fig)

    # ── c / d / e: the population view ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(1, 3, figsize=th.figsize("double", h=2.8))
    bins = np.linspace(0, max(0.6, float(rec_spread.max())), 61)
    ax2[0].hist(rec_spread[is_null], bins=bins, color=th.PALETTE["grey"], alpha=0.85,
                density=True, label=f"null (n={int(is_null.sum()):,})")
    ax2[0].hist(rec_spread[~is_null], bins=bins, histtype="step", lw=1.4,
                color=th.PALETTE["teal"], density=True,
                label=f"differential (n={int((~is_null).sum()):,})")
    ax2[0].axvline(args.delta, color=th.INK, lw=0.9, ls="--")
    ax2[0].annotate(f"$\\delta$ = {args.delta}", (args.delta, ax2[0].get_ylim()[1] * 0.88),
                    fontsize=th.PT["annot"], ha="left", color=th.INK, xytext=(3, 0),
                    textcoords="offset points")
    ax2[0].set_xlabel("recovered spread across programs")
    ax2[0].set_ylabel("density")
    ax2[0].set_title("Discrimination", fontsize=th.PT["panel"])
    ax2[0].legend(frameon=False, fontsize=th.PT["legend"])
    th.panel_label(ax2[0], "c")

    sub = rng.choice(psi.shape[1], size=min(6000, psi.shape[1]), replace=False)
    dj = ~is_null[inv_ev[sub]]
    ax2[1].scatter(tpsi[:, sub][:, ~dj].ravel(), psi[:, sub][:, ~dj].ravel(), s=1.5,
                   c=th.PALETTE["grey"], linewidths=0, rasterized=True, label="null ATSE")
    ax2[1].scatter(tpsi[:, sub][:, dj].ravel(), psi[:, sub][:, dj].ravel(), s=1.5,
                   c=th.PALETTE["teal"], linewidths=0, rasterized=True, label="differential")
    ax2[1].plot([0, 1], [0, 1], color=th.INK, lw=0.8, ls="--", zorder=0)
    ax2[1].set_xlabel("true $\\psi$"); ax2[1].set_ylabel("recovered $\\psi$")
    ax2[1].set_xlim(0, 1); ax2[1].set_ylim(0, 1)
    r = np.corrcoef(tpsi[:, sub].ravel(), psi[:, sub].ravel())[0, 1]
    ax2[1].set_title(f"Recovery (r = {r:.3f})", fontsize=th.PT["panel"])
    ax2[1].legend(markerscale=4, frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax2[1], "d")

    sums = np.zeros((K, n_atse))
    np.add.at(sums.T, inv_ev, psi.T)
    ax2[2].hist(sums.ravel(), bins=np.linspace(0.5, 1.5, 81), color=th.PALETTE["teal"],
                density=True)
    ax2[2].axvline(1.0, color=th.INK, lw=0.9, ls="--")
    ax2[2].set_xlabel("$\\sum_{j \\in e} \\psi_{kj}$"); ax2[2].set_ylabel("density")
    pct = 100 * float(((sums >= 0.9) & (sums <= 1.1)).mean())
    ax2[2].set_title(f"Within-ATSE sums ({pct:.1f}% valid)", fontsize=th.PT["panel"])
    th.panel_label(ax2[2], "e")

    fig2.tight_layout()
    th.save(fig2, os.path.join(args.out, f"model_population_{stem}"))
    plt.close(fig2)

    print(f"[2] panels -> {args.out}")
    print(f"[3] recovered spread: null median {np.median(rec_spread[is_null]):.4f} "
          f"(should be ~0), differential median {np.median(rec_spread[~is_null]):.4f}")
    print(f"    psi recovery r = {r:.4f}   within-ATSE sums valid {pct:.1f}%")
    print(f"    nulls above delta={args.delta}: "
          f"{100*np.mean(rec_spread[is_null] > args.delta):.1f}%  "
          f"| differentials above: {100*np.mean(rec_spread[~is_null] > args.delta):.1f}%")


if __name__ == "__main__":
    main()
