#!/usr/bin/env python
"""Three layers on the same ATSEs: what was SIMULATED, what the DATA looks like, what the MODEL
recovered.

Run: python panels_truth_data_model.py --fit <fit_*.npz> --sim <sim.h5ad> --out <dir>

The earlier panels showed truth against the model. This inserts the layer between them -- the raw
observed junction usage -- because that is where a discrepancy actually originates. Three things
can go wrong and they look different here:

  truth -> data   the simulation did not deliver the effect it claims (finite coverage, or a
                  generator bug). Visible as observed bars not matching the open circles.
  data -> model   the model failed to recover a signal that IS present in the counts. Visible as
                  recovered bars flat where observed bars separate.
  neither         both track, and any downstream problem is in the STATISTIC, not the model.
                  This is what A38 argued for the null events specifically.

Top row: ATSEs the simulator designed to be DIFFERENTIAL across programs.
Bottom row: ATSEs designed NULL. The residual spread there is pure estimation noise, and it is
what makes TV positively biased under the null.

Per junction, per program, three marks:
  open circle  true psi
  hollow bar   observed mean usage in the raw counts (y_j / N_e, averaged over that program's cells)
  solid bar    recovered psi

Labels 1-indexed; colours from the project palette.
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


def load(path):
    f = h5py.File(path, "r")

    def dense(layer):
        g = f["layers"][layer]
        shape = tuple(g.attrs["shape"])
        return np.asarray(sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                        shape=shape).todense(), dtype=np.float32)

    def cat(node):
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return c[node["codes"][:]]
        return np.asarray(node[:])

    d = dict(junc=dense("cell_by_junction_matrix"), clus=dense("cell_by_cluster_matrix"),
             ev=cat(f["var"]["event_id"]).astype(str),
             role=cat(f["var"]["junction_role"]).astype(str),
             prog=cat(f["obs"]["true_program"]).astype(str),
             delta=np.asarray(f["uns"]["atse_delta_psi"]),
             psi_skip=np.asarray(f["uns"]["true_psi_skip"]),
             atse_ids=np.array([x.decode() if isinstance(x, bytes) else x
                                for x in f["uns"]["atse_ids"][:]]).astype(str))
    f.close()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    z = np.load(args.fit, allow_pickle=True)
    psi = z["psi_mean"]
    lik, par, seed = str(z["likelihood"]), str(z["psi_parameterization"]), int(z["seed"])
    s = load(args.sim)
    uq, inv = np.unique(s["ev"], return_inverse=True)
    n_atse, K = len(uq), psi.shape[0]
    progs = sorted(set(s["prog"]), key=int)
    cols = [th.PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive")][:len(progs)]

    order = {a: i for i, a in enumerate(s["atse_ids"])}
    tidx = np.array([order[a] for a in uq])
    delta = s["delta"][tidx]
    ps = s["psi_skip"][:, tidx]
    is_null = delta <= args.null_tol
    tpsi = np.zeros((K, len(s["ev"])))
    sk = s["role"] == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]
    tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2

    rng = np.random.default_rng(0)
    diff_ex = np.argsort(-delta)[:args.n_examples]
    null_ex = rng.choice(np.flatnonzero(is_null), size=args.n_examples, replace=False)

    fig, axes = plt.subplots(2, args.n_examples, figsize=th.figsize("double", h=5.2))
    for row, (exs, lab) in enumerate([(diff_ex, "designed DIFFERENTIAL"),
                                      (null_ex, "designed NULL")]):
        for col, e in enumerate(exs):
            ax = axes[row, col]
            cj = np.flatnonzero(inv == e)
            roles = s["role"][cj]
            cov = s["clus"][:, cj[0]] > 0
            Ne = s["clus"][cov, cj[0]]
            w = 0.8 / len(progs)
            for i, p in enumerate(progs):
                m = s["prog"][cov] == p
                x = np.arange(len(cj)) + i * w - 0.4 + w / 2
                obs = [(s["junc"][cov][m, c] / Ne[m]).mean() for c in cj]   # RAW data
                # hollow bar = observed; solid bar = recovered; circle = truth
                ax.bar(x, obs, w, facecolor="none", edgecolor=cols[i], linewidth=1.0,
                       label="observed (raw counts)" if (row == 0 and col == 0 and i == 0) else None)
                ax.bar(x, psi[i, cj], w * 0.45, color=cols[i], linewidth=0,
                       label="recovered (model)" if (row == 0 and col == 0 and i == 0) else None)
                ax.scatter(x, tpsi[i, cj], s=11, facecolors="none", edgecolors=th.INK,
                           linewidths=0.7, zorder=4,
                           label="true $\\psi$" if (row == 0 and col == 0 and i == 0) else None)
            ax.set_xticks(np.arange(len(cj)))
            ax.set_xticklabels(roles, fontsize=th.PT["tick"])
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel("junction usage")
            obs_spread = max(abs(np.mean([(s["junc"][cov][s["prog"][cov] == p1, c] / Ne[s["prog"][cov] == p1]).mean()
                                          for c in cj[roles == "J3"]])
                                 - np.mean([(s["junc"][cov][s["prog"][cov] == p2, c] / Ne[s["prog"][cov] == p2]).mean()
                                            for c in cj[roles == "J3"]]))
                             for p1 in progs for p2 in progs)
            ax.set_title(f"{lab}\ntrue $\\Delta\\psi$={delta[e]:.2f}  observed={obs_spread:.2f}  "
                         f"model={psi[:, cj[roles=='J3']].max()-psi[:, cj[roles=='J3']].min():.2f}",
                         fontsize=th.PT["annot"])
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
                th.panel_label(ax, "a")
            if row == 1 and col == 0:
                th.panel_label(ax, "b")

    fig.suptitle(f"{lik} + {par}   |   outline = raw data,  filled = model,  circle = truth",
                 fontsize=th.PT["panel"], y=0.995)
    fig.tight_layout()
    simtag = os.path.basename(args.sim).replace(".h5ad", "").replace("sim_heart_K5_", "") \
                                       .replace("_frac0.3_seed0", "").replace(".", "p")
    th.save(fig, os.path.join(args.out,
                              f"truth_data_model_{simtag}_{lik[:2]}{par[:2]}_seed{seed}"))
    plt.close(fig)
    print(f"-> truth_data_model_{simtag}_{lik[:2]}{par[:2]}_seed{seed}")


if __name__ == "__main__":
    main()
