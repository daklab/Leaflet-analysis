#!/usr/bin/env python
"""Do independent seeds agree on the SAME ATSE? Raw data and learned psi, side by side.

Run: python panels_seed_consistency_atse.py --dir <DM_TEST> --sim-base <tag> --cell <cell>
     --out <panels>

Why this and not the aggregate number
-------------------------------------
The across-seed correlation of the TV statistic is one number over 4,268 events. It cannot
distinguish "the seeds disagree about the biology" from "I matched the programs wrong" -- and the
first version of that measurement WAS wrong, because psi was permuted into a reference seed's
program order while phi was left in its own, so every term paired the wrong program. Looking at
specific events, one column per seed, makes the failure mode visible instead of inferable.

Layout
------
Rows are ATSEs: the top block is events the simulator designed DIFFERENTIAL across programs, the
bottom block events designed NULL. Columns are seeds. Within each panel, one bar per program.

  filled bar    psi learned by that seed
  hollow bar    observed usage in the RAW counts of that seed's simulation (y_j / N_e, averaged
                over cells assigned to that program). Included because the seeds have DIFFERENT
                simulated data, so a disagreement may be in the data rather than the model.
  open circle   true psi for that seed's simulation

Programs are matched to seed 0 by the Hungarian algorithm on psi, and the SAME permutation is
applied to phi. Program identity is arbitrary across fits; comparing unmatched manufactures
disagreement, and applying the permutation to only one of psi/phi manufactures a different one.

What to look for
----------------
Differential rows should show the same program standing apart in every column. Null rows should
be flat in every column; residual spread there is the estimation noise that biases TV upward
under the null, and seeing it vary between seeds is what "unstable calls" would actually look
like.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/"
                   "figures/_common")
import leaflet_theme as th  # noqa: E402


def load_sim(path):
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

    ev = cat(f["var"]["event_id"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x
                    for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {a: i for i, a in enumerate(ids)}
    tidx = np.array([order[a] for a in uq])
    role = cat(f["var"]["junction_role"]).astype(str)
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]
    K = ps.shape[0]
    tpsi = np.zeros((K, len(ev)))
    sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]
    tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(junc=dense("cell_by_junction_matrix"), clus=dense("cell_by_cluster_matrix"),
             ev=ev, uq=uq, inv=inv, role=role, tpsi=tpsi,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str))
    f.close()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sim-dir", default="/gpfs/commons/groups/knowles_lab/Karin/"
                                         "Leaflet-analysis-WD/SIMULATION")
    ap.add_argument("--sim-base", default="sim_heart_K5_dm50_eff0.5_frac0.3",
                    help="simulation tag WITHOUT the trailing _seed<N>")
    ap.add_argument("--cell", default="dirichletmultinomial_simplex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-seed", type=int, default=0,
                    help="which simulation to use when --vary model")
    ap.add_argument("--vary", choices=["model", "sim"], default="model",
                    help="'model' (default, and the only coherent choice for a PER-EVENT view): "
                         "one simulation, three model seeds -- the truth is held fixed so the "
                         "columns are three opinions about the SAME event. 'sim': independent "
                         "simulations per seed, where the same event_id has DIFFERENT true psi in "
                         "each, so columns are not comparable event-by-event (a designed-null "
                         "event in seed 0 can be genuinely differential in seed 1). Use 'sim' "
                         "only for aggregate metrics, never for this panel.")
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--null-tol", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    seeds, fits, sims = [], {}, {}
    for s in range(3):
        if args.vary == "model":
            # ONE simulation (its own seed fixed at --sim-seed), three MODEL seeds. The truth is
            # identical in every column, which is what makes a per-event comparison meaningful.
            fp = os.path.join(args.dir,
                              f"fit_{args.sim_base}_seed{args.sim_seed}_{args.cell}_seed{s}.npz")
            sp_ = os.path.join(args.sim_dir, f"{args.sim_base}_seed{args.sim_seed}.h5ad")
        else:
            fp = os.path.join(args.dir, f"fit_{args.sim_base}_seed{s}_{args.cell}_seed{s}.npz")
            sp_ = os.path.join(args.sim_dir, f"{args.sim_base}_seed{s}.h5ad")
        if os.path.exists(fp) and os.path.exists(sp_):
            seeds.append(s)
            fits[s] = np.load(fp, allow_pickle=True)
            sims[s] = load_sim(sp_)
    if len(seeds) < 2:
        print(f"need >=2 seeds; found {seeds} for {args.sim_base} / {args.cell}")
        return
    print(f"[1] {args.cell} on {args.sim_base}, seeds {seeds}")

    ref = seeds[0]
    K = fits[ref]["psi_mean"].shape[0]
    cols = [th.PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive")][:K]

    # match every seed's programs onto the reference, by psi; apply the SAME permutation to phi
    perm = {ref: np.arange(K)}
    for s in seeds[1:]:
        C = np.corrcoef(fits[ref]["psi_mean"], fits[s]["psi_mean"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
        inv_perm = np.empty(K, dtype=int)
        inv_perm[r_] = c_
        perm[s] = inv_perm
        print(f"    seed {s} -> ref program map {inv_perm.tolist()}  "
              f"(mean matched r {np.mean([C[i, j] for i, j in zip(r_, c_)]):.4f})")

    # pick events on the REFERENCE sim; the same event_id exists in every seed's sim
    d0 = sims[ref]
    is_null = d0["delta"] <= args.null_tol
    diff_ex = np.argsort(-d0["delta"])[:args.n_examples]
    rng = np.random.default_rng(0)
    null_ex = rng.choice(np.flatnonzero(is_null), size=args.n_examples, replace=False)
    picks = [(e, "DIFFERENTIAL") for e in diff_ex] + [(e, "NULL") for e in null_ex]

    nrow, ncol = len(picks), len(seeds)
    fig, axes = plt.subplots(nrow, ncol, figsize=th.figsize("double", h=2.0 * nrow),
                             squeeze=False)
    for r, (e, kind) in enumerate(picks):
        atse = d0["uq"][e]
        for c, s in enumerate(seeds):
            ax = axes[r][c]
            d, z = sims[s], fits[s]
            e_s = int(np.flatnonzero(d["uq"] == atse)[0])
            cj = np.flatnonzero(d["inv"] == e_s)
            roles = d["role"][cj]
            cov = d["clus"][:, cj[0]] > 0
            Ne = d["clus"][cov, cj[0]]
            psi = z["psi_mean"][perm[s]]                     # into reference program order
            w = 0.8 / K
            for i in range(K):
                x = np.arange(len(cj)) + i * w - 0.4 + w / 2
                m = d["prog"][cov] == str(i)                 # truth labels are the sim's own
                obs = [(d["junc"][cov][m, cc] / Ne[m]).mean() if m.sum() else np.nan for cc in cj]
                ax.bar(x, obs, w, facecolor="none", edgecolor=cols[i], linewidth=0.9)
                ax.bar(x, psi[i, cj], w * 0.45, color=cols[i], linewidth=0,
                       label=f"program {i+1}" if (r == 0 and c == 0) else None)
                ax.scatter(x, d["tpsi"][i, cj], s=9, facecolors="none", edgecolors=th.INK,
                           linewidths=0.6, zorder=4)
            ax.set_xticks(np.arange(len(cj)))
            ax.set_xticklabels(roles, fontsize=th.PT["tick"])
            ax.set_ylim(0, 1)
            if c == 0:
                ax.set_ylabel(f"{kind}\n$\\psi$", fontsize=th.PT["label"])
            if r == 0:
                ax.set_title(f"seed {s}", fontsize=th.PT["panel"])
            sp_rec = psi[:, cj[roles == "J3"]].max() - psi[:, cj[roles == "J3"]].min()
            ax.text(0.02, 0.95, f"true $\\Delta\\psi$={d['delta'][e_s]:.2f}\nmodel={sp_rec:.2f}",
                    transform=ax.transAxes, fontsize=th.PT["annot"], va="top", color=th.INK)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper right")
    vtxt = (f"one simulation (seed {args.sim_seed}), {len(seeds)} MODEL seeds -- truth identical "
            f"in every column" if args.vary == "model" else
            "INDEPENDENT simulations -- truth DIFFERS per column, events not comparable")
    fig.suptitle(f"{args.cell} on {args.sim_base}   |   filled = learned, outline = raw data, "
                 f"circle = truth   |   programs matched to seed {ref}\n{vtxt}",
                 fontsize=th.PT["annot"], y=0.998)
    fig.tight_layout()
    tag = args.sim_base.replace("sim_heart_K5_", "").replace("_frac0.3", "").replace(".", "p")
    th.save(fig, os.path.join(args.out, f"seed_consistency_{args.vary}_{tag}_{args.cell[:2]}"
                                        f"{args.cell.split('_')[-1][:2]}"))
    plt.close(fig)
    print(f"[2] -> seed_consistency_{tag}")


if __name__ == "__main__":
    main()
