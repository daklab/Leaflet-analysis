#!/usr/bin/env python
"""QC a simulated dataset BEFORE any model is fitted to it.

Run: python 01_qc_simulation.py --sim <sim.h5ad> [--compare <other.h5ad>] --out <dir>

Purpose: confirm the simulation delivers what it claims, so that a later model result cannot be
explained by the data being wrong or unexpectedly easy. Four panels, each answering one question:

  a  Are the programs actually separable?      UMAP of the TRUE phi, coloured by true program.
                                               If these do not separate, no method could recover
                                               them and a poor result would mean nothing.
  b  Is there real differential splicing?      One high-effect ATSE: observed junction usage per
                                               program, against the true psi. Shows the signal
                                               exists in the COUNTS, not just in the parameters.
  c  Is the requested effect delivered?        True delta-psi against observed delta-psi per ATSE.
                                               Null events should sit at zero, differential ones
                                               on the diagonal.
  d  Do the junctions of an ATSE sum to 1?     The property that separates the compositional
                                               (Dirichlet-multinomial) emission from the
                                               independent-Binomial one. Exact for the former,
                                               only-in-expectation for the latter.

Colours come from `figures/_common/leaflet_theme.py`, which is the project's validated palette
(five categorical series max, CVD-checked) — not invented here.

LABELLING: anything shown to a reader is 1-INDEXED. Programs are "program 1..K", never
"program 0" — zero-based indices are an implementation detail and read as a bug in a figure.
The underlying arrays stay 0-indexed; only the label is shifted.
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
    """Layers, truth and metadata, without needing anndata."""
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
             prog=cat(f["obs"]["true_program"]).astype(str),
             role=cat(f["var"]["junction_role"]).astype(str),
             phi=np.asarray(f["obsm"]["true_phi"]),
             psi_skip=np.asarray(f["uns"]["true_psi_skip"]),
             delta=np.asarray(f["uns"]["atse_delta_psi"]),
             atse_ids=np.array([x.decode() if isinstance(x, bytes) else x
                                for x in f["uns"]["atse_ids"][:]]).astype(str),
             params=str(np.asarray(f["uns"]["sim_params"])))
    f.close()
    return d


def embed(phi, seed=0):
    """UMAP if available, PCA otherwise. Either shows whether programs separate."""
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(phi), "UMAP"
    except Exception:
        x = phi - phi.mean(0)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (u[:, :2] * s[:2]), "PCA"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--compare", default=None,
                    help="a second sim (e.g. the independent-Binomial one) for panel d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cells", type=int, default=6000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    d = load(args.sim)
    tag = os.path.basename(args.sim).replace(".h5ad", "").replace("sim_heart_K5_", "")
    uq_ev, inv_ev = np.unique(d["ev"], return_inverse=True)
    n_atse = len(uq_ev)
    progs = sorted(set(d["prog"]), key=int)
    K = len(progs)
    cols = [th.PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive")][:K]
    print(f"[1] {tag}: {d['junc'].shape[0]:,} cells x {d['junc'].shape[1]:,} junctions, "
          f"{n_atse:,} ATSEs, K={K}")
    print(f"    {d['params'][:150]}")

    fig, axes = plt.subplots(1, 4, figsize=th.figsize("double", h=2.9))

    # ── a. do the programs separate? ────────────────────────────────────────────────────
    ax = axes[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(d["prog"]), size=min(args.max_cells, len(d["prog"])), replace=False)
    xy, method = embed(d["phi"][sub])
    for i, p in enumerate(progs):
        m = d["prog"][sub] == p
        ax.scatter(xy[m, 0], xy[m, 1], s=1.5, c=cols[i], label=f"program {int(p) + 1}",
                   linewidths=0, rasterized=True)
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("True programs separate", fontsize=th.PT["panel"])
    ax.legend(markerscale=4, frameon=False, fontsize=th.PT["legend"], loc="best")
    th.panel_label(ax, "a")

    # ── b. one high-effect ATSE, in the COUNTS ──────────────────────────────────────────
    ax = axes[1]
    order = np.argsort(-d["delta"])
    pick = int(order[0])
    cols_j = np.flatnonzero(inv_ev == pick)
    roles = d["role"][cols_j]
    ratio = np.divide(d["junc"][:, cols_j], np.maximum(d["clus"][:, cols_j], 1),
                      out=np.zeros((d["junc"].shape[0], len(cols_j)), np.float32),
                      where=d["clus"][:, cols_j] > 0)
    covered = d["clus"][:, cols_j[0]] > 0
    width = 0.8 / K
    for i, p in enumerate(progs):
        m = covered & (d["prog"] == p)
        means = [ratio[m, t].mean() for t in range(len(cols_j))]
        ax.bar(np.arange(len(cols_j)) + i * width - 0.4 + width / 2, means, width,
               color=cols[i], label=f"program {int(p) + 1}", linewidth=0)
    ax.set_xticks(np.arange(len(cols_j)))
    ax.set_xticklabels(roles, fontsize=th.PT["tick"])
    ax.set_ylabel("observed junction usage")
    ax.set_title(f"Highest-effect ATSE\n(true $\\Delta\\psi$ = {d['delta'][pick]:.2f})",
                 fontsize=th.PT["panel"])
    th.panel_label(ax, "b")

    # ── c. is the requested effect delivered? ───────────────────────────────────────────
    ax = axes[2]
    obs = np.zeros(n_atse)
    skip_col = {}
    for e in range(n_atse):
        cj = np.flatnonzero(inv_ev == e)
        skip_col[e] = cj[np.flatnonzero(d["role"][cj] == "J3")[0]]
    sc = np.array([skip_col[e] for e in range(n_atse)])
    for e in range(n_atse):
        j = sc[e]
        cov = d["clus"][:, j] > 0
        if cov.sum() < 20:
            continue
        r = d["junc"][cov, j] / d["clus"][cov, j]
        pm = [r[d["prog"][cov] == p].mean() for p in progs if (d["prog"][cov] == p).sum() > 5]
        obs[e] = (max(pm) - min(pm)) if len(pm) > 1 else 0.0
    is_null = d["delta"] <= 0.05
    ax.scatter(d["delta"][is_null], obs[is_null], s=2, c=th.PALETTE["grey"],
               linewidths=0, rasterized=True, label="null")
    ax.scatter(d["delta"][~is_null], obs[~is_null], s=2, c=th.PALETTE["teal"],
               linewidths=0, rasterized=True, label="differential")
    lim = max(float(d["delta"].max()), float(obs.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=th.INK, lw=0.8, ls="--", zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("true $\\Delta\\psi$"); ax.set_ylabel("observed $\\Delta\\psi$")
    ax.set_title("Effect is delivered", fontsize=th.PT["panel"])
    ax.legend(markerscale=3, frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax, "c")

    # ── d. does each ATSE's counts sum to its total? ────────────────────────────────────
    ax = axes[3]
    def sums(dd):
        ii = np.unique(dd["ev"], return_inverse=True)[1]
        na = ii.max() + 1
        seg = np.zeros((dd["junc"].shape[0], na), np.float32)
        np.add.at(seg.T, ii, dd["junc"].T)
        rep = np.array([np.flatnonzero(ii == e)[0] for e in range(na)])
        Ne = dd["clus"][:, rep]
        cov = Ne > 0
        return (seg[cov] / Ne[cov])
    series = [(tag, sums(d), th.PALETTE["teal"])]
    if args.compare:
        d2 = load(args.compare)
        lab2 = os.path.basename(args.compare).replace(".h5ad", "").replace("sim_heart_K5_", "")
        series.append((lab2, sums(d2), th.PALETTE["ochre"]))
    bins = np.linspace(0.5, 1.5, 81)
    for lab, v, c in series:
        ax.hist(v, bins=bins, histtype="step", lw=1.4, color=c, density=True,
                label=f"{lab}\n({100*np.mean(np.abs(v-1) < 1e-6):.1f}% exact)")
    ax.axvline(1.0, color=th.INK, lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("$\\sum_j y_j\\, /\\, N_e$"); ax.set_ylabel("density")
    ax.set_title("Counts sum to the ATSE total", fontsize=th.PT["panel"])
    ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper right")
    th.panel_label(ax, "d")

    fig.tight_layout()
    # leaflet_theme.save() uses Path.with_suffix(), which treats the LAST dot-segment of the stem
    # as an extension -- so "simqc_dm50_eff0.5_frac0.3_seed0" was written as
    # "simqc_dm50_eff0.5_frac0.png", silently dropping "_seed0". Our tags carry decimals, so
    # sanitise the dots out before handing the stem over.
    safe_tag = tag.replace(".", "p")
    stem = os.path.join(args.out, f"simqc_{safe_tag}")
    th.save(fig, stem)
    print(f"[2] -> {stem}.pdf / .png")

    # numbers, so the panels are not the only record
    print(f"\n[3] summary")
    print(f"    programs (1-indexed): {K}, sizes "
          f"{ {int(p)+1: int((d['prog']==p).sum()) for p in progs} }")
    print(f"    highest-effect ATSE: {d['atse_ids'][pick]}, true delta-psi {d['delta'][pick]:.3f}")
    print(f"    differential ATSEs : {int((~is_null).sum()):,} / {n_atse:,}")
    print(f"    obs vs true delta  : r = {np.corrcoef(d['delta'][~is_null], obs[~is_null])[0,1]:.4f} "
          f"on differential events")
    print(f"    null obs delta     : median {np.median(obs[is_null]):.4f} (should be near 0)")
    for lab, v, _ in series:
        print(f"    {lab:<28} sum_j y_j / N_e: median {np.median(v):.6f}, "
              f"{100*np.mean(np.abs(v-1) < 1e-6):.1f}% exactly 1")


if __name__ == "__main__":
    main()
