#!/usr/bin/env python
"""Fig 2 supplement: held-out imputation metrics (moved out of the main figure -- the claim is
transferable programs, not imputation as an application).
a  within-junction r on entries whose reads were ALL removed (A3, 3 seeds): LeafletFA prior, NMF x2, junction mean
b  the same on all covered entries (cell's remaining reads used): posterior mean vs junction-mean+reads vs NMF
c  cell-type-level within-junction r (A66, all held-out cells): mean phi.psi per type vs pooled observed per type"""
import os, sys, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
LFA, LC, GREY = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"]


def celltype_level():
    rows = []
    for tag, d in [("seed0", f"{M}/IMPUTATION_DM/a66_all_cells"), ("seed1", f"{M}/IMPUTATION_DM/a66_all_cells_seed1"), ("seed2", f"{M}/IMPUTATION_DM/a66_all_cells_seed2")]:
        for lev in (0, 90):
            f = f"{d}/level_{lev:03d}.npz"
            if not os.path.exists(f): continue
            z = np.load(f)
            if "TL_n" not in z.files: continue
            n = z["TL_n"]; ok = n >= 10; obs = np.where(ok, z["TL_obs_w"] / np.maximum(z["TL_reads"], 1), np.nan)
            preds = {"prior": z["TL_pure"], "posterior mean": z["TL_post"]}
            nf = f"{d}/nmf_level_{lev:03d}.npz"
            if os.path.exists(nf) and "TL_nmf" in np.load(nf).files: preds["NMF"] = np.load(nf)["TL_nmf"]
            for k, P in preds.items():
                pred = np.where(ok, P / np.maximum(n, 1), np.nan); rs = []
                for j in range(n.shape[1]):
                    m = ok[:, j]
                    if m.sum() >= 5 and np.nanstd(obs[m, j]) > 1e-6 and np.nanstd(pred[m, j]) > 1e-6: rs.append(np.corrcoef(obs[m, j], pred[m, j])[0, 1])
                rows.append(dict(seed=tag, level=lev, method=k, r=np.median(rs)))
    return pd.DataFrame(rows)


def main():
    apply_theme()
    fs = sorted(glob.glob(f"{M}/IMPUTATION_DM/results/a3_heldout70_seed*.csv"))
    d = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)]); d = d[d.dataset == "all"]
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", h=2.5))
    ax = axes[0]; L = d[d.entry_set == "lost_to_downsampling"]
    for k, lab, c, mk, ls in [("lf", "LeafletFA prior $\\phi\\cdot\\psi$", LFA, "o", "-"), ("nmfm", "NMF, mean-imputed", LC, "s", "--"), ("nmf0", "NMF, zero-imputed", PALETTE["indigo"], "s", ":"), ("mb", "junction mean", GREY, "^", "-")]:
        g = L.groupby("downsampling_pct")[f"median_pearson_{k}"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=c, ms=3, lw=1.3, capsize=2, label=lab)
    ax.set_ylim(-0.005, 0.12); ax.set_xticks([10, 25, 50, 75, 90, 99, 100]); ax.set_xlabel("% of reads removed"); ax.set_ylabel("within-junction Pearson across cells\n(entries with NO reads left)")
    ax.set_title("single-cell resolution, no reads: at the floor for all", fontsize=6.5, loc="left"); ax.legend(frameon=False, fontsize=5)
    ax = axes[1]; A = d[d.entry_set == "all_covered"]
    for k, lab, c, mk, ls in [("lf_bb", "LeafletFA posterior mean", LFA, "o", "-"), ("mb", "junction mean + own reads", GREY, "^", "-"), ("nmfm", "NMF, mean-imputed", LC, "s", "--")]:
        g = A.groupby("downsampling_pct")[f"median_pearson_{k}"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=c, ms=3, lw=1.3, capsize=2, label=lab)
    ax.set_ylim(0, 1); ax.set_xticks([10, 25, 50, 75, 90, 99, 100]); ax.set_xlabel("% of reads removed"); ax.set_ylabel("within-junction Pearson across cells\n(all covered entries)")
    ax.set_title("with the cell's remaining reads: graceful degradation", fontsize=6.5, loc="left"); ax.legend(frameon=False, fontsize=5)
    ax = axes[2]; ct = celltype_level()
    if len(ct):
        meths = [m for m in ["prior", "posterior mean", "NMF"] if m in set(ct.method)]
        cols = {"prior": PALETTE["olive"], "posterior mean": LFA, "NMF": LC}
        x = np.arange(len(meths)); w = 0.36
        for i, lev in enumerate([0, 90]):
            g = ct[ct.level == lev].groupby("method").r.agg(["mean", "std"]).reindex(meths)
            ax.bar(x + (i - 0.5) * w, g["mean"], w, yerr=g["std"].fillna(0), color=[cols[m] for m in meths], alpha=1.0 if lev == 0 else 0.55, capsize=2, label=f"{lev}% of reads removed")
        ax.set_xticks(x); ax.set_xticklabels(meths, fontsize=6); ax.set_ylim(0, 1); ax.set_ylabel("within-junction Pearson across CELL TYPES\n(mean predicted vs pooled observed per type)")
        ax.set_title("cell-type resolution: the programs alone recover usage", fontsize=6.5, loc="left"); ax.legend(frameon=False, fontsize=5)
    for ax, L_ in zip(axes, "abc"): despine(ax); panel_label(ax, L_)
    fig.suptitle(f"Held-out imputation, atlas 70/30 split (K = 20; {len(fs)} seeds; 39,812 unseen cells)", fontsize=8, y=1.03)
    fig.tight_layout(); save(fig, os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_supp_imputation_metrics"))
    print(ct.groupby(["method", "level"]).r.agg(["mean", "std"]).round(3).to_string() if len(ct) else "no cell-type-level data")


if __name__ == "__main__":
    main()
