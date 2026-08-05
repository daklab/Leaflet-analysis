#!/usr/bin/env python
"""Three further views, modelled on panels that already exist in FIGURE2.

Run: fresh_Q5_extra.py --dir <DM_FRESH> --sim-dir <SIMULATION_DM_FRESH> --out <panels>

  a  PROGRAM CONFUSION  (FIG2 `panel_program_celltype`)
     Learned program against TRUE program: the mean loading each true program's cells place on
     each learned program, after Hungarian matching. A clean diagonal means programs map one to
     one. Off-diagonal mass means the model split one true program across two, or merged two into
     one -- a failure a UMAP can hide and a correlation cannot localise.

  b  RECOVERY vs COVERAGE
     psi recovery binned by the ATSE's read depth. None of the other panels say WHERE the model
     works; splicing coverage spans orders of magnitude and a method can look fine on average
     while failing on the shallow events that dominate real data.

  c  DS PRECISION-RECALL  (extends Q4 beyond a single alpha)
     Ranking every ATSE by the DS statistic and sweeping the threshold. Independent of any choice
     of alpha or of delta, so it measures the ranking itself rather than a calibration decision
     layered on top -- useful because A38 showed the threshold, not the ranking, is what is broken.

Labels 1-indexed; colours from the project palette.
"""
import argparse, glob, os, re, sys
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
import leaflet_theme as th  # noqa: E402


def load_sim(p):
    f = h5py.File(p, "r")
    import scipy.sparse as sp
    def dense(l):
        g = f["layers"][l]; s = tuple(g.attrs["shape"])
        return np.asarray(sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                        shape=s).todense(), np.float32)
    def cat(n):
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.asarray(n[:])
    ev = cat(f["var"]["event_id"]).astype(str); role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {x: i for i, x in enumerate(ids)}; tidx = np.array([order[x] for x in uq])
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]; K = ps.shape[0]
    tpsi = np.zeros((K, len(ev))); sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]; tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(clus=dense("cell_by_cluster_matrix"), inv=inv, n_atse=len(uq), tpsi=tpsi, role=role,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str),
             true_phi=np.asarray(f["obsm"]["true_phi"]))
    f.close(); return d


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True); ap.add_argument("--sim-dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--null-tol", type=float, default=0.05)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); th.apply_theme()

sims = sorted(glob.glob(os.path.join(a.sim_dir, "sim_heart_K5_dm50_eff*_seed0.h5ad")))
if not sims: print("no sims"); sys.exit(0)
ref = sims[-1]                                     # largest effect for the illustrative panels
base = os.path.basename(ref).replace(".h5ad", "")
eff_ref = re.search(r"eff([\d.]+)_", base).group(1)
s = load_sim(ref)
fits = {int(re.search(r"seed(\d)\.npz", p).group(1)): np.load(p, allow_pickle=True)
        for p in sorted(glob.glob(os.path.join(a.dir, f"fit_{base}_dirichletmultinomial_simplex_seed*.npz")))}
if not fits: print(f"no fits for {base}"); sys.exit(0)
K = s["tpsi"].shape[0]; ts0 = sorted(fits)[0]; z = fits[ts0]
print(f"[1] {base}, seeds {sorted(fits)}")

teal, grey = th.PALETTE["teal"], th.PALETTE["grey"]
fig, ax = plt.subplots(1, 5, figsize=th.figsize("double", h=2.6))

# ---- a. program confusion ----
C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
perm = np.empty(K, int); perm[c_] = r_
phi = z["phi"][:, perm]
M = np.zeros((K, K))
for t in range(K):
    m = s["prog"] == str(t)
    if m.sum(): M[t] = phi[m].mean(0)
im = ax[0].imshow(M, cmap=th.CMAP["sequential"], vmin=0, vmax=1, aspect="auto")
ax[0].set_xticks(range(K)); ax[0].set_xticklabels([str(i+1) for i in range(K)], fontsize=th.PT["tick"])
ax[0].set_yticks(range(K)); ax[0].set_yticklabels([str(i+1) for i in range(K)], fontsize=th.PT["tick"])
ax[0].set_xlabel("learned program"); ax[0].set_ylabel("true program")
ax[0].set_title(f"Program confusion (effect {eff_ref})", fontsize=th.PT["panel"])
plt.colorbar(im, ax=ax[0], fraction=0.046, label="mean loading")
th.panel_label(ax[0], "a")
print(f"    diagonal mean {np.mean(np.diag(M)):.4f}, max off-diagonal {np.max(M - np.diag(np.diag(M))):.4f}")

# ---- b. recovery vs coverage ----
rows = []
for p in sims:
    b = os.path.basename(p).replace(".h5ad", "")
    e = float(re.search(r"eff([\d.]+)_", b).group(1))
    ss = load_sim(p)
    for ts, zz in {int(re.search(r"seed(\d)\.npz", q).group(1)): np.load(q, allow_pickle=True)
                   for q in sorted(glob.glob(os.path.join(a.dir, f"fit_{b}_dirichletmultinomial_simplex_seed*.npz")))}.items():
        KK = ss["tpsi"].shape[0]
        Cc = np.corrcoef(zz["psi_mean"], ss["tpsi"])[:KK, KK:]
        rr, cc = linear_sum_assignment(-np.nan_to_num(Cc))
        pm = np.empty(KK, int); pm[cc] = rr
        psi = zz["psi_mean"][pm]
        rep = np.array([np.flatnonzero(ss["inv"] == q)[0] for q in range(ss["n_atse"])])
        depth = ss["clus"][:, rep].mean(0)                  # mean coverage per ATSE
        q = np.quantile(depth, [0, .2, .4, .6, .8, 1.0])
        for bi in range(5):
            m = (depth >= q[bi]) & (depth <= q[bi + 1])
            cols_ = np.isin(ss["inv"], np.flatnonzero(m))
            if cols_.sum() < 50: continue
            rows.append(dict(eff=e, seed=ts, bin=bi, depth=float(np.median(depth[m])),
                             r=float(np.corrcoef(psi[:, cols_].ravel(), ss["tpsi"][:, cols_].ravel())[0, 1])))
B = pd.DataFrame(rows)
if len(B):
    for e, g in B.groupby("eff"):
        m = g.groupby("depth").r.agg(["mean", "std"]).reset_index()
        ax[1].errorbar(m["depth"], m["mean"], yerr=m["std"], marker="o", ms=3.5, lw=1.5,
                       capsize=2, label=f"effect {e}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("ATSE mean coverage (reads)")
    ax[1].set_ylabel("corr(learned $\\psi$, true $\\psi$)")
    ax[1].legend(frameon=False, fontsize=th.PT["legend"], loc="lower right")
ax[1].set_title("Recovery vs coverage", fontsize=th.PT["panel"]); th.panel_label(ax[1], "b")

# ---- c/d/e. DS ranking: PR with AUPRC, ROC with AUROC, and both vs effect ----
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402
auc_rows = []
for p in sims:
    b = os.path.basename(p).replace(".h5ad", "")
    e = float(re.search(r"eff([\d.]+)_", b).group(1))
    ss = load_sim(p)
    fl = sorted(glob.glob(os.path.join(a.dir, f"fit_{b}_dirichletmultinomial_simplex_seed*.npz")))
    if not fl: continue
    for q in fl:                                   # every training seed, not just the first
        ts = int(re.search(r"seed(\d)\.npz", q).group(1))
        zz = np.load(q, allow_pickle=True)
        KK = ss["tpsi"].shape[0]
        Cc = np.corrcoef(zz["psi_mean"], ss["tpsi"])[:KK, KK:]
        rr, cc = linear_sum_assignment(-np.nan_to_num(Cc)); pm = np.empty(KK, int); pm[cc] = rr
        psi = zz["psi_mean"][pm]
        skip = np.array([np.flatnonzero(ss["inv"] == w)[np.flatnonzero(ss["role"][ss["inv"] == w] == "J3")[0]]
                         for w in range(ss["n_atse"])])
        score = psi[:, skip].max(0) - psi[:, skip].min(0)
        truth = (ss["delta"] > a.null_tol).astype(int)
        auc_rows.append(dict(eff=e, seed=ts, auprc=float(average_precision_score(truth, score)),
                             auroc=float(roc_auc_score(truth, score)),
                             prevalence=float(truth.mean()), score=score, truth=truth))
A = pd.DataFrame(auc_rows)
if len(A):
    prev = A.prevalence.iloc[0]
    # c: precision-recall, seed 0 curve per effect, AUPRC averaged over seeds in the legend
    for e in sorted(A.eff.unique()):
        g = A[A.eff == e]; r0 = g[g.seed == g.seed.min()].iloc[0]
        o = np.argsort(-r0["score"]); t = r0["truth"][o]
        tp = np.cumsum(t); prec = tp / np.arange(1, len(t) + 1); rec = tp / max(t.sum(), 1)
        ax[2].plot(rec, prec, lw=1.6,
                   label=f"effect {e}  AUPRC {g.auprc.mean():.3f}")
    ax[2].axhline(prev, color=th.INK, lw=0.8, ls="--", zorder=0)
    ax[2].annotate(f"chance {prev:.2f}", (0.02, prev + 0.02), fontsize=th.PT["annot"], color=th.INK)
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_ylim(0, 1.02)
    ax[2].set_title("DS ranking: precision-recall", fontsize=th.PT["panel"])
    ax[2].legend(frameon=False, fontsize=th.PT["legend"], loc="lower left")
    # d: ROC
    for e in sorted(A.eff.unique()):
        g = A[A.eff == e]; r0 = g[g.seed == g.seed.min()].iloc[0]
        fpr, tpr, _ = roc_curve(r0["truth"], r0["score"])
        ax[3].plot(fpr, tpr, lw=1.6, label=f"effect {e}  AUROC {g.auroc.mean():.3f}")
    ax[3].plot([0, 1], [0, 1], color=th.INK, lw=0.8, ls="--", zorder=0)
    ax[3].set_xlabel("false positive rate"); ax[3].set_ylabel("true positive rate")
    ax[3].set_xlim(0, 1); ax[3].set_ylim(0, 1.02)
    ax[3].set_title("DS ranking: ROC", fontsize=th.PT["panel"])
    ax[3].legend(frameon=False, fontsize=th.PT["legend"], loc="lower right")
    # e: both AUCs against effect size, error bars over training seeds
    for k, col, lab in (("auprc", th.PALETTE["teal"], "AUPRC"),
                        ("auroc", th.PALETTE["ochre"], "AUROC")):
        m = A.groupby("eff")[k].agg(["mean", "std"])
        ax[4].errorbar(m.index, m["mean"], yerr=m["std"], marker="o", ms=4.5, lw=1.7,
                       capsize=3, color=col, label=lab)
    ax[4].axhline(prev, color=th.INK, lw=0.8, ls="--", zorder=0)
    ax[4].annotate("AUPRC chance", (min(A.eff), prev + 0.02), fontsize=th.PT["annot"], color=th.INK)
    ax[4].axhline(0.5, color=th.PALETTE["grey"], lw=0.8, ls=":", zorder=0)
    ax[4].annotate("AUROC chance", (min(A.eff), 0.52), fontsize=th.PT["annot"], color=th.PALETTE["grey"])
    ax[4].set_xscale("log"); ax[4].set_xticks(sorted(A.eff.unique()))
    ax[4].set_xticklabels([str(x) for x in sorted(A.eff.unique())])
    ax[4].set_ylim(0, 1.05); ax[4].set_xlabel("simulated effect size"); ax[4].set_ylabel("area under curve")
    ax[4].set_title("Ranking quality vs effect", fontsize=th.PT["panel"])
    ax[4].legend(frameon=False, fontsize=th.PT["legend"], loc="lower right")
    A.drop(columns=["score", "truth"]).to_csv(os.path.join(a.dir, "Q5_auc.tsv"), sep="\t", index=False)
    print("\n=== ranking quality (mean over 3 training seeds) ===")
    print(A.groupby("eff")[["auprc", "auroc"]].agg(["mean", "std"]).round(4).to_string())
    print(f"    chance AUPRC = prevalence = {prev:.4f};  chance AUROC = 0.5")
th.panel_label(ax[2], "c"); th.panel_label(ax[3], "d"); th.panel_label(ax[4], "e")

fig.suptitle("DM + simplex, further views modelled on FIGURE2 panels",
             fontsize=th.PT["annot"], y=1.0)
fig.tight_layout()
th.save(fig, os.path.join(a.out, "Q5_confusion_coverage_PR"))
if len(B): B.to_csv(os.path.join(a.dir, "Q5_recovery_by_coverage.tsv"), sep="\t", index=False)
print(f"[2] -> Q5_confusion_coverage_PR")
