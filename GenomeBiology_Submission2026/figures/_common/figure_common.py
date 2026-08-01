"""Helpers shared by the simulation and atlas analyses.

The two are kept in SEPARATE notebooks and separate plotting scripts even though they overlap heavily,
because they answer different questions and their conclusions are not interchangeable -- simulation has
ground truth and has twice mispredicted the atlas (an initialisation that helped at c=2 in simulation
made the real atlas worse). Anything genuinely common lives here so the two cannot silently drift apart.

Nothing in this module plots. It loads, matches, and measures.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
ATLAS_WD = f"{WD}/MOUSE_SPLICING_FOUNDATION"
SIM_DIR = f"{WD}/SIMULATION"
TRAIN_H5 = f"{ATLAS_WD}/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad"
TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}
STEP_CAP = 5000


# ── loading ────────────────────────────────────────────────────────────────────────────────
def load_fits(model_dir, tasks=None):
    """Every saved latent per task, plus K/seed/steps pulled from result.json."""
    out = {}
    for t in (range(9) if tasks is None else tasks):
        p = f"{model_dir}/mb_run_{t}/fit.npz"
        if not os.path.exists(p):
            continue
        d = dict(np.load(p))
        r = json.load(open(f"{model_dir}/mb_run_{t}/result.json"))
        d["K"], d["seed"] = int(r["K"]), int(r["seed"])
        d["steps"] = len(np.asarray(d["losses"]))
        out[t] = d
    return out


def health(fits, cap: int = STEP_CAP) -> pd.DataFrame:
    """Finite? Stopped before the cap? Flat at the end?

    Run this before any metric. On 2026-07-29 four fits hit a non-finite loss, recovered psi/phi/
    psis_loc through the checkpoint restore, exited 0, and left psis_scale 100% NaN -- and the
    evaluations were queued against them before anyone noticed.
    """
    rows = []
    for t, f in fits.items():
        bad = [k for k, v in f.items()
               if isinstance(v, np.ndarray) and v.size
               and not np.isfinite(np.asarray(v, np.float64)).all()]
        L = np.asarray(f["losses"], float)
        k = max(10, len(L) // 5)
        rows.append(dict(task=t, K=f["K"], seed=f["seed"], steps=f["steps"],
                         stopped_early=f["steps"] < cap,
                         tail_slope=float(np.polyfit(np.arange(k), L[-k:], 1)[0] / abs(L[-1])),
                         nonfinite=", ".join(bad) or "-"))
    d = pd.DataFrame(rows)
    d.attrs["all_ok"] = bool((d.nonfinite == "-").all() and d.stopped_early.all())
    return d


def celltypes(path: str = TRAIN_H5, key: str = "broad_cell_type") -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as fh:
        s = fh["mod/splicing"] if "mod/splicing" in fh else fh
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)])


# ── comparing fits ─────────────────────────────────────────────────────────────────────────
def perm_from_psi(a, b):
    """Program correspondence from Psi, WITH the margin over the next-best partner.

    The margin is not optional. At K=30 on the atlas, 62% of programs match a second partner within
    0.02 of the one chosen, so a matched statistic can be a coin-flip between similar programs. Check
    the margin before quoting anything that depends on this permutation.
    """
    K = a.shape[0]
    A = a - a.mean(1, keepdims=True)
    B = b - b.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    C = np.abs(A @ B.T)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-C)
    col = np.empty(K, int)
    col[r] = c
    margin = np.array([C[i, col[i]] - np.delete(C[i], col[i]).max() for i in range(K)])
    return col, C[r, c], margin


def cca(A, B):
    """Canonical correlations between two subspaces.

    Invariant to program order AND to how the same variance was divided among programs, so it needs
    no matching. Controls on the atlas: self vs self 1.000, self vs column-permuted self 0.997,
    self vs junction-shuffled 0.016.
    """
    A = np.asarray(A, np.float64); A = A - A.mean(0)
    B = np.asarray(B, np.float64); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def psi_subspace_agreement(psi_a, psi_b):
    """Matching-free psi agreement, on the CENTRED profiles.

    Centring matters: 88% of psi's variance is the per-junction baseline every program shares, which
    carries no information about program identity and inflates any raw comparison.
    """
    a = (np.asarray(psi_a, np.float64) - np.asarray(psi_a, np.float64).mean(0)).T
    b = (np.asarray(psi_b, np.float64) - np.asarray(psi_b, np.float64).mean(0)).T
    return cca(a, b)


def cellcell(phi):
    """Cell x cell similarity of the loading vectors. Permutation-invariant."""
    Z = np.asarray(phi, np.float32)
    Z = Z - Z.mean(1, keepdims=True)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return Z @ Z.T


def group_rate(phi, psi, labels, groups):
    """Predicted splicing rate per (group, junction): (mean phi within group) @ psi.

    The biological output, and the one comparison that needs no matching at all -- group labels are
    fixed, so two fits are directly comparable.
    """
    M = np.vstack([np.asarray(phi, np.float32)[labels == g].mean(0) for g in groups])
    return M @ np.asarray(psi, np.float32)


def effect_slope(dev_a, dev_b):
    """Regression slope of one fit's effects on the other's -- NOT the correlation.

    Correlation is scale-invariant and reads ~1.0 even when one fit has systematically shrunk every
    effect. Only the slope detects attenuation.
    """
    return float(np.polyfit(np.asarray(dev_a).ravel(), np.asarray(dev_b).ravel(), 1)[0])


def topn_overlap(dev_a, dev_b, n: int = 10000):
    ia = set(np.argsort(-np.asarray(dev_a).ravel())[:n].tolist())
    ib = set(np.argsort(-np.asarray(dev_b).ravel())[:n].tolist())
    return len(ia & ib) / n


def pairs(tasks):
    return list(itertools.combinations(sorted(tasks), 2))
