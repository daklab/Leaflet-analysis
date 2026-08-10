#!/usr/bin/env python
"""A53: the POSTERIOR effect-size test (A51) on LeafCutter's own permutation null.

A51 is validated on simulations with known truth (FDP ratio 0.00 / 0.35 / 0.31 at effect
0.25 / 0.5 / 1.0). It has never been run on a null CONSTRUCTION, so its calibration cannot be put
next to the LRT's or LeafCutter's on equal terms. This closes that gap, using the same construction
A52 gives the LRT: donors shuffled within a single cell type.

Why donors and not cells: donor is where batch structure lives, and LeafCutter permutes donor
labels. Splitting cells at random would ignore that and produce an artificially clean null that
flatters LeafletFA.

Why this null has no contamination problem: within one cell type there is no real between-group
splicing difference, so any donor split is genuinely null -- unlike permuting phi across the whole
atlas, which converts real between-program variation on differential events into an overdispersed
mixture and inflates the null.

What to expect, stated before running. Both arms are the same cell type, so dphi should be near
zero, hence TV near zero and P(TV > delta) near zero. A calibrated test should call almost nothing.
If it calls a noticeable fraction, the posterior is too narrow for the delta being used and delta
must rise or the width must be inflated -- and that would be a real finding, not a nuisance.
"""

import argparse, os, sys
import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leafletfa.dirichlet_multinomial import build_atse_index   # noqa: E402
from A49_ds_vs_leafcutter_lrt import norm, obs_cat, H5, LC     # noqa: E402
from A51_group_ds_posterior import draw_psi                    # noqa: E402
from _phi_rows import phi_row_of                               # noqa: E402


def bh(p, alpha):
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--delta", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.fit, allow_pickle=True)
    phi = torch.from_numpy(np.asarray(z["phi"], np.float32))
    with h5py.File(H5, "r") as f:
        n = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]].astype(str)
        ct = obs_cat(f, "tissue_cell_type")
        donor = obs_cat(f, "mouse.id")
    PROW = phi_row_of(z, len(ct))
    atse_idx, _, _ = build_atse_index(ev)
    uq = np.unique(ev)
    n_atse = len(uq)

    print(f"[1] drawing {args.samples} psi posterior samples ...", flush=True)
    psi_s = draw_psi(z, atse_idx, n_atse, args.samples, args.seed)
    print(f"    {tuple(psi_s.shape)}", flush=True)

    amap = {}
    for a in np.unique(ct):
        amap.setdefault(norm(a), a)
    man = pd.read_csv(f"{LC}/permutation_null/permutation_manifest.tsv", sep="\t")
    man = man[man.status == "ready"]
    spec = man.groupby("cell_type").first().reset_index()[["cell_type", "n_armA", "n_armB"]]

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, s in spec.iterrows():
        a = amap.get(norm(s.cell_type))
        if a is None:
            continue
        cells = np.flatnonzero(ct == a); cells = cells[PROW[cells] >= 0]
        dons = np.unique(donor[cells])
        nA = max(1, min(len(dons) - 1,
                        int(round(len(dons) * s.n_armA / (s.n_armA + s.n_armB)))))
        dcell = donor[cells]
        print(f"[2] {s.cell_type}: {len(cells):,} cells, {len(dons)} donors, split {nA}/{len(dons)-nA}",
              flush=True)
        for b in range(args.perms):
            sh = rng.permutation(dons)
            m1 = np.isin(dcell, sh[:nA])
            c1 = cells[m1]
            c2 = cells[~m1]
            dphi = (phi[torch.from_numpy(PROW[c1])].mean(0)
                    - phi[torch.from_numpy(PROW[c2])].mean(0))
            d = torch.einsum("k,skj->sj", dphi, psi_s)
            tv = 0.5 * torch.zeros(psi_s.shape[0], n_atse).index_add(1, atse_idx, d.abs())
            for dl in args.delta:
                prob = (tv > dl).float().mean(0).numpy()
                p = 1.0 - prob
                sig = bh(p, args.alpha)
                rows.append(dict(cell_type=s.cell_type, perm=b, delta=dl,
                                 dphi_l1=float(dphi.abs().sum()),
                                 called=int(sig.sum()), frac=float(sig.mean()),
                                 median_tv=float(np.median(tv.mean(0).numpy()))))
            print(f"    perm {b}: |dphi|_1 {float(dphi.abs().sum()):.4f}  " +
                  "  ".join(f"delta={dl:g}->{rows[-len(args.delta)+i]['called']:,}"
                            for i, dl in enumerate(args.delta)), flush=True)

    R = pd.DataFrame(rows)
    fp = os.path.join(args.out, "a53_posterior_matched_null.tsv")
    R.to_csv(fp, sep="\t", index=False)
    print(f"\n[3] -> {fp}")
    print("\n=== posterior test on a TRUE null: what fraction of ATSEs does it call? ===")
    print(f"  a calibrated test at q<={args.alpha} should call at most ~{args.alpha:.0%} here\n")
    print(f"  {'delta':>7} {'mean called':>12} {'mean frac':>11} {'max frac':>10}")
    for dl, g in R.groupby("delta"):
        print(f"  {dl:>7.2f} {g.called.mean():>12,.0f} {g.frac.mean():>11.4f} {g.frac.max():>10.4f}")
    print(f"\n  |dphi|_1 across permutations: median {R.dphi_l1.median():.4f}, max {R.dphi_l1.max():.4f}")
    print("  (near zero is expected -- both arms are the same cell type. A large value would mean")
    print("   the donor split is picking up real donor-level structure, which is worth knowing.)")


if __name__ == "__main__":
    main()
