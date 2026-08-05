#!/usr/bin/env python
"""A47: rank lr/gamma arms on what the model LEARNED, not on whether it survived.

Stability is necessary and not sufficient. A run that never diverges but sits at a symmetric
collapse is worse than one that takes a restart and finds real structure, so ranking on "did it
finish" would pick the wrong recipe. The atlas ELBO cannot break the tie either -- its whole 11.25%
drop lands in the first ~50 steps and the tail slope flips sign between fits.

What it ranks on instead, in order:

  psi_contrast   psi.std(axis=0).mean() -- the spread of psi ACROSS programs. This IS the
                 differential-splicing effect size the model is for. ~0.003 is symmetric collapse,
                 ~0.10 is real structure. It is NECESSARY BUT NOT SUFFICIENT (a run once passed a
                 0.04 gate with true-psi recovery of only r=0.22), which is why it is read together
                 with the columns below rather than alone.
  n_live         programs carrying >=1% of cells. Effective K is 5-8 on this atlas regardless of
                 nominal K, so a recipe that finds MORE live programs is finding more real ones.
  settled        relative change in psi_contrast over the last tracked interval. This is the
                 convergence criterion, since the ELBO cannot supply one.
  restarts       how many times the NaN guard fired. Ties are broken toward fewer.

Reads the trajectories recorded by fit(track_every=...) out of each arm's fit.npz.
"""

import argparse, glob, json, os, re
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
    ap.add_argument("--sweep-dir", default=f"{W}/ATLAS_DM/lrsweep")
    ap.add_argument("--out", default=f"{W}/ATLAS_DM/lrsweep")
    args = ap.parse_args()

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.sweep_dir, "**", "fit.npz"), recursive=True)):
        d = os.path.dirname(fp)
        rj = os.path.join(d, "result.json")
        z = np.load(fp, allow_pickle=True)
        r = json.load(open(rj)) if os.path.exists(rj) else {}
        m = re.search(r"lr([0-9.]+)_g([0-9.]+)", fp)
        lr, gam = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        pc = np.asarray(z["trace_psi_contrast"], float) if "trace_psi_contrast" in z else np.zeros(0)
        nl = np.asarray(z["trace_n_live"], float) if "trace_n_live" in z else np.zeros(0)
        settled = (abs(pc[-1] - pc[-2]) / max(abs(pc[-2]), 1e-12)) if len(pc) > 2 else np.nan
        rows.append(dict(lr=lr, gamma=gam,
                         steps=r.get("steps_run", int(z.get("steps_run", 0))),
                         restarts=r.get("n_nan_restarts", np.nan),
                         psi_contrast=float(pc[-1]) if len(pc) else np.nan,
                         n_live=int(nl[-1]) if len(nl) else -1,
                         settled_pct=100 * settled if np.isfinite(settled) else np.nan,
                         conc_med=r.get("bb_conc", np.nan),
                         frac_valid=r.get("frac_valid", np.nan)))
    if not rows:
        print(f"no completed sweep arms under {args.sweep_dir}")
        return
    R = pd.DataFrame(rows).sort_values(["psi_contrast", "n_live"], ascending=False)
    R.to_csv(os.path.join(args.out, "a47_lr_sweep.tsv"), sep="\t", index=False)
    print("=== lr / gamma sweep, ranked by learned structure ===")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = R.iloc[0]
    print(f"\n  RECOMMEND lr0={best.lr:g} gamma={best.gamma:g}: "
          f"psi_contrast {best.psi_contrast:.4f}, {best.n_live} live programs, "
          f"{best.restarts:g} restarts, psi_contrast still moving {best.settled_pct:.2f}% per interval")
    if len(R) > 1:
        s = R.iloc[1]
        print(f"  runner-up lr0={s.lr:g} gamma={s.gamma:g}: psi_contrast {s.psi_contrast:.4f}, "
              f"{s.n_live} live")
        if abs(best.psi_contrast - s.psi_contrast) / max(s.psi_contrast, 1e-9) < 0.05:
            print("  NOTE the top two are within 5% on psi_contrast -- treat as a tie and prefer "
                  "the one with fewer restarts / the schedule closer to the validated recipe.")


if __name__ == "__main__":
    main()
