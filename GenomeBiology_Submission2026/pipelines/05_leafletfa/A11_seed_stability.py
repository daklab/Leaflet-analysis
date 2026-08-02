"""Why do independent seeds disagree on phi, and can it be fixed?

A10 established that only ~40-50% of the phi directions reproduce across seeds at any K, that this is
not sparse coverage (agreement is flat across a 450-fold depth range) and not dead programs (pi is
near-uniform, no program below half-uniform).

RESULT (2026-07-27): BOTH candidates below were tested and BOTH are refuted. Doubling the training
budget (1,334 -> 2,584 steps) moved effective K by +1.0, and dropping the junction-specific prior by
-1.0, against a within-arm spread of 2 programs across seed-pairs. Neither shift is readable.

The prior arm is worth keeping despite the null, because it is a direct test of the collinearity
hypothesis: it DID lower program correlation (+0.876 -> +0.807) and reproducibility did NOT improve.
Manipulating the proposed cause changed nothing, which is evidence against it. (Raw Psi-Psi
correlation of +0.87 is in any case mostly the shared per-junction baseline, which cannot cause
degeneracy since phi is on the simplex; the CENTRED correlations are exactly -1/(K-1), i.e. maximally
decorrelated.)

The surviving hypothesis is the learned overdispersion capping per-cell information -- see
`09_simulation/A13_overdispersion_vs_reproducibility.py`.

The two candidates, scored as a 2x2 against the existing models:

  CONVERGENCE  every model in the original run stopped early at 1,151-1,401 steps of a 5,000 cap with
               the ELBO still moving ~1% over the final 20%. `min_delta_rel=1e-4` demands ~1e5
               improvement against a ~1e9 loss while late-stage progress is ~4e4 per step, so the rule
               fires while the objective is still descending -- and at a different step in every seed.
               Seeds stopped at different places on a moving objective will differ for reasons that
               have nothing to do with the model.
  PSI PRIOR    the junction-specific prior ties all K programs at junction j to a shared
               Beta(a_j, b_j). That is a mechanism for pulling programs together, and is the standing
               suspicion for the +0.87 program correlation. The global-prior arm removes it.

Readout is EFFECTIVE K: canonical correlations between two seeds' phi, counted above a threshold.
Computed on TRAINING phi, which covers the same 92,894 cells in every seed, so no E-step is needed
(verified against the held-out version: identical at K=10, within 1-2 of 30 at K=20/30).

Also reported: program-program correlation (the proposed mechanism -- if the junction prior is the
cause, the global arm should show lower correlation AND higher effective K), and steps used (did the
looser stopping rule actually let the fits run longer).

Usage: A11_seed_stability.py <out_csv> [threshold]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
BASE = f"{WD}/LEAFLETFA_MODELS/2026-07-26_atlas9"
NEW = f"{WD}/LEAFLETFA_MODELS/2026-07-27_seedstab"

# (label, model_dir, tasks) -- the fourth cell of the 2x2 comes from the original run
ARMS = [
    ("junction-specific, default stopping", BASE, (3, 4, 5)),
    ("junction-specific, tight stopping",   NEW,  (0, 1, 2)),
    ("global prior, tight stopping",        NEW,  (3, 4, 5)),
]


def cca(A, B):
    """Canonical correlations: invariant to program ordering and to any redivision among programs."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def load(model_dir, task):
    f = f"{model_dir}/mb_run_{task}/fit.npz"
    r = f"{model_dir}/mb_run_{task}/result.json"
    if not (os.path.exists(f) and os.path.exists(r)):
        return None
    d = np.load(f)
    return dict(phi=np.asarray(d["phi"]), psi=np.asarray(d["psi"], float),
                n_steps=len(np.asarray(d["losses"])), res=json.load(open(r)))


def main():
    out_csv = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    rows = []
    for label, md, tasks in ARMS:
        M = {t: load(md, t) for t in tasks}
        M = {t: v for t, v in M.items() if v is not None}
        if len(M) < 2:
            print(f"{label:<38} not ready ({len(M)}/3 models)", flush=True)
            continue
        K = int(list(M.values())[0]["res"]["K"])
        # program-program correlation, the proposed mechanism
        pp = []
        for v in M.values():
            C = np.corrcoef(v["psi"])
            pp.append(C[~np.eye(K, dtype=bool)].mean())
        eff, steps = [], [v["n_steps"] for v in M.values()]
        for a, b in itertools.combinations(sorted(M), 2):
            eff.append(int((cca(M[a]["phi"], M[b]["phi"]) > thr).sum()))
        rows.append(dict(arm=label, K=K, n_models=len(M), effective_K=float(np.mean(eff)),
                         effective_K_min=int(min(eff)), effective_K_max=int(max(eff)),
                         frac=float(np.mean(eff)) / (K - 1), prog_corr=float(np.mean(pp)),
                         steps_mean=float(np.mean(steps)), steps_min=int(min(steps)),
                         steps_max=int(max(steps))))
        print(f"{label:<38} effective K = {np.mean(eff):.1f}/{K-1} ({np.mean(eff)/(K-1):.0%})   "
              f"prog-corr {np.mean(pp):+.3f}   steps {int(np.mean(steps))} "
              f"[{min(steps)}-{max(steps)}]", flush=True)

    if not rows:
        sys.exit("no arms ready")
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)

    print("\n=== what moved? ===", flush=True)
    base = d[d.arm.str.contains("default stopping")]
    tight = d[d.arm == "junction-specific, tight stopping"]
    glob = d[d.arm.str.contains("global prior")]
    # The noise floor is not a fixed constant: it is how much each arm's OWN seed-pairs disagree.
    # An arm spanning [8, 11] cannot support a between-arm claim of +1.
    floor = float(np.max(d.effective_K_max - d.effective_K_min))
    print(f"  (noise floor: the widest within-arm spread across seed-pairs is {floor:.0f} programs, "
          f"so a between-arm difference must exceed that to be readable)", flush=True)

    def verdict(delta, what):
        return (f"{what} IS a contributor." if abs(delta) > floor else
                f"{what} is NOT the explanation -- the shift is inside the within-arm spread.")

    if len(base) and len(tight):
        de = tight.effective_K.iloc[0] - base.effective_K.iloc[0]
        ds = tight.steps_mean.iloc[0] - base.steps_mean.iloc[0]
        print(f"  CONVERGENCE: {ds:+.0f} steps -> effective K {de:+.1f}. "
              + verdict(de, "Under-training"), flush=True)
    if len(tight) and len(glob):
        de = glob.effective_K.iloc[0] - tight.effective_K.iloc[0]
        dc = glob.prog_corr.iloc[0] - tight.prog_corr.iloc[0]
        print(f"  PSI PRIOR: dropping the junction-specific prior moves program correlation "
              f"{dc:+.3f} and effective K {de:+.1f}. " + verdict(de, "The prior"), flush=True)
        if abs(dc) > 0.03 and abs(de) <= floor:
            print(f"     NOTE: program correlation moved {dc:+.3f} while reproducibility did not. "
                  f"That is a direct manipulation of the proposed cause with no effect -- evidence "
                  f"AGAINST program collinearity driving the degeneracy.", flush=True)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
