#!/usr/bin/env python
"""
Audit PSI contrast (program separation) across trained LeafletFA runs.

Answers, without retraining anything: did the K programs ever separate, or did this run
stay at the symmetric initialization?

    psi_contrast = psi.std(axis=0).mean()

psi is (K, J), so std(axis=0) is the spread across the K programs at each junction. All K
programs start at nearly the same psi. A run whose learning-rate schedule never breaks that
symmetry keeps a contrast of ~0.003 forever; a run that separates lands around ~0.1.

This is the diagnostic the existing checks miss: "0 dead programs / 0 never-dominant" reports
a fully-collapsed model as healthy, because every program being an identical copy of the others
is not the same thing as a program being empty.

Runs trained before 2026-07-12 used lr<=1e-3 with gamma=0.01, the schedule that recovers nothing
on the Figure 2 simulation. Use this to triage which of them (if any) are salvageable: the sweep
included waypoints_use=True, which starts outside the saddle, and chunked training warm-starts psi
from the previous chunk, which is itself symmetry-breaking after chunk 1 -- so waypoint runs may
have separated despite the schedule.

Usage:
    python pipelines/05_leafletfa/00c_audit_psi_contrast.py <models_dir> [<models_dir> ...]

where <models_dir> contains run_*/leafletfa_model.pkl.gz
"""

import glob
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Below this, the programs are indistinguishable from each other -- the model learned nothing.
DEAD_THRESHOLD = 0.01
# Runs that recovered structure in the calibration sim sit around here or above.
HEALTHY_REFERENCE = 0.1


def load_psi(pkl_path):
    """Stream the pickled dicts back out; run_training dumps _meta then one dict per attr."""
    meta, psi = {}, None
    with gzip.open(pkl_path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if "_meta" in obj:
                meta = obj["_meta"]
            if "psi" in obj and psi is None:
                psi = obj["psi"]
    if psi is None:
        return meta, None
    psi = np.asarray(psi.cpu().numpy() if hasattr(psi, "cpu") else psi, dtype=np.float64)
    return meta, psi


def audit(models_dir):
    rows = []
    for pkl in sorted(glob.glob(os.path.join(models_dir, "run_*", "leafletfa_model.pkl.gz"))):
        run = os.path.basename(os.path.dirname(pkl))
        try:
            meta, psi = load_psi(pkl)
        except Exception as e:  # a truncated/failed run should not kill the whole audit
            print(f"  [skip] {run}: {type(e).__name__}: {e}")
            continue
        if psi is None or psi.ndim != 2:
            print(f"  [skip] {run}: no usable 2-D psi")
            continue

        contrast = float(psi.std(axis=0).mean())
        rows.append({
            "models_dir": models_dir,
            "run": run,
            "K": psi.shape[0],
            "n_junctions": psi.shape[1],
            "psi_contrast": contrast,
            "verdict": "DEAD (never separated)" if contrast < DEAD_THRESHOLD
                       else "ok" if contrast >= HEALTHY_REFERENCE
                       else "marginal",
            "waypoints_used": meta.get("waypoints_used"),
            "best_elbo": meta.get("best_elbo"),
            "training_date": meta.get("training_date"),
        })
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    rows = []
    for models_dir in sys.argv[1:]:
        print(f"Auditing {models_dir}")
        rows.extend(audit(models_dir))

    if not rows:
        print("No runs with a saved psi found.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("psi_contrast")
    pd.set_option("display.width", 200, "display.max_rows", None)
    print()
    print(df[["run", "K", "psi_contrast", "verdict", "waypoints_used", "best_elbo"]].to_string(index=False))

    n_dead = int((df.psi_contrast < DEAD_THRESHOLD).sum())
    print(f"\n{n_dead}/{len(df)} runs below the dead threshold ({DEAD_THRESHOLD}) "
          f"-- these learned nothing and need retraining with lr=0.01 / gamma=0.5.")
    if n_dead < len(df):
        print(f"{len(df) - n_dead}/{len(df)} separated (contrast >= {DEAD_THRESHOLD}); "
              f"waypoint init and chunk warm-starting can break the symmetry even at a low LR.")

    out = "psi_contrast_audit.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
