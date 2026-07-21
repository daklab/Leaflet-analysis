"""Adapt a minibatch fit (mb_run_<id>/fit.npz + result.json) into the run-dir layout that 04e/07c
expect: leafletfa_model.pkl.gz + run_summary.csv.

The minibatch trainer saves psi/phi/globals in an .npz; the eval scripts read a pickled dict plus a
run_summary.csv. This writes those two files into <out_dir> so the existing 04e/07c can run unchanged
on a minibatch model. psis_loc/psis_scale are NOT emitted (the minibatch fit stores the posterior
mean, not the guide's per-site loc/scale), so 04e falls back to the psi point estimate -- fine for a
smoke; note it discards psi posterior uncertainty.

Usage: mb_to_evalmodel.py <mb_run_dir> <out_dir>
"""
import sys, os, json, gzip, pickle
import numpy as np
import pandas as pd

mb_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
d = np.load(os.path.join(mb_dir, "fit.npz"))
r = json.load(open(os.path.join(mb_dir, "result.json")))

psi = np.asarray(d["psi"], np.float32)              # (K, J)
model = {
    "psi": psi,
    "psi_learned": psi,
    "assign_post": np.asarray(d["phi"], np.float32),  # (N, K)
    "pi": np.asarray(d["pi"], np.float32),
    "alpha_pi": np.asarray(d["alpha_pi"], np.float32),
    "dir_conc": float(np.asarray(d["dir_conc"])),
    "bb_conc": float(np.asarray(d["bb_conc"])),
    "a": np.asarray(d["a"], np.float32), "b": np.asarray(d["b"], np.float32),
    "K": int(r["K"]),
}
with gzip.open(os.path.join(out_dir, "leafletfa_model.pkl.gz"), "wb") as f:
    pickle.dump(model, f)

summary = {
    "param_id": r.get("task_id", 0), "K": int(r["K"]),
    "junc_specific_prior": bool(r.get("junc_prior", False)),
    "waypoints_use": False, "pruned_K": int(r["K"]),
    "batch_size": int(r.get("subsample_size", 0)),
    "best_elbo": float(r.get("final_loss") or float("nan")),
    "trainer": "minibatch",
}
pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)
print(f"wrote {out_dir}/leafletfa_model.pkl.gz  (K={r['K']}, psi {psi.shape}, "
      f"psi_mean={psi.mean():.4f}) + run_summary.csv", flush=True)
