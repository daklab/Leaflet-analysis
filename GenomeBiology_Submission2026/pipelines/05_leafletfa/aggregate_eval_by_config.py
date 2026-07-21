"""Aggregate 04e (imputation) + 07c (structure) eval outputs across seed replicates -> mean +/- SE.

Under the minibatch trainer each (K, junc_prior) config has several replicate models (3 seeds x the
now-equivalent waypoint/random arms). We report each metric as mean +/- standard error across those
replicates so Figure 3 has error bars.

Expects per-model eval outputs under <base>/task_<id>/{04e_out.csv,07c_out.csv} and reads
<model_dir>/parameter_combinations.json to map task_id -> (K, junc_prior, seed).

Usage: aggregate_eval_by_config.py <eval_base_dir> <model_dir> <out_prefix>
"""
import sys, os, json, glob
import numpy as np
import pandas as pd

base, model_dir, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
params = json.load(open(os.path.join(model_dir, "parameter_combinations.json")))


def config_of(task_id):
    c = params[int(task_id)]
    return c["K"], ("junc" if c["junc_specific_prior"] else "global"), c.get("seed")


def mean_se(df, group_cols, value_cols):
    out = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_replicates"] = len(g)
        for v in value_cols:
            if v in g and g[v].notna().any():
                x = g[v].dropna().values.astype(float)
                row[v + "_mean"] = float(np.mean(x))
                row[v + "_se"] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


rows04 = []
for f in glob.glob(os.path.join(base, "task_*", "04e_out.csv")):
    tid = os.path.basename(os.path.dirname(f)).replace("task_", "")
    K, prior, seed = config_of(tid)
    d = pd.read_csv(f)
    if "dataset" in d:
        d = d[d["dataset"] == "all"]
    for _, r in d.iterrows():
        rec = {"task": tid, "K": K, "prior": prior, "seed": seed,
               "entry_set": r.get("entry_set", "all_covered"),
               "downsampling_pct": r.get("downsampling_pct")}
        for k in r.index:
            if k.startswith(("median_pearson_", "calib_slope_", "calib_bias_", "mae_", "global_pearson_")):
                rec[k] = r[k]
        rows04.append(rec)
if rows04:
    df04 = pd.DataFrame(rows04)
    vcols = [c for c in df04.columns if c.startswith(("median_pearson_", "calib_slope_", "calib_bias_", "mae_", "global_pearson_"))]
    agg04 = mean_se(df04, ["K", "prior", "entry_set", "downsampling_pct"], vcols)
    agg04.to_csv(out_prefix + "_04e.csv", index=False)
    print("04e: %d model-rows -> %d config rows" % (len(df04), len(agg04)), flush=True)
    sub = agg04[agg04["entry_set"] == "lost_to_downsampling"] if "entry_set" in agg04 else agg04
    for _, r in sub.iterrows():
        print("  K=%s %-6s lost-entry lf=%.3f+-%.3f nmfm=%.3f+-%.3f calib_slope=%.3f" % (
            r["K"], r["prior"], r.get("median_pearson_lf_mean", float("nan")), r.get("median_pearson_lf_se", 0),
            r.get("median_pearson_nmfm_mean", float("nan")), r.get("median_pearson_nmfm_se", 0),
            r.get("calib_slope_lf_mean", float("nan"))), flush=True)
else:
    print("04e: no task_*/04e_out.csv found", flush=True)

rows07 = []
KEYM = ["ari_leiden_vs_broad_cell_type", "silhouette_by_broad_cell_type", "ari_leiden_vs_tissue_cell_type",
        "silhouette_by_leiden", "leiden_purity_broad_ct", "phi_perplexity_median", "n_leiden_clusters"]
for f in glob.glob(os.path.join(base, "task_*", "07c_out.csv")):
    tid = os.path.basename(os.path.dirname(f)).replace("task_", "")
    K, prior, seed = config_of(tid)
    d = pd.read_csv(f)
    for src in ("heldout_inferred", "heldout_nmfm", "train_assign_post"):
        sub = d[d["phi_source"] == src]
        row = {"task": tid, "K": K, "prior": prior, "seed": seed, "phi_source": src}
        for m in KEYM:
            v = sub[sub["metric"] == m]["value"]
            if len(v):
                row[m] = float(v.iloc[0])
        rows07.append(row)
if rows07:
    df07 = pd.DataFrame(rows07)
    agg07 = mean_se(df07, ["K", "prior", "phi_source"], KEYM)
    agg07.to_csv(out_prefix + "_07c.csv", index=False)
    print("\n07c: %d model-rows -> %d config rows" % (len(df07), len(agg07)), flush=True)
    for _, r in agg07[agg07["phi_source"] == "heldout_inferred"].iterrows():
        print("  K=%s %-6s held-out ARI_broad=%.3f+-%.3f perp=%.2f (n=%d)" % (
            r["K"], r["prior"], r.get("ari_leiden_vs_broad_cell_type_mean", float("nan")),
            r.get("ari_leiden_vs_broad_cell_type_se", 0),
            r.get("phi_perplexity_median_mean", float("nan")), int(r["n_replicates"])), flush=True)
else:
    print("07c: no task_*/07c_out.csv found", flush=True)
