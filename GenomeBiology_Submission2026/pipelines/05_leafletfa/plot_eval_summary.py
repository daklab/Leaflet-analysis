"""Summary plot of the fair comparison: LeafletFA phi vs NMF-transfer vs NMF-in-sample.
Core configs only (K x prior); binomial/ablation tasks (>=36) excluded.
Panel A: structure (ARI vs cell type, 07c). Panel B: imputation calibration slope (04e, all_covered).
Mean +/- SE across seed replicates. Usage: plot_eval_summary.py <eval_base> <model_dir> <out_png>"""
import sys, os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

base, model_dir, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
params = json.load(open(os.path.join(model_dir, "parameter_combinations.json")))
def cfg(t):
    c = params[int(t)]; return c["K"], ("junc" if c["junc_specific_prior"] else "glob")

# ---- collect 07c ARI + 04e calibration per model ----
ari = {}    # (K,prior) -> {src:[vals]}
cal = {}    # (K,prior) -> [calib_slope_lf]
def add(d, k, sub, v):
    d.setdefault(k, {}).setdefault(sub, []).append(v)
for f in glob.glob(os.path.join(base, "task_*", "07c_out.csv")):
    t = int(os.path.basename(os.path.dirname(f)).replace("task_", ""))
    if t >= 36: continue
    K, pr = cfg(t); df = pd.read_csv(f)
    for src in ("heldout_inferred","heldout_nmfm_tr","heldout_nmf0_tr","heldout_nmfm","heldout_nmf0"):
        v = df[(df.phi_source==src)&(df.metric=="ari_leiden_vs_broad_cell_type")]["value"]
        if len(v): add(ari,(K,pr),src,float(v.iloc[0]))
for f in glob.glob(os.path.join(base, "task_*", "04e_out.csv")):
    t = int(os.path.basename(os.path.dirname(f)).replace("task_", ""))
    if t >= 36: continue
    K, pr = cfg(t); df = pd.read_csv(f)
    if "dataset" in df: df = df[df.dataset=="all"]
    ac = df[df.entry_set=="all_covered"] if "entry_set" in df else df
    if len(ac): cal.setdefault((K,pr),[]).append(float(ac.iloc[0].get("calib_slope_lf", np.nan)))

configs = sorted(ari.keys())
labels = [f"K{K}\n{pr}" for K,pr in configs]
x = np.arange(len(configs))

# series: label -> (source key, color, marker)
SER = [("LeafletFA φ","heldout_inferred","#0E6B63","o"),
       ("NMF-transfer (mean)","heldout_nmfm_tr","#A8641B","s"),
       ("NMF-transfer (0)","heldout_nmf0_tr","#D9A05B","^"),
       ("NMF in-sample (mean)","heldout_nmfm","#7C3B5E","D")]
def mse(vals):
    a=np.array(vals,float); return (a.mean(), a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
w = 0.19
for i,(lab,src,c,mk) in enumerate(SER):
    ms=[mse(ari[k][src]) if src in ari[k] else (np.nan,0) for k in configs]
    ax[0].errorbar(x+(i-1.5)*w, [m[0] for m in ms], yerr=[m[1] for m in ms],
                   fmt=mk, color=c, capsize=3, ms=7, label=lab, lw=1.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("ARI (Leiden φ vs broad cell type)")
ax[0].set_title("A · Structure recovery (held-out)")
ax[0].legend(fontsize=8, loc="best"); ax[0].grid(axis="y", alpha=0.3)
ax[0].axhline(0, color="k", lw=0.5)

# Panel B: calibration slope (LeafletFA) -- 1.0 = perfectly calibrated
cm=[mse(cal[k]) for k in configs]
ax[1].errorbar(x, [m[0] for m in cm], yerr=[m[1] for m in cm], fmt="o", color="#0E6B63",
               capsize=3, ms=8, lw=1.5, label="LeafletFA φ@ψ")
ax[1].axhline(1.0, ls="--", color="k", lw=1, label="perfect calibration")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("calibration slope (truth ~ pred)")
ax[1].set_title("B · Imputation calibration (all covered)")
ax[1].set_ylim(0, 1.3); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)

nrep = min((len(v.get("heldout_inferred",[])) for v in ari.values()), default=0)
fig.suptitle(f"LeafletFA vs NMF (transfer & in-sample) — mean±SE, n≈{nrep}/config  [core configs, binomial excluded]",
             fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out_png, dpi=130)
print(f"wrote {out_png}  ({len(configs)} configs: {labels})", flush=True)
