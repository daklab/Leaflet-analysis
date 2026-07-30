#!/usr/bin/env python
"""Step 08c: select the released model on HELD-OUT metrics -- gate on cell type, RANK on the rest.

WHY THE RANKING METRIC IS NOT CELL-TYPE DECODABILITY
----------------------------------------------------
The paper's claim is that LeafletFA finds splicing programs *without cell-type labels*. Selecting
the model that best reproduces those labels is therefore circular, and it actively selects against
the interesting result: a model whose phi merely recovers cell type adds nothing you could not get
from cell type itself. It also pushes K toward "number of cell types", which is not obviously the
right K for splicing.

So cell-type decodability is used as a GATE (is the latent space biologically real at all?), and
models are RANKED on metrics that measure information BEYOND cell type:

  GATE   held-out broad_cell_type balanced accuracy > shuffled-phi null + margin
         -- a floor, not a goal. Fails => the model learned nothing usable.

  RANK   1. age_within_tissue_cell_type (Spearman, MEDIAN across surviving groups, sex partialled
            out) -- does phi carry signal beyond cell type? This is the value proposition.
         2. imputation margin over mb_ct (the cell-type-stratified mean). If LeafletFA cannot beat
            a per-cell-type average, its programs are redundant with cell type.
         3. coverage robustness -- psi_r retained at low read depth.

  REPORT memorization gap (train - heldout) and cross-seed agreement, always. A model that scores
         well but disagrees with its own replicate seeds is not a result.

Usage:
    python 08c_select_on_heldout.py <heldout_csv> [--imputation-csv F] [--out F]
"""
import sys, argparse, numpy as np, pandas as pd

CONFIG = ["K", "likelihood", "junc_prior", "waypoints", "delta_fixed"]


def _cfg_cols(df):
    return [c for c in CONFIG if c in df.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("heldout_csv")
    ap.add_argument("--imputation-csv", default=None)
    ap.add_argument("--gate-margin", type=float, default=0.05,
                    help="held-out cell-type accuracy must exceed the shuffled null by this much")
    ap.add_argument("--out", default="heldout_selection.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.heldout_csv)
    cfg = _cfg_cols(d)
    if not cfg:
        sys.exit(f"no config columns found in {a.heldout_csv}; have {list(d.columns)[:12]}")

    # ── GATE: held-out cell type vs its shuffled-phi null ────────────────────────────────────
    ct = d[(d.target == "broad_cell_type") & (d.get("phi_source", "heldout") == "heldout")]
    null = d[d.get("scope", "") == "shuffled_phi_null"]
    nullv = float(null.score.mean()) if len(null) else np.nan
    gate = (ct.groupby(cfg, dropna=False).score.mean().rename("celltype_heldout").reset_index())
    gate["null"] = nullv
    gate["passes_gate"] = gate.celltype_heldout > (nullv + a.gate_margin) if np.isfinite(nullv) \
                          else gate.celltype_heldout > a.gate_margin

    # ── RANK 1: age beyond cell type (median across groups, sex partialled out) ──────────────
    age = d[(d.target == "age") & (d.get("scope", "all") == "all")]
    agg = {"age_spearman_median": ("score", "mean")}
    if "spearman_min" in age.columns:
        agg |= {"age_spearman_min": ("spearman_min", "mean"),
                "age_spearman_max": ("spearman_max", "mean")}
    rank = age.groupby(cfg, dropna=False).agg(**agg).reset_index() if len(age) else gate[cfg].copy()

    out = gate.merge(rank, on=cfg, how="outer")

    # seed spread + memorization gap
    for src, name in (("heldout", "heldout"), ("train", "train")):
        sub = d[(d.target == "broad_cell_type") & (d.get("phi_source", "heldout") == src)]
        if len(sub):
            g = sub.groupby(cfg, dropna=False).score.agg(["mean", "std", "size"])
            out = out.merge(g.rename(columns={"mean": f"ct_{name}", "std": f"ct_{name}_sd",
                                              "size": f"n_seeds_{name}"}).reset_index(),
                            on=cfg, how="left")
    if "ct_train" in out and "ct_heldout" in out:
        out["memorization_gap"] = out.ct_train - out.ct_heldout

    # ── RANK 2: imputation margin over the cell-type-stratified mean ─────────────────────────
    if a.imputation_csv:
        try:
            im = pd.read_csv(a.imputation_csv)
            if {"median_pearson_lf", "median_pearson_mb_ct"} <= set(im.columns):
                im["margin_over_mb_ct"] = im.median_pearson_lf - im.median_pearson_mb_ct
                icfg = _cfg_cols(im)
                if icfg:
                    out = out.merge(im.groupby(icfg, dropna=False)
                                      .margin_over_mb_ct.mean().reset_index(), on=icfg, how="left")
        except Exception as e:
            print(f"  (imputation csv not merged: {e})")

    # ── rank the models that PASS the gate ───────────────────────────────────────────────────
    keys = [c for c in ("age_spearman_median", "margin_over_mb_ct") if c in out.columns]
    passed = out[out.passes_gate].copy()
    if keys and len(passed):
        passed = passed.sort_values(keys, ascending=False)

    pd.set_option("display.width", 200)
    print(f"GATE: held-out broad_cell_type > shuffled null ({nullv:.3f}) + {a.gate_margin}")
    print(f"  {int(out.passes_gate.sum())}/{len(out)} configs pass\n")
    if len(passed):
        show = cfg + [c for c in ("celltype_heldout", "memorization_gap", "age_spearman_median",
                                  "margin_over_mb_ct", "ct_heldout_sd", "n_seeds_heldout")
                      if c in passed.columns]
        print("RANKED (on information BEYOND cell type, not on cell type itself):")
        print(passed[show].round(4).to_string(index=False))
        print("\nSELECTED:", passed.iloc[0][cfg].to_dict())
    else:
        print("*** NO CONFIG PASSES THE GATE -- the latent space is not decodable above the null.")
    failed = out[~out.passes_gate]
    if len(failed):
        print(f"\nfailed the gate ({len(failed)}):")
        print(failed[cfg + ["celltype_heldout"]].round(4).to_string(index=False))
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
