#!/usr/bin/env python
"""Recompute the NMF baseline rows with a converging solver, and test the psi_r matching rule.

    python code/refresh_nmf.py [--dry-run]

Two audit findings need the benchmark data refreshed:

  1. NMF `max_iter` was 300. `mean-impute` does NOT converge there -- it needs ~2,068 iterations --
     so the published baseline was stopped short. `0-impute` converges in ~78 either way. An
     under-converged baseline is not a fair comparison.

  2. `psi_r` matched programs on |C| but reported the signed C. Psi is a usage probability, not a
     sign-flippable loading, so an anti-correlated program has not recovered anything; matching on
     |C| could select such a pairing and then score it negative.

Only the NMF rows are refitted here, on CPU, because refitting LeafletFA means a GPU sweep that
would contend with the imputation array. Whether the LeafletFA rows ALSO need refitting is an
empirical question this script answers: it records `n_anti_matched` for every NMF fit. If the |C|
rule never once selected an anti-correlated pair across all of them, the two matching rules agree on
this data and the LeafletFA rows are unaffected -- and the script says so explicitly rather than
leaving it assumed.

Rewrites the NMF rows in data/benchmark_results.csv, data/benchmark_effect_sweep.csv and
data/rerun_logs/{odbench,prbench}_*.csv in place. LeafletFA rows are untouched.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import anndata as ad

_CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(_CODE), "data")
SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
sys.path.insert(0, _CODE)
import sim_benchmark as SB          # noqa: E402  -- reuse the fixed metrics() and fit_nmf()


def sim_path(row) -> str | None:
    """Reconstruct the .h5ad this row was produced from, the way the simulator named it.

    Two columns are absent from the older CSVs and are filled from the driver that produced them:

      benchmark_results.csv (frac sweep, step1_simulate.sh)  -- no `effect`, no `sim_bb_conc`;
          that arm ran `--effect 1.0 --bb-conc inf`.
      benchmark_effect_sweep.csv (eff1_simulate.sh)          -- no `sim_bb_conc`; `--bb-conc inf`.
          panels.py:load_overdispersion() already assumes inf for this file, so this matches.
    """
    c = row["sim_bb_conc"] if "sim_bb_conc" in row.index else float("inf")
    etag = "binom" if (pd.isna(c) or np.isinf(c)) else f"bb{c:g}"
    eff = float(row["effect"]) if "effect" in row.index and not pd.isna(row["effect"]) else 1.0
    frac, seed = float(row["frac"]), int(row["seed"])
    # Match the simulator's own tag exactly (00_simulate_splicing.py):
    #     f"...eff{args.effect}_frac{args.frac_differential:g}_seed{args.seed}"
    # `effect` is the raw argparse float, so 1.0 -> "1.0"; using :g here would render "1" and miss
    # every file. `frac` DOES use :g, so 0.1 -> "0.1". The two are formatted differently on purpose.
    p = f"{SIM}/sim_heart_K5_{etag}_eff{eff}_frac{frac:g}_seed{seed}.h5ad"
    if os.path.exists(p):
        return p
    hits = glob.glob(f"{SIM}/sim_heart_K5_{etag}_eff{eff}_*seed{seed}.h5ad")
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = ([os.path.join(DATA, f) for f in ("benchmark_results.csv",
                                                "benchmark_effect_sweep.csv")]
               + sorted(glob.glob(os.path.join(DATA, "rerun_logs", "odbench_*.csv")))
               + sorted(glob.glob(os.path.join(DATA, "rerun_logs", "prbench_*.csv"))))
    targets = [t for t in targets if os.path.exists(t)]

    cache, anti_total, checked, changed = {}, 0, 0, []
    for csv in targets:
        d = pd.read_csv(csv)
        if "model" not in d or not (d.model == "NMF").any():
            continue
        new_rows = {}
        for i, row in d[d.model == "NMF"].iterrows():
            p = sim_path(row)
            if p is None:
                print(f"  no .h5ad for {os.path.basename(csv)} row {i} "
                      f"(eff={row['effect']} frac={row['frac']} seed={row['seed']}) -- left as is")
                continue
            if p not in cache:
                sim = ad.read_h5ad(p)
                T = SB.truth_of(sim)
                res = {}
                for tag, (H, W, secs, n_it, conv) in SB.fit_nmf(sim, T["K_true"]).items():
                    m = SB.metrics(T, H, W)
                    m.update(seconds=secs, steps_used=n_it, nmf_converged=conv)
                    res[tag] = m
                cache[p] = res
                print(f"  {os.path.basename(p)}: "
                      + "  ".join(f"{t}: n_iter={v['steps_used']} conv={v['nmf_converged']} "
                                  f"anti={v['n_anti_matched']}" for t, v in res.items()), flush=True)
            m = cache[p].get(row["likelihood"])
            if m is None:
                continue
            checked += 1
            anti_total += m["n_anti_matched"]
            if abs(float(row["psi_r"]) - m["psi_r"]) > 1e-9 or \
               abs(float(row["sil"]) - m["sil"]) > 1e-9:
                changed.append((os.path.basename(csv), row["likelihood"],
                                float(row["psi_r"]), m["psi_r"], float(row["sil"]), m["sil"]))
            new_rows[i] = m
        if new_rows and not args.dry_run:
            for i, m in new_rows.items():
                for k, v in m.items():
                    if k not in d.columns:
                        d[k] = np.nan
                    d.at[i, k] = v
            d.to_csv(csv, index=False)

    print(f"\n=== psi_r matching rule ===")
    print(f"  NMF fits checked: {checked}")
    print(f"  times the |C| rule selected an ANTI-correlated pair: {anti_total}")
    print("  -> the two matching rules AGREE on this data; the LeafletFA rows are unaffected "
          "and need no refit." if anti_total == 0 else
          "  -> the rules DISAGREE somewhere; LeafletFA rows must be refitted too.")
    print(f"\n=== NMF rows whose metrics moved ({len(changed)}) ===")
    for c, lik, a, b, sa, sb in changed[:20]:
        print(f"  {c:<28} {lik:<12} psi_r {a:.3f}->{b:.3f}   sil {sa:.3f}->{sb:.3f}")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
