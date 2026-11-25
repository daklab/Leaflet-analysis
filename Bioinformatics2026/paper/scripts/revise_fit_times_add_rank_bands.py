#!/usr/bin/env python3
"""revise_fit_times_add_rank_bands.py -- two ledger changes, from existing files only (Karin, 2026-11-25).

1. FitMinutesTMS / FitMinutesAllen are replaced in place: source is now RUNTIME_TABLE.md (seeds 1-2, same recipe and GPU),
   fit 12.9 / 37.7 min. The seed-0 jobs (22400619, 22400622) failed while saving, so their log span is not a measured fit time.
2. Rank-band rates for the browser are appended: LeafletFA confirmed fraction and excess (confirmed - within-mouse/-donor null)
   by rank within pair (1-5, 6-20, 21-100) and half-A / within-donor read bin. TMS from check 18a, Allen from check 19.

Usage:  python scripts/revise_fit_times_add_rank_bands.py [--dry-run]
"""
import csv, hashlib, os, re, sys

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
RT = "/gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/RUNTIME_TABLE.md"
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ledger", "results_ledger.tsv")
VR = "LeafletFA VALIDATION_REPORT.md"
COLS = ["macro", "value", "display", "ci_lo", "ci_hi", "unit", "description", "source_file", "source_md5", "job_id", "record", "status"]


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def tsv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


with open(LEDGER) as fh:
    L = list(csv.DictReader(fh, delimiter="\t"))
idx = {r["macro"]: i for i, r in enumerate(L)}

# ---- 1. fit times from RUNTIME_TABLE.md ----
txt = open(RT).read()
for ds, mac, cells in (("TMS", "FitMinutesTMS", "54,694"), ("Allen", "FitMinutesAllen", "38,199")):
    line = next(l for l in txt.splitlines() if l.startswith(f"| LeafletFA (seeds 1–2) | {ds} |"))
    jobs = re.findall(r"\b2240\d{4}\b", line.split("|")[3]); m = re.search(r"fit (\d+\.\d) min", line).group(1)
    assert mac in idx, mac
    L[idx[mac]].update(value=m, display=m, source_file=RT, source_md5=md5(RT), job_id=",".join(jobs),
                       description=f"production fit, {ds}, {cells} TRAIN cells, 16,200 steps, batch 8,192, one RTX PRO 6000 Blackwell: "
                                   f"fit minutes, seeds 1-2 (same recipe and GPU as seed 0; the seed-0 jobs failed while saving)",
                       record="SpliceVI-utils RUNTIME_TABLE.md (5c1dbb4), LeafletFA rows")

# ---- 2. rank-band rates ----
BANDS = (("1-5", "OneToFive"), ("6-20", "SixToTwenty"), ("21-100", "TwentyOneToHundred"))
BINS = (("1-19", "Thin"), ("20-99", "Low"), ("100-499", "Mid"), (">=500", "High"))
new = []
for ds, src, job, rec, desc, ukey in (
        ("TMS", f"{V}/c18/18a_rank_bands.tsv", "23370802", f"{VR} Check 18", "check 18a, TMS pairs, refit on half-A mice, confirmed in half-B mice", "units"),
        ("Allen", f"{V}/c19/rank_bands.tsv", "23371384", f"{VR} Check 19", "check 19, Allen pairs with >= 4 shared donors, within donors", "pairs")):
    T = [r for r in tsv(src) if r["method"] == "LeafletFA"]; h = md5(src)
    for band, bw in BANDS:
        for b, bl in BINS:
            r = next(x for x in T if x["rank_band"] == band and x["bin"] == b)
            for q, qw, lab in (("confirmed", "Confirmed", "confirmed fraction"),
                               ("excess", "Excess", "excess confirmation (confirmed - null)")):
                v = r[q]
                new.append(dict(macro=f"RankBand{ds}{bw}{bl}{qw}", value=v, display=f"{float(v):.2f}", ci_lo="", ci_hi="", unit="fraction",
                                description=f"{desc}: LeafletFA {lab}, calls ranked {band} within pair, {b} reads; {r['calls']} calls, "
                                            f"{r[ukey]} {ukey}; uncapped, descriptive (no rule)",
                                source_file=src, source_md5=h, job_id=job, record=rec, status="final"))

names = [r["macro"] for r in new]
bad = [n for n in names if n in idx or names.count(n) > 1 or not re.fullmatch(r"[A-Za-z]+", n)]
if bad:
    sys.exit(f"refused: {bad}")
for mac in ("FitMinutesTMS", "FitMinutesAllen"):
    r = L[idx[mac]]; print(mac, r["display"], r["job_id"], r["source_md5"][:8])
for r in new:
    print(r["macro"], r["display"], os.path.basename(r["source_file"]), r["source_md5"][:8])
print(f"{len(new)} rows added, 2 revised")
if "--dry-run" not in sys.argv:
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", lineterminator="\n"); w.writeheader()
        w.writerows(L + new)
    print(f"wrote {LEDGER}")
