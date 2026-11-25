#!/usr/bin/env python3
"""add_validation_ledger_rows.py -- append the LeafletFA validation numbers to ledger/results_ledger.tsv (one row per number).

Reads only existing output files of the validation checks (no recomputation): checks 10.1, 12, 17, 18a, 18b, 19, the Allen
sense checks, and the scaling and runtime record (seed-0 production fit times from their logs; the check 18 refit's run
record). Each row carries the source file and its md5. Refuses to add a macro that already exists.

Usage:  python scripts/add_validation_ledger_rows.py [--dry-run]
"""
import csv, hashlib, json, os, re, sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
BL = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline/logs"
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ledger", "results_ledger.tsv")
VR = "LeafletFA VALIDATION_REPORT.md"
COLS = ["macro", "value", "display", "ci_lo", "ci_hi", "unit", "description", "source_file", "source_md5", "job_id", "record", "status"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tsv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def num(x, d):
    """round half-up from the value's decimal string (the source TSVs store 4 decimals, e.g. 0.0545 -> 0.055, as the report);
    a nonzero value that would print as zero gets 4 decimals"""
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if q == 0 and x != 0:
        q = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    s = f"{abs(q)}"
    return "$-$" + s if x < 0 else s


def sgn(x, d):
    return ("+" if x >= 0 else "") + num(x, d)


def ci(lo, hi, d, signed=False):
    f = sgn if signed else num
    return f" (95\\% CI {f(lo, d)} to {f(hi, d)})"


rows = []


def add(macro, value, display, unit, desc, src, job, record, lo="", hi=""):
    rows.append(dict(macro=macro, value=value, display=display, ci_lo=lo, ci_hi=hi, unit=unit, description=desc,
                     source_file=src, source_md5=md5(src), job_id=job, record=record, status="final"))


# ---- check 10.1: accuracy against held-out pseudobulk ----
for ds, job in (("TMS", ""), ("Allen", "")):
    src = f"{V}/tsv/c10_{ds}_s0_accuracy.tsv"; T = tsv(src)
    al = next(r for r in T if r["by"] == "all"); th = next(r for r in T if r["by"] == "reads" and r["bin"] == "1-19")
    e_l, e_r = float(al["err_leafletfa"]), float(al["err_raw"])
    add(f"AccErrLeafletFA{ds}", f"{e_l:.6g}", num(e_l, 4), "TV", f"check 10.1 mean TV error vs held-out pseudobulk, LeafletFA, {ds}, all units", src, job, f"{VR} Check 10")
    add(f"AccErrRaw{ds}", f"{e_r:.6g}", num(e_r, 4), "TV", f"check 10.1 mean TV error, raw TRAIN pseudobulk, {ds}, all units", src, job, f"{VR} Check 10")
    for tag, r, lab in (("All", al, "all units"), ("Thin", th, "units with 1-19 TRAIN reads")):
        d, lo, hi = float(r["diff"]), float(r["diff_lo"]), float(r["diff_hi"])
        add(f"AccDiff{tag}{ds}", f"{d:.6g}", num(d, 4) + ci(lo, hi, 4), "TV", f"check 10.1 LeafletFA minus raw TV error, {ds}, {lab}; 95% CI over cell types (1,000 resamples)",
            src, job, f"{VR} Check 10", f"{lo:.6g}", f"{hi:.6g}")

# ---- check 12: recalibrated interval coverage and tau^2 ----
for ds in ("TMS", "Allen"):
    src = f"{V}/tsv/c12_{ds}_coverage.tsv"; T = tsv(src)
    for cal, ev in (("A", "B"), ("B", "A")):
        for b, bl in (("100-499", "Mid"), ("500-inf", "High")):
            r = next(x for x in T if x["draws"] == "combined" and x["method"] == f"(a)+(b), calibrated on {cal}" and x["evaluated_on"] == ev and x["test_read_bin"] == b)
            c = float(r["coverage"])
            add(f"IntervalCoverage{ds}{bl}Cal{cal}", f"{c:.6g}", num(c, 3), "fraction",
                f"check 12 (a)+(b) interval coverage of held-out pseudobulk psi, {ds}, TEST reads {b}, tau2 from half {cal}, evaluated on half {ev} ({r['junctions']} junctions); nominal 0.95",
                src, "22816659" if ds == "TMS" else "22815095", f"{VR} Check 12")
    src = f"{V}/tsv/c12_{ds}_tau2.tsv"
    for b, bl in (("100-499", "Mid"), ("500-inf", "High")):
        r = next(x for x in tsv(src) if x["draws"] == "combined" and x["calibrated_on"] == "A" and x["test_read_bin"] == b)
        add(f"IntervalTauSq{ds}{bl}", r["tau2"], r["tau2"], "psi^2", f"check 12 model-error variance tau2, {ds}, TEST reads {b} (same from half A and half B)",
            src, "22816659" if ds == "TMS" else "22815095", f"{VR} Check 12")

BINS = (("1-19", "Thin"), ("20-99", "Low"), ("100-499", "Mid"), (">=500", "High"))


def diffs(prefix, src, job, rec, desc, comps, scope="all", key="pairs", bins=BINS):
    T = [r for r in tsv(src) if r["scope"] == scope]
    for comp, cl in comps:
        for b, bl in bins:
            r = next((x for x in T if x["bin"] == b and x["comparator"] == comp), None)
            if r is None or r.get("compared") != "True":
                continue
            d, lo, hi = float(r["diff"]), float(r["diff_lo"]), float(r["diff_hi"])
            n = r.get(key, r.get("units", ""))
            add(f"{prefix}{cl}{bl}", f"{d:.6g}", sgn(d, 3) + ci(lo, hi, 3, True), "excess",
                f"{desc}: LeafletFA minus {comp} excess confirmation (confirmed - within-mouse null), {b} reads, {n} {key if key in r else 'units'}; 95% CI bootstrap",
                src, job, rec, f"{lo:.6g}", f"{hi:.6g}")


diffs("CheckSeventeen", f"{V}/c17/differences.tsv", "23367292", f"{VR} Check 17", "check 17, TMS pairs, full-data fit",
      (("raw_pseudobulk", "Raw"), ("LeafCutter", "LeafCutter")))
diffs("CheckEighteenA", f"{V}/c18/18a_differences.tsv", "23370802", f"{VR} Check 18", "check 18a, TMS pairs, refit on half-A mice, confirmed in half-B mice",
      (("raw_pseudobulk", "Raw"), ("LeafCutter", "LeafCutter")), key="units")
diffs("CheckEighteenASameType", f"{V}/c18/18a_differences.tsv", "23370802", f"{VR} Check 18", "check 18a, same cell type across tissues",
      (("raw_pseudobulk", "Raw"),), scope="same_cell_type", key="units", bins=BINS[:2])   # the limit: 1-19 and 20-99 only
diffs("CheckEighteenB", f"{V}/c18/18b_differences.tsv", "23370802", f"{VR} Check 18", "check 18b, TMS type vs rest, refit on half-A mice, confirmed in half-B mice",
      (("raw_pseudobulk", "Raw"),), key="units")
diffs("CheckNineteen", f"{V}/c19/differences.tsv", "23371384", f"{VR} Check 19", "check 19, Allen pairs with >= 4 shared donors, within donors",
      (("raw_pseudobulk", "Raw"),))

# ---- Allen donors per pair ----
src = f"{V}/c17/allen_donors_summary.tsv"; D = {r["donors_with_both_groups"]: r for r in tsv(src)}
for lab, mac in (("0", "AllenPairsNoSharedDonor"), (">=4", "AllenPairsFourSharedDonors")):
    v = int(float(D[lab]["train"]))
    add(mac, str(v), f"{v:,}", "pairs", f"Allen group pairs (of 2,278) with {lab} donors contributing TRAIN cells of both groups", src, "23367293", f"{VR} Check 17 (Allen donor count)")

# ---- Allen sense checks ----
src = f"{V}/allen_sense/controls_summary.tsv"; C = {r["control"]: r for r in tsv(src)}
for g, gw in (("Nrxn1", "NrxnOne"), ("Nrxn2", "NrxnTwo"), ("Nrxn3", "NrxnThree")):
    r = C[f"{g} SS4"]; v = float(r["share_negative"])
    add(f"Sense{gw}SSFourDirection", f"{v:.6g}", f"{100 * v:.0f}\\%", "fraction",
        f"Allen sense check: share of {r['pairs']} excitatory x inhibitory pairs where {g} SS4 inclusion is lower in the excitatory group (expected, Nguyen 2016)",
        src, "23371679", f"{VR} Allen sense checks")
r = C["Nrxn1 SS4"]; add("SenseNrxnOneSSFourRank", r["median_rank"], f"{float(r['median_rank']):.0f}", "rank",
                        "Allen sense check: median rank of Nrxn1 SS4 by |dpsi| within excitatory x inhibitory pairs", src, "23371679", f"{VR} Allen sense checks")
src = f"{V}/allen_sense/lists_enrichment.tsv"; r = next(x for x in tsv(src) if x["list"].startswith("Irimia2014 microexons") and x["top"] == "200")
add("SenseMicroexonFold", r["fold_matched"], f"{float(r['fold_matched']):.0f}", "fold",
    "Allen sense check: enrichment of Irimia 2014 microexons (3-27 nt) among top-200 neuron vs glia differences vs matched background", src, "23371679", f"{VR} Allen sense checks")
add("SenseMicroexonDirection", r["share_expected_direction"], f"{100 * float(r['share_expected_direction']):.0f}\\%", "fraction",
    "Allen sense check: share of those microexon occurrences with higher inclusion in neurons", src, "23371679", f"{VR} Allen sense checks")
src = f"{V}/allen_sense/hierarchy_summary.tsv"; H = {r["kind"]: r for r in tsv(src)}
for k, mac in (("neuron vs glia", "NeuronGlia"), ("excitatory vs inhibitory", "ExcInh"), ("same class, other subclass", "SameClass"), ("close (same subclass)", "SameSubclass")):
    v = float(H[k]["median_frac_ge_0_1"])
    add(f"SenseHierarchy{mac}", f"{v:.6g}", num(v, 4), "fraction", f"Allen sense check: median share of ATSEs with |dpsi| >= 0.1 per pair, {k} ({H[k]['pairs']} pairs)",
        src, "23371679", f"{VR} Allen sense checks")
src = f"{V}/allen_sense/summary.json"; J = json.load(open(src))["clustering"]
for k, mac, lab in (("spearman_cophenetic_vs_taxonomy", "SenseCopheneticSpearman", "Spearman of cophenetic distance vs Allen taxonomy distance"),
                    ("ari_class_k3", "SenseARIClass", "adjusted Rand index vs Allen class (k = 3)"),
                    ("ari_subclass", "SenseARISubclass", "adjusted Rand index vs Allen subclass (k = 42)")):
    add(mac, f"{J[k]:.6g}", num(J[k], 2), "statistic", f"Allen sense check, clustering of 68 groups by splicing distance: {lab}", src, "23371679", f"{VR} Allen sense checks")
src = f"{V}/allen_sense/frame.tsv"; r = next(x for x in tsv(src) if x["scope"] == "all pairs" and x["ats_class"] == "skipped_exon")
add("SenseFrameSkippedTop", r["share_mod3_top"], num(float(r["share_mod3_top"]), 3), "fraction",
    "Allen sense check: share of top-200 skipped exons whose length is a multiple of 3, all pairs", src, "23371679", f"{VR} Allen sense checks")
add("SenseFrameSkippedBackground", r["share_mod3_matched_bg"], num(float(r["share_mod3_matched_bg"]), 3), "fraction",
    "Allen sense check: same share in matched non-differing ATSEs", src, "23371679", f"{VR} Allen sense checks")

# ---- scaling and runtime ----
for ds, job, mac, cells in (("TMS", "22400619", "FitMinutesTMS", "54,694"), ("Allen", "22400622", "FitMinutesAllen", "38,199")):
    src = f"{BL}/lfa_{ds}_c1_{job}.log"; txt = open(src).read()
    t0 = re.search(r"\[(\d\d:\d\d:\d\d)\] train counts", txt).group(1); t1 = re.findall(r"\[(\d\d:\d\d:\d\d)\]\s+\[checkpoint\] step 16,199", txt)[-1]
    m = (datetime.strptime(t1, "%H:%M:%S") - datetime.strptime(t0, "%H:%M:%S")).total_seconds() / 60
    add(mac, f"{m:.4g}", f"{m:.1f}", "min",
        f"seed-0 production fit, {ds}, {cells} TRAIN cells, 16,200 steps, one RTX PRO 6000 Blackwell: counts loaded to last checkpoint (step 16,199), from log timestamps {t0}-{t1}",
        src, job, f"{VR} Scaling and runtime record item 1")
src = f"{V}/c18/fit_halfA/run_record.json"; F = json.load(open(src))["stages"]["fit"]
add("RefitHalfMinutes", f"{F['minutes']}", f"{F['minutes']:.1f}", "min", "check 18 refit on half-A mice, production recipe, 16,200 steps, RTX PRO 6000 Blackwell: fit minutes", src, "23370799", f"{VR} Check 18")
add("RefitHalfPeakGPU", f"{F['peak_gpu_gb']}", f"{F['peak_gpu_gb']:.1f}", "GB", "check 18 refit: peak GPU memory (torch max allocated), production recipe", src, "23370799", f"{VR} Check 18")
add("RefitHalfCells", f"{F['n_cells']}", f"{F['n_cells']:,}", "cells", "check 18 refit: cells (all cells of half-A mice)", src, "23370799", f"{VR} Check 18")
src = f"{V}/scaling/scaling.tsv"; S = [r for r in tsv(src) if r["method"] == "LeafletFA"]
g5 = next(r for r in S if r["device"] == "GPU" and r["cells"] == "5000"); gf = next(r for r in S if r["device"] == "GPU" and r["cells"] == "54694")
c5 = next(r for r in S if r["device"] == "CPU")
add("ScaleGPUStepSmall", g5["s_per_step"], num(float(g5["s_per_step"]), 3), "s/step", "LeafletFA GPU seconds per step at 5,000 cells (2,000-step controlled run)", src, g5["job"], f"{VR} Scaling and runtime record")
add("ScaleGPUStepFull", gf["s_per_step"], num(float(gf["s_per_step"]), 3), "s/step", "LeafletFA GPU seconds per step at 54,694 cells (2,000-step controlled run)", src, gf["job"], f"{VR} Scaling and runtime record")
ratio = float(c5["s_per_step"]) / float(g5["s_per_step"])
add("ScaleCPUOverGPU", f"{ratio:.4g}", f"{ratio:.0f}", "ratio", "LeafletFA CPU (16 cores) over GPU seconds per step at 5,000 cells", src, f"{c5['job']},{g5['job']}", f"{VR} Scaling and runtime record")

with open(LEDGER) as fh:
    old = list(csv.DictReader(fh, delimiter="\t"))
have = {r["macro"] for r in old}; clash = [r["macro"] for r in rows if r["macro"] in have]
names = [r["macro"] for r in rows]; dup = {n for n in names if names.count(n) > 1}
bad = [n for n in names if not re.fullmatch(r"[A-Za-z]+", n)]
if clash or dup or bad:
    sys.exit(f"refused: existing {clash}, duplicate {sorted(dup)}, not letters-only {bad}")
for r in rows:
    print("\t".join([r["macro"], r["display"], os.path.basename(r["source_file"]), r["source_md5"][:8]]))
print(f"{len(rows)} rows")
if "--dry-run" not in sys.argv:
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        for r in rows:
            w.writerow(r)
    print(f"appended to {LEDGER}")
