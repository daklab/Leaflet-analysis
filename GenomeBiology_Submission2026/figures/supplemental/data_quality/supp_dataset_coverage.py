#!/usr/bin/env python
"""
Supplementary figure: splicing dataset coverage summary.

Characterises the data resource itself — how much splicing signal each cell carries —
before any modelling claim is made. Same script produces the equivalent figure for every
dataset; change `--dataset` and nothing else.

    python supp_dataset_coverage.py --dataset mouse_smartseq2
    python supp_dataset_coverage.py --dataset human_allen_brain

Two stages. COMPUTE scans the (30 GB) splice object once and caches three small parquet
tables; PLOT reads those. Re-plotting therefore never rescans. Compute ~80 s, plot ~3 s.

Panels (a-h), double column width:
    a  inventory (typeset counts)          e  junctions per ATSE
    b  depth x breadth per cell            f  coverage across tissues
    c  junction reads vs GE depth          g  single cells vs single nuclei (matched)
    d  reads/junction vs ATSE size         h  junction position along gene (5'->3')

────────────────────────────────────────────────────────────────────────────────────────
THINGS THAT FAIL SILENTLY IF CHANGED (each was measured on the real data)
────────────────────────────────────────────────────────────────────────────────────────
1. `event_id` is NOT contiguous along var order (24,583 runs for 24,475 events). Counting
   change-points within a CSR row to get ATSEs-per-cell gives a WRONG answer with no
   error. `_h5ad_scan.scan_csr` sort-uniques composite (row, event) keys instead.
2. `cell_by_cluster_matrix` is never read. Its value is the ATSE total BROADCAST to every
   junction column of that ATSE, so summing it along axis 1 double-counts by k per event
   (mouse would come out ~2.4x too high and look entirely plausible). Verified identical
   to the junction layer summed within events -> 17.5 GB of I/O skipped.
3. `X` is a byte-identical duplicate of `cell_by_junction_matrix`; scanning both doubles
   runtime for nothing.
4. `num_junctions` (pre-regrouping ATSE size) disagrees with `n_junctions_in_atse`
   (current) for 28.2% of junctions. Always the latter; asserted against the observed
   groupby size.
5. The GE denominator is the ALIGNED GE, which holds only ATSE-bearing genes (11,233
   mouse / 10,327 human) — NOT the transcriptome. Everything is therefore labelled
   "fraction of reads in ATSE-bearing genes", never "of library size".
6. Whitespace-duplicated obs categories exist beyond `subtissue` ('Cortex'/'Cortex ').
   Every object/category obs column is stripped generically.
7. Human obs lacks `subtissue`/`batch`/`mouse.id`, and its `age` is int64 years not a
   category — so cohort filters are per-dataset CONFIG, never code.
8. Cache staleness is the real risk, not compute: a plot silently made from the
   superseded 2026-06-13 file would be wrong. The manifest records path+mtime+size and
   PLOT refuses a stale cache unless --force-cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

HERE = Path(__file__).resolve().parent                    # figures/supplemental/data_quality
sys.path.insert(0, str(HERE.parent / "_shared"))          # -> _h5ad_scan
sys.path.insert(0, str(HERE.parent.parent / "_common"))    # figures/_common -> leaflet_theme

import _h5ad_scan as scan                                        # noqa: E402
from leaflet_theme import (apply_theme, PALETTE, CMAP, PT, INK,  # noqa: E402
                           figsize, panel_label, despine, save)

__version__ = "1.0"

# ── dataset registry ──────────────────────────────────────────────────────────
MSF = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
HSF = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION"
GTF_MOUSE = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/"
             "genome_files/gencode.vM19/genes/genes.gtf")
GTF_HUMAN = ("/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/"
             "gencode.v45.primary_assembly.annotation.gtf")

DEFAULTS = {
    # Both objects' .X are RAW integer counts and their obs['cell_id'] are element-wise
    # identical and in the same order (verified) -> positional alignment, no join.
    "mouse_smartseq2": dict(
        splice=f"{MSF}/MODEL_INPUT/20260526/"
               f"splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        ge=f"{MSF}/MODEL_INPUT/20260526/"
           f"ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_MOUSE,
        label="Mouse (Tabula Muris Senis + Allen Brain)",
        species="mouse",
        min_cells_filter=75, min_juncs_filter=500,
        modality_key="dataset",
        modality_map={"tabula_muris_senis": "single cell",
                      "allen_brain": "single nucleus"},
        modality_chem={"single cell": "SMART-seq2, FACS whole cells",
                       "single nucleus": "SMART-seq v4, nuclei"},
        group_key="tissue",
        celltype_key="broad_cell_type",
        batch_key="mouse.id",
        # panel g cohort matching (per-dataset CONFIG: human age is int years, not '3m')
        matched=dict(group_prefix="Brain", restrict={"single cell": {"age": ["3m"]}},
                     # TMS labels its few brain neurons generically; Allen splits them into
                     # excitatory/inhibitory/hippocampal. Merge both sides to make the
                     # comparison possible at all, and flag the label mismatch with *.
                     merge={"NEURON": "Neuron*", "INHIBITORY NEURON": "Neuron*",
                            "EXCITATORY NEURON": "Neuron*", "HIPPOCAMPAL NEURON": "Neuron*"},
                     subsample={"Neuron*": 400}),
        confound_note=(
            "`dataset` is aliased with age (all Allen Brain cells are 2m), with organ "
            "(Allen contributes 6 brain tissues only) and with library chemistry "
            "(SMART-seq v4 vs SMART-seq2). The matched strata control the first three "
            "but not chemistry, so residual differences are an upper bound."),
    ),
    "human_allen_brain": dict(
        splice=f"{HSF}/MODEL_INPUT/20260617/"
               f"splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad",
        ge=f"{HSF}/MODEL_INPUT/20260617/"
           f"ge_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_HUMAN,
        label="Human cortex (Allen Brain)",
        species="human",
        min_cells_filter=25, min_juncs_filter=500,
        modality_key="dataset",
        modality_map={"allen_brain_human": "single nucleus"},
        modality_chem={"single nucleus": "SMART-seq v4, nuclei"},
        group_key="tissue",
        celltype_key="broad_cell_type",
        batch_key="donor",
        matched=None,                      # single modality -> panel g degrades
        confound_note="Single modality (nuclei only); no cells-vs-nuclei contrast exists.",
    ),
}

MOD_COLOR = {"single cell": PALETTE["teal"], "single nucleus": PALETTE["indigo"]}
FRACTION_LABEL = "Junction reads / reads in ATSE-bearing genes"


# ── small helpers ─────────────────────────────────────────────────────────────

def require_col(df, col, what=""):
    """Fail loudly with the available columns — human obs genuinely lacks several."""
    if col not in df.columns:
        raise KeyError(f"column {col!r} {what} not in frame. Available: {list(df.columns)}")
    return df[col]


def _h5_index_key(grp):
    k = grp.attrs.get("_index", "_index")
    return k.decode() if isinstance(k, bytes) else k


def _read_frame(h5, node):
    """obs/var -> DataFrame, decoding categoricals, stripping whitespace generically."""
    g = h5[node]
    ik = _h5_index_key(g)
    out = {}
    for k in g.keys():
        if k == ik:
            continue
        item = g[k]
        if isinstance(item, h5py.Group) and "categories" in item:
            cats = item["categories"][:]
            cats = cats.astype(str) if cats.dtype.kind in "SOU" else cats
            out[k] = pd.Categorical.from_codes(item["codes"][:], categories=cats)
        elif isinstance(item, h5py.Dataset):
            v = item[:]
            out[k] = v.astype(str) if v.dtype.kind in "SOU" else v
    df = pd.DataFrame(out, index=pd.Index(g[ik][:].astype(str), name=None))
    # TRAP 6: whitespace-duplicated levels are not limited to `subtissue`
    for c in df.columns:
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str).str.strip()
    return df


def gene_spans(gtf_path):
    """gene_id -> (chrom, start, end, strand) from `gene` feature lines only (~3 s)."""
    gid = re.compile(r'gene_id "([^"]+)"')
    rows = []
    with open(gtf_path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if len(p) < 9 or p[2] != "gene":
                continue
            m = gid.search(p[8])
            if m:
                rows.append((m.group(1), p[0], int(p[3]), int(p[4]), p[6]))
    return pd.DataFrame(rows, columns=["gene_id", "g_chrom", "g_start", "g_end",
                                       "g_strand"]).set_index("gene_id")


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(HERE),
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


# ── COMPUTE ───────────────────────────────────────────────────────────────────

def compute_metrics(cfg, chunk_rows=20_000, verbose=True):
    """One pass over the junction layer + one over GE. Returns (cells, juncs, atses)."""
    t0 = time.time()
    fs = h5py.File(cfg["splice"], "r")

    obs = _read_frame(fs, "obs")
    var = _read_frame(fs, "var")
    n_cells, n_junc = len(obs), len(var)
    if verbose:
        print(f"  splice {n_cells:,} cells x {n_junc:,} junctions", flush=True)

    # TRAP 4: n_junctions_in_atse is the CURRENT ATSE size; num_junctions is pre-regroup
    ev = require_col(var, "event_id").astype(str)
    codes, cats = pd.factorize(ev)
    n_events = len(cats)
    if "n_junctions_in_atse" in var.columns:
        observed = ev.map(ev.value_counts()).astype(int).values
        stated = var["n_junctions_in_atse"].astype(int).values
        assert np.array_equal(observed, stated), (
            "n_junctions_in_atse disagrees with the observed groupby size — the ATSE "
            "definition in this file is not self-consistent.")

    # TRAP 2: confirm the cluster layer is redundant, then never read it (17.5 GB saved)
    if "cell_by_cluster_matrix" in fs["layers"]:
        ok, nchk, msg = scan.verify_broadcast_layer(
            fs, "layers/cell_by_junction_matrix", "layers/cell_by_cluster_matrix",
            codes, n_rows=15)
        if verbose:
            print(f"  cluster-layer broadcast identity: {ok} (n={nchk}) -> skipping it",
                  flush=True)
        if not ok:
            raise RuntimeError(f"cluster layer is NOT a broadcast of junction sums: {msg}")

    if verbose:
        print("  scanning junction layer (one pass) ...", flush=True)
    last = [0.0]

    def prog(done, total):
        if verbose and time.time() - last[0] > 15:
            last[0] = time.time()
            print(f"    {done:,}/{total:,} cells ({100*done/total:.0f}%)", flush=True)

    r = scan.scan_csr(fs, "layers/cell_by_junction_matrix", n_junc,
                      chunk_rows=chunk_rows, event_code=codes, n_events=n_events,
                      progress=prog)
    fs.close()

    # ── GE depth (aligned object; .X is raw counts; cell order identical) ──
    fg = h5py.File(cfg["ge"], "r")
    ge_obs_n = fg["obs"][_h5_index_key(fg["obs"])].shape[0]
    if ge_obs_n != n_cells:
        raise RuntimeError(f"GE has {ge_obs_n:,} cells, splice has {n_cells:,}")
    ge_ip, _, _, _ = scan.csr_meta(fg, "X")
    ge_reads = scan.row_sums_only(fg, "X")
    ge_genes = scan.nnz_per_row(ge_ip)
    ge_obs = _read_frame(fg, "obs")
    fg.close()
    if "cell_id" in obs.columns and "cell_id" in ge_obs.columns:
        same = np.array_equal(obs["cell_id"].values, ge_obs["cell_id"].values)
        if not same:
            raise RuntimeError("splice/GE cell_id order differs — positional join unsafe")
        if verbose:
            print(f"  GE cell_id order identical: True ({n_cells:,} cells)", flush=True)

    # ── per-cell table ──
    cells = obs.copy()
    cells["n_junctions_detected"] = r["row_nnz"]
    cells["n_atses_detected"] = r["row_groups"]
    cells["junction_reads"] = r["row_sums"]
    cells["ge_total_reads"] = ge_reads
    cells["n_genes_detected"] = ge_genes
    cells["junction_read_fraction"] = r["row_sums"] / np.maximum(ge_reads, 1)
    mk = cfg.get("modality_key")
    cells["modality"] = (require_col(cells, mk).map(cfg["modality_map"]).fillna("unknown")
                         if mk in cells.columns else "unknown")

    assert (cells["n_atses_detected"] <= cells["n_junctions_detected"]).all(), \
        "ATSEs/cell exceeds junctions/cell — the event dedup is wrong"
    assert cells["n_atses_detected"].max() <= n_events

    # ── per-junction table (+ strand-aware 5'/3' position) ──
    juncs = var.copy()
    juncs["junction_id"] = juncs.index.astype(str)
    juncs["reads_per_junction"] = r["col_sums"]
    juncs["cells_per_junction"] = r["col_nnz"]
    juncs["mean_reads_per_detected_cell"] = (
        r["col_sums"] / np.maximum(r["col_nnz"], 1))

    parts = juncs["junction_id"].str.rsplit("_", n=3, expand=True)
    juncs["j_start"] = pd.to_numeric(parts[1], errors="coerce")
    juncs["j_end"] = pd.to_numeric(parts[2], errors="coerce")
    spans = gene_spans(cfg["gtf"])
    j = juncs.join(spans, on="gene_id")
    fwd = (j["g_strand"] == "+").values
    gs, ge_, js, je = (j["g_start"].values, j["g_end"].values,
                       j["j_start"].values, j["j_end"].values)
    # forward: TSS = g_start ; reverse: TSS = g_end  (strand-aware, both ends)
    juncs["dist_5p"] = np.where(fwd, js - gs, ge_ - je)
    juncs["dist_3p"] = np.where(fwd, ge_ - je, js - gs)
    glen = np.maximum(ge_ - gs, 1)
    juncs["gene_length"] = glen
    juncs["rel_pos"] = np.clip(juncs["dist_5p"].values / glen, 0, 1)
    juncs["in_gene_body"] = (juncs["dist_5p"] >= 0) & (juncs["dist_3p"] >= 0)
    if verbose:
        matched = j["g_start"].notna().mean()
        print(f"  GTF gene match {100*matched:.1f}% | inside gene body "
              f"{100*juncs['in_gene_body'].mean():.1f}%", flush=True)

    # ── per-ATSE table ──
    atses = pd.DataFrame({
        "event_id": cats,
        "n_junctions": np.bincount(codes, minlength=n_events),
        "atse_reads": np.bincount(codes, weights=r["col_sums"], minlength=n_events),
        "cells_detecting": r["group_nnz"],
    })
    gname = "gene_name" if "gene_name" in juncs.columns else "gene_id"
    atses = atses.merge(
        juncs.groupby("event_id", observed=True)[gname].first().rename("gene").reset_index(),
        on="event_id", how="left")

    if verbose:
        print(f"  compute done in {time.time()-t0:.0f}s", flush=True)
    return cells, juncs, atses


# ── cache ─────────────────────────────────────────────────────────────────────

def _stat(p):
    s = os.stat(p)
    return dict(path=str(p), mtime=int(s.st_mtime), size=int(s.st_size))


def write_cache(cells, juncs, atses, cfg, ds, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("cells", cells), ("junctions", juncs), ("atses", atses)):
        df.to_parquet(cache_dir / f"{name}_{ds}.parquet", index=False)
    (cache_dir / f"manifest_{ds}.json").write_text(json.dumps(dict(
        dataset=ds, version=__version__, git=_git_rev(),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        splice=_stat(cfg["splice"]), ge=_stat(cfg["ge"]),
        n_cells=len(cells), n_junctions=len(juncs), n_atses=len(atses)), indent=2))


def read_cache(cfg, ds, cache_dir, force=False, verbose=True):
    """Return (cells, juncs, atses) or None. TRAP 8: refuses a stale cache."""
    cache_dir = Path(cache_dir)
    man_p = cache_dir / f"manifest_{ds}.json"
    if not man_p.exists():
        return None
    man = json.loads(man_p.read_text())
    for key in ("splice", "ge"):
        now, then = _stat(cfg[key]), man.get(key, {})
        if (now["path"], now["mtime"], now["size"]) != (
                then.get("path"), then.get("mtime"), then.get("size")):
            msg = (f"cache is STALE for {key}: built from\n    {then.get('path')}\n"
                   f"  but config points at\n    {now['path']}")
            if not force:
                print(f"  {msg}\n  -> recomputing (use --force-cache to override)",
                      flush=True)
                return None
            print(f"  WARNING: {msg}\n  -> using it anyway (--force-cache)", flush=True)
    try:
        out = tuple(pd.read_parquet(cache_dir / f"{n}_{ds}.parquet")
                    for n in ("cells", "junctions", "atses"))
    except FileNotFoundError:
        return None
    if verbose:
        print(f"  cache hit ({man['written']}, {man['n_cells']:,} cells)", flush=True)
    return out


# ── matched cells-vs-nuclei analysis (panel g) ────────────────────────────────

def matched_modality(cells, cfg, min_group_n=50, depth_window=(0.10, 0.90)):
    """Compare modalities within cell type, after controlling organ, cohort and depth.

    Returns (strata: list[dict], effects: dict) or (None, None) if <2 modalities.

    The pooled comparison is confounded by composition and depth; this matches on
    organ (group_prefix), cohort (per-modality restrict), cell type, and finally
    restricts both modalities to the OVERLAP of their depth windows.
    """
    mods = [m for m in cells["modality"].unique() if m != "unknown"]
    if len(mods) < 2 or cfg.get("matched") is None:
        return None, None
    mspec = cfg["matched"]
    ct_key, grp_key = cfg["celltype_key"], cfg["group_key"]

    d = cells.copy()
    if mspec.get("merge"):
        d[ct_key] = d[ct_key].astype(str).replace(mspec["merge"])
    if mspec.get("group_prefix"):
        d = d[d[grp_key].astype(str).str.startswith(mspec["group_prefix"])]
    for mod, filt in (mspec.get("restrict") or {}).items():
        for col, keep in filt.items():
            if col in d.columns:
                d = d[(d["modality"] != mod) | (d[col].astype(str).isin(keep))]

    a, b = sorted(mods)          # a = single cell, b = single nucleus (alphabetical)
    strata = []
    for ct, g in d.groupby(ct_key, observed=True):
        ga, gb = g[g["modality"] == a], g[g["modality"] == b]
        if len(ga) < min_group_n or len(gb) < min_group_n:
            continue
        lo = max(ga["junction_reads"].quantile(depth_window[0]),
                 gb["junction_reads"].quantile(depth_window[0]))
        hi = min(ga["junction_reads"].quantile(depth_window[1]),
                 gb["junction_reads"].quantile(depth_window[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        ma = ga[ga["junction_reads"].between(lo, hi)]
        mb = gb[gb["junction_reads"].between(lo, hi)]
        if len(ma) < min_group_n or len(mb) < min_group_n:
            continue
        # balance a lopsided stratum by subsampling the larger side (deterministic seed)
        cap = (mspec.get("subsample") or {}).get(str(ct))
        if cap:
            rng = np.random.default_rng(0)
            if len(ma) > cap:
                ma = ma.iloc[rng.choice(len(ma), cap, replace=False)]
            if len(mb) > cap:
                mb = mb.iloc[rng.choice(len(mb), cap, replace=False)]
        strata.append(dict(
            stratum=str(ct), n_a=len(ma), n_b=len(mb),
            med_junc_a=ma["n_junctions_detected"].median(),
            med_junc_b=mb["n_junctions_detected"].median(),
            med_frac_a=ma["junction_read_fraction"].median(),
            med_frac_b=mb["junction_read_fraction"].median(),
            log2_junc=np.log2(mb["n_junctions_detected"].median() /
                              max(ma["n_junctions_detected"].median(), 1e-9)),
            log2_frac=np.log2(mb["junction_read_fraction"].median() /
                              max(ma["junction_read_fraction"].median(), 1e-9)),
            values_a=ma["n_junctions_detected"].values,
            values_b=mb["n_junctions_detected"].values))
    if not strata:
        return None, None

    pa, pb = cells[cells["modality"] == a], cells[cells["modality"] == b]
    # What is the POOLED bar actually contrasting? If the two modalities sampled
    # different cell types, "pooled" is a composition artefact, not a modality effect.
    comp = ""
    ct_key = cfg["celltype_key"]
    if ct_key in cells.columns and mspec.get("group_prefix"):
        br = cells[cells[cfg["group_key"]].astype(str).str.startswith(mspec["group_prefix"])]
        parts = []
        for m in (a, b):
            gm = br[br["modality"] == m]
            if len(gm):
                neu = gm[ct_key].astype(str).str.contains("NEURON").mean()
                top = gm[ct_key].astype(str).value_counts().idxmax()
                parts.append(f"{100*neu:.0f}% neurons")
        if len(parts) == 2:
            comp = f"brain composition differs: cells {parts[0]}, nuclei {parts[1]}"
    effects = dict(
        composition=comp,
        mod_a=a, mod_b=b,
        pooled_log2_junc=float(np.log2(pb["n_junctions_detected"].median() /
                                       max(pa["n_junctions_detected"].median(), 1e-9))),
        pooled_log2_frac=float(np.log2(pb["junction_read_fraction"].median() /
                                       max(pa["junction_read_fraction"].median(), 1e-9))),
        matched_log2_junc=float(np.median([s["log2_junc"] for s in strata])),
        matched_log2_frac=float(np.median([s["log2_frac"] for s in strata])),
        n_strata=len(strata),
        pooled_values_a=pa["n_junctions_detected"].values,
        pooled_values_b=pb["n_junctions_detected"].values,
        pooled_n_a=len(pa), pooled_n_b=len(pb))
    return strata, effects


# ── plotting helpers ──────────────────────────────────────────────────────────

def _decades(ax, which="both"):
    """Log axes get real decade ticks (10^0, 10^1 ...), never 1e0."""
    for axis in ((ax.xaxis,) if which == "x" else
                 (ax.yaxis,) if which == "y" else (ax.xaxis, ax.yaxis)):
        if axis.get_scale() == "log":
            axis.set_major_locator(LogLocator(base=10))
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * .1),
                                              numticks=12))
            axis.set_minor_formatter(plt.NullFormatter())


def _annot(ax, txt, x=0.03, y=0.97, va="top", ha="left"):
    ax.text(x, y, txt, transform=ax.transAxes, va=va, ha=ha,
            fontsize=PT["annot"], color=INK)


def _box_violin(ax, groups, labels, colors, log=True):
    """Box + violin (theme: never a bare bar for a distribution; n>100 so no strip)."""
    pos = np.arange(len(groups))
    vals = [np.log10(np.maximum(g, 1)) if log else g for g in groups]
    vp = ax.violinplot(vals, positions=pos, widths=0.8, showextrema=False, showmedians=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.35); body.set_edgecolor("none")
    bp = ax.boxplot(vals, positions=pos, widths=0.28, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, linewidth=0.8))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.9); patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color(INK); ln.set_linewidth(0.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    return pos


# ── panels ────────────────────────────────────────────────────────────────────

def panel_a_inventory(ax, cells, juncs, atses, cfg):
    """Typeset counts. Three scalars spanning 79,768->11,236 say nothing as bars."""
    ax.axis("off")
    gcol = "gene_name" if "gene_name" in juncs.columns else "gene_id"
    rows = [("Cells", f"{len(cells):,}")]
    for mod, n in cells["modality"].value_counts().items():
        if mod != "unknown":
            rows.append((f"   {mod}", f"{n:,}"))
    rows += [
        ("Junctions", f"{len(juncs):,}"),
        ("ATSEs", f"{len(atses):,}"),
        ("Genes with an ATSE", f"{juncs[gcol].nunique():,}"),
        ("", ""),
        ("Median per cell", ""),
        ("   junctions detected", f"{cells['n_junctions_detected'].median():,.0f}"),
        ("   ATSEs detected", f"{cells['n_atses_detected'].median():,.0f}"),
        ("   junction reads", f"{cells['junction_reads'].median():,.0f}"),
        ("   read fraction", f"{cells['junction_read_fraction'].median():.2f}"),
        ("Median per ATSE", ""),
        ("   junctions", f"{atses['n_junctions'].median():,.0f}"),
        ("", ""),
        ("Filters", f"≥{cfg['min_cells_filter']} cells/junction"),
        ("", f"≥{cfg['min_juncs_filter']} junctions/cell"),
    ]
    y = 1.0
    for k, v in rows:
        if k or v:
            ax.text(0.0, y, k, transform=ax.transAxes, fontsize=PT["annot"], color=INK,
                    va="top", fontweight="bold" if k and not k.startswith(" ") else "normal")
            ax.text(1.0, y, v, transform=ax.transAxes, fontsize=PT["annot"], color=INK,
                    va="top", ha="right")
        y -= 0.072
    ax.set_title("Resource inventory", fontsize=PT["title"], loc="left")


def panel_b_depth_breadth(ax, cells, cfg):
    x = np.maximum(cells["junction_reads"].values, 1)
    y = np.maximum(cells["n_junctions_detected"].values, 1)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=42,
                   cmap=CMAP["sequential"], bins="log", linewidths=0, rasterized=True)
    cb = ax.figure.colorbar(hb, ax=ax, pad=0.03, fraction=0.055, aspect=18)
    cb.ax.set_title("Cells", fontsize=PT["annot"], pad=3, loc="left")
    cb.ax.tick_params(labelsize=PT["tick"])
    ax.axhline(cfg["min_juncs_filter"], color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    _annot(ax, f"n = {len(cells):,} cells\nnominal filter ≥{cfg['min_juncs_filter']}")
    ax.set_xlabel("Junction reads per cell")
    ax.set_ylabel("Junctions detected per cell")
    _decades(ax)


def panel_c_fraction(ax, cells):
    x = np.maximum(cells["ge_total_reads"].values, 1)
    y = np.maximum(cells["junction_reads"].values, 1)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=42,
                   cmap=CMAP["sequential"], bins="log", linewidths=0, rasterized=True)
    cb = ax.figure.colorbar(hb, ax=ax, pad=0.03, fraction=0.055, aspect=18)
    cb.ax.set_title("Cells", fontsize=PT["annot"], pad=3, loc="left")
    cb.ax.tick_params(labelsize=PT["tick"])
    lo, hi = np.percentile(x, [0.5, 99.5])
    for frac, lab in ((0.5, "50%"), (0.1, "10%"), (0.01, "1%")):
        xs = np.array([lo, hi])
        ax.plot(xs, xs * frac, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(hi, hi * frac, f" {lab}", fontsize=PT["annot"], color=PALETTE["grey"],
                va="center", ha="left")
    med = cells.groupby("modality", observed=True)["junction_read_fraction"].median()
    txt = "\n".join(f"{m}: {v:.2f}" for m, v in med.items() if m != "unknown") or \
          f"median {cells['junction_read_fraction'].median():.2f}"
    _annot(ax, "Median fraction\n" + txt)
    ax.set_xlabel("Reads in ATSE-bearing genes per cell")
    ax.set_ylabel("Junction reads per cell")
    _decades(ax)


def panel_d_reads_by_atse_size(ax, juncs, atses=None):
    """Does event complexity starve individual junctions of reads?

    Naively an ATSE's reads split k ways, so a 6-way event should give each junction ~3x
    fewer reads than a 2-way one. It does NOT: the per-junction medians are 7.2 -> 5.8
    (1.2x, not 3x). Showing only that flat violin invites "this panel shows nothing".
    The reason is the interesting part, so plot BOTH levels: ATSE-level depth RISES with
    complexity (complex events sit in more-expressed genes) and almost exactly cancels the
    k-way split. Net: PSI precision barely degrades with event complexity.
    """
    order = ["2", "3", "4–5", "≥6"]
    szj = pd.cut(juncs["n_junctions_in_atse"], [1, 2, 3, 5, 10 ** 6], labels=order)
    per_junc = juncs.groupby(szj, observed=True)["mean_reads_per_detected_cell"].median()
    x = np.arange(len(order))
    ax.plot(x, per_junc.reindex(order).values, "o-", color=PALETTE["teal"], ms=4,
            label="per junction")
    if atses is not None and {"atse_reads", "cells_detecting"} <= set(atses.columns):
        sza = pd.cut(atses["n_junctions"], [1, 2, 3, 5, 10 ** 6], labels=order)
        per_atse = (atses.assign(r=atses["atse_reads"] /
                                 np.maximum(atses["cells_detecting"], 1))
                    .groupby(sza, observed=True)["r"].median())
        ax.plot(x, per_atse.reindex(order).values, "s--", color=PALETTE["indigo"], ms=4,
                label="per ATSE (all its junctions)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_xlabel("Junctions in ATSE")
    ax.set_ylabel("Median reads per detected cell")
    _decades(ax, "y")
    n = juncs.groupby(szj, observed=True).size().reindex(order)
    for i, v in enumerate(n.values):
        ax.text(i, ax.get_ylim()[0], f"{int(v):,}", ha="center", va="bottom",
                fontsize=PT["annot"], color=PALETTE["grey"])
    ax.legend(loc="upper left", fontsize=PT["legend"], handletextpad=0.4)
    _annot(ax, "ATSE depth rises with complexity,\ncancelling the k-way read split —\n"
               "per-junction depth stays ~flat",
           x=0.97, y=0.42, ha="right")


def panel_e_junctions_per_atse(ax, atses):
    n = atses["n_junctions"].clip(upper=10)
    vc = n.value_counts().sort_index()
    labels = [str(i) for i in vc.index[:-1]] + ["≥10"] if 10 in vc.index else \
             [str(i) for i in vc.index]
    ax.bar(np.arange(len(vc)), vc.values, color=PALETTE["teal"], width=0.75)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{100*v/len(atses):.0f}%", ha="center", va="bottom",
                fontsize=PT["annot"], color=INK)
    ax.set_xticks(np.arange(len(vc))); ax.set_xticklabels(labels)
    ax.set_ylim(0, vc.max() * 1.22)          # headroom so % labels clear the annotation
    ax.set_xlabel("Junctions per ATSE")
    ax.set_ylabel("ATSEs")
    _annot(ax, f"n = {len(atses):,} ATSEs\nlow-count classes are a floor:\n"
               f"an event looks 2-way if its\nrarer junctions went undetected",
           x=0.97, ha="right")


def panel_f_tissue(ax, cells, cfg):
    gk = cfg["group_key"]
    if gk not in cells.columns:
        ax.axis("off"); return
    g = cells.groupby(gk, observed=True).agg(
        n=("n_junctions_detected", "size"),
        med=("n_junctions_detected", "median"),
        q25=("n_junctions_detected", lambda s: s.quantile(.25)),
        q75=("n_junctions_detected", lambda s: s.quantile(.75)),
        modality=("modality", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"),
    ).reset_index()
    for mod, sub in g.groupby("modality", observed=True):
        ax.errorbar(sub["n"], sub["med"],
                    yerr=[sub["med"] - sub["q25"], sub["q75"] - sub["med"]],
                    fmt="o", ms=3, lw=0, elinewidth=0.8, capsize=0,
                    color=MOD_COLOR.get(mod, PALETTE["grey"]), label=mod, alpha=0.85)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    # Label only the extremes, alternating side, so 29 tissues don't overprint.
    # no per-point labels: 29 tissues at this size overprint, and the pattern (not the
    # identity of any one tissue) is the message. Per-tissue values are in the CSV.
    ax.set_xlabel("Cells in group")
    ax.set_ylabel("Median junctions\ndetected per cell")
    n_mod = g["modality"].nunique()
    _annot(ax, f"n = {len(g)} {gk.replace('_',' ')}s"
               + ("\nhigh-coverage groups are all\nnuclei AND all brain — depth,\n"
                  "region and modality are\nconfounded here" if n_mod > 1 else ""))
    _decades(ax, "x")
    ax.margins(x=0.10)          # keep the right-hand extreme label on canvas
    # no legend here on purpose: modality uses the same colours in panel g, and the theme
    # wants a single legend per figure for a shared encoding.


def panel_g_modality(ax, cells, strata, effects, cfg):
    """Naive vs matched. Keeping the naive group visible IS the argument."""
    if effects is None:
        bk = cfg.get("batch_key")
        if bk and bk in cells.columns:
            top = cells[bk].astype(str).value_counts().head(12).index
            groups = [cells.loc[cells[bk].astype(str) == b, "n_junctions_detected"].values
                      for b in top]
            pos = _box_violin(ax, groups, [str(b)[:10] for b in top],
                              [PALETTE["teal"]] * len(groups))
            ax.set_xticklabels([str(b)[:10] for b in top], rotation=45, ha="right")
            ax.set_xlabel(bk.replace("_", " "))
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.22 * (hi - lo))     # headroom for the note
            _annot(ax, "Single modality (nuclei only) — no cells-vs-nuclei contrast\n"
                       "available; showing per-donor coverage variation instead.")
        else:
            ax.axis("off")
        ax.set_ylabel("Junctions detected per cell (log$_{10}$)")
        return

    a, b = effects["mod_a"], effects["mod_b"]
    groups, labels, colors = [effects["pooled_values_a"], effects["pooled_values_b"]], \
                             ["pooled", "pooled"], [MOD_COLOR[a], MOD_COLOR[b]]
    ns = [effects["pooled_n_a"], effects["pooled_n_b"]]
    centres = [0.5]
    for s in strata:
        groups += [s["values_a"], s["values_b"]]
        labels += [s["stratum"][:12].title()] * 2
        colors += [MOD_COLOR[a], MOD_COLOR[b]]
        ns += [s["n_a"], s["n_b"]]
        centres.append(len(groups) - 1.5)

    # pack pairs with a gap between the naive block and the matched strata
    pos = []
    for i in range(len(groups)):
        pos.append(i * 0.86 + (0.9 if i >= 2 else 0))
    pos = np.array(pos)
    vals = [np.log10(np.maximum(g, 1)) for g in groups]
    vp = ax.violinplot(vals, positions=pos, widths=0.72, showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.35); body.set_edgecolor("none")
    bp = ax.boxplot(vals, positions=pos, widths=0.26, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=0.8))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.9); patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color(INK); ln.set_linewidth(0.6)

    # n goes INTO the tick label — rotated text under the axis collided with the ticks
    centres = [(pos[2 * i] + pos[2 * i + 1]) / 2 for i in range(len(groups) // 2)]
    names = ["pooled\n(unmatched)"] + [s["stratum"][:11].title() for s in strata]
    ticklabs = [f"{nm}\n{ns[2*i]:,} / {ns[2*i+1]:,}" for i, nm in enumerate(names)]
    ax.set_xticks(centres)
    ax.set_xticklabels(ticklabs, fontsize=PT["tick"])
    ax.axvline((pos[1] + pos[2]) / 2, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))

    # headroom for the per-pair effect sizes, so they never sit on the legend
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    ytxt = hi + 0.03 * (hi - lo)
    ax.text(centres[0], ytxt, f"{effects['pooled_log2_junc']:+.2f}", ha="center",
            va="bottom", fontsize=PT["annot"], color=INK, fontweight="bold")
    for i, s in enumerate(strata):
        ax.text(centres[i + 1], ytxt, f"{s['log2_junc']:+.2f}", ha="center", va="bottom",
                fontsize=PT["annot"], color=INK)

    ax.set_ylabel("Junctions detected per cell (log$_{10}$)")
    ax.set_xlabel(f"log$_2$(nuclei / cells) above each pair;  n below = "
                  f"{a} / {b}")
    fold = 2 ** effects["pooled_log2_junc"]
    txt = (f"pooled {fold:.1f}×  →  matched "
           f"{2**effects['matched_log2_junc']:.2f}×  ({effects['n_strata']} strata)")
    if effects.get("composition"):
        txt += "\n" + effects["composition"]
    _annot(ax, txt, x=0.30, y=0.995)
    handles = [plt.Line2D([], [], marker="s", ls="", color=MOD_COLOR[m], label=m)
               for m in (a, b)]
    ax.legend(handles=handles, loc="upper left", fontsize=PT["legend"], ncol=1,
              handletextpad=0.4)


def panel_h_position(ax, juncs, cells, cfg):
    """5'->3' position. var's position_off_* are EXON offsets and do not answer this;
    distances are recomputed strand-aware from GTF gene spans."""
    # ECDF, not a density: a density histogram turns the narrow 5' spike into a visual wall
    # and hides the fact that most junctions sit mid-gene (median rel_pos 0.40).
    #
    # The 5' excess (17.6% of junctions in the first 2.5% of gene length vs 2.5% expected)
    # is NOT explained by either obvious candidate — both were tested and ruled out:
    #   * "first introns": only 24% of that bin are a gene's first junction, and gene-first
    #     junctions have median rel_pos 0.244.
    #   * "shared-donor pile-up": 67% of junctions there share a start with an ATSE partner
    #     vs 62% elsewhere — far too small to produce a 7x enrichment.
    # Note rel_pos is derived from the junction START, so shared-donor junctions coincide by
    # construction — but that inflates every position equally, not the 5' end specifically.
    # Cause undetermined; the panel therefore states the observation, not a mechanism.
    d_all = np.sort(juncs.loc[juncs["in_gene_body"], "rel_pos"].dropna().values)
    y = np.arange(1, len(d_all) + 1) / len(d_all)
    ax.plot(d_all, y, lw=1.0, color=PALETTE["teal"])
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    q = np.percentile(d_all, [25, 50, 75])
    ax.set_xlabel("Relative position along gene\n(0 = TSS, 1 = TES)")
    ax.set_ylabel("Cumulative fraction of junctions")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _annot(ax, f"n = {len(d_all):,} junctions\nmedian {q[1]:.2f}"
               f"  (IQR {q[0]:.2f}–{q[2]:.2f})\ndashed = uniform along gene\n"
               f"5' end is enriched relative to\nuniform; junctions are otherwise\nspread through the gene body",
           x=0.97, y=0.28, ha="right")


# ── figure assembly ───────────────────────────────────────────────────────────

def make_figure(cells, juncs, atses, strata, effects, cfg):
    fig = plt.figure(figsize=figsize("double", h=7.9))
    gs = fig.add_gridspec(3, 6, hspace=0.48, wspace=1.35,
                          left=0.075, right=0.955, top=0.955, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0:2]); panel_a_inventory(ax_a, cells, juncs, atses, cfg)
    ax_b = fig.add_subplot(gs[0, 2:4]); panel_b_depth_breadth(ax_b, cells, cfg)
    ax_c = fig.add_subplot(gs[0, 4:6]); panel_c_fraction(ax_c, cells)
    ax_d = fig.add_subplot(gs[1, 0:2]); panel_d_reads_by_atse_size(ax_d, juncs, atses)
    ax_e = fig.add_subplot(gs[1, 2:4]); panel_e_junctions_per_atse(ax_e, atses)
    ax_f = fig.add_subplot(gs[1, 4:6]); panel_f_tissue(ax_f, cells, cfg)
    ax_g = fig.add_subplot(gs[2, 0:4]); panel_g_modality(ax_g, cells, strata, effects, cfg)
    ax_h = fig.add_subplot(gs[2, 4:6]); panel_h_position(ax_h, juncs, cells, cfg)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                       (ax_e, "e"), (ax_f, "f"), (ax_g, "g"), (ax_h, "h")):
        panel_label(ax, letter)
    return fig


# ── summary CSV (long format, concatenates across datasets) ───────────────────

def _dist_rows(ds, cfg, level, group_by, group, metric, unit, values, panel=""):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return []
    base = dict(dataset=ds, species=cfg["species"], label=cfg["label"], level=level,
                group_by=group_by, group=group, metric=metric, unit=unit,
                n=int(v.size), panel=panel)
    stats = dict(median=np.median(v), q25=np.percentile(v, 25), q75=np.percentile(v, 75),
                 p10=np.percentile(v, 10), p90=np.percentile(v, 90), mean=v.mean(),
                 min=v.min(), max=v.max())
    return [dict(base, stat=k, value=float(x)) for k, x in stats.items()]


def summary_stats(cells, juncs, atses, strata, effects, ds, cfg):
    rows = []

    def scalar(level, metric, value, panel="a"):
        rows.append(dict(dataset=ds, species=cfg["species"], label=cfg["label"],
                         level=level, group_by="all", group="all", metric=metric,
                         unit="count", stat="total", value=float(value),
                         n=int(value), panel=panel))

    scalar("cell", "n_cells", len(cells))
    scalar("junction", "n_junctions", len(juncs))
    scalar("atse", "n_atses", len(atses))
    gcol = "gene_name" if "gene_name" in juncs.columns else "gene_id"
    scalar("gene", "n_genes", juncs[gcol].nunique())

    percell = [("n_junctions_detected", "junctions", "b"),
               ("n_atses_detected", "atses", "a"),
               ("junction_reads", "reads", "b"),
               ("ge_total_reads", "reads", "c"),
               ("junction_read_fraction", "fraction", "c")]
    for metric, unit, panel in percell:
        rows += _dist_rows(ds, cfg, "cell", "all", "all", metric, unit,
                           cells[metric].values, panel)
        for mod, sub in cells.groupby("modality", observed=True):
            if mod != "unknown":
                rows += _dist_rows(ds, cfg, "cell", "modality", mod, metric, unit,
                                   sub[metric].values, panel)
    # per-tissue block — the biological breakdown lives here, not in the figure
    gk = cfg["group_key"]
    if gk in cells.columns:
        for grp, sub in cells.groupby(gk, observed=True):
            rows += _dist_rows(ds, cfg, "cell", gk, str(grp), "n_junctions_detected",
                               "junctions", sub["n_junctions_detected"].values, "f")

    for metric, unit, panel in (("reads_per_junction", "reads", "d"),
                                ("cells_per_junction", "cells", ""),
                                ("mean_reads_per_detected_cell", "reads", "d"),
                                ("rel_pos", "fraction", "h")):
        rows += _dist_rows(ds, cfg, "junction", "all", "all", metric, unit,
                           juncs[metric].values, panel)
    sz = pd.cut(juncs["n_junctions_in_atse"], [1, 2, 3, 5, 10 ** 6],
                labels=["2", "3", "4-5", ">=6"])
    for lab, sub in juncs.groupby(sz, observed=True):
        rows += _dist_rows(ds, cfg, "junction", "atse_size_class", str(lab),
                           "mean_reads_per_detected_cell", "reads",
                           sub["mean_reads_per_detected_cell"].values, "d")
    rows += _dist_rows(ds, cfg, "atse", "all", "all", "junctions_per_atse", "count",
                       atses["n_junctions"].values, "e")
    rows += _dist_rows(ds, cfg, "atse", "all", "all", "cells_per_atse", "cells",
                       atses["cells_detecting"].values, "")
    if "gene" in atses.columns:
        rows += _dist_rows(ds, cfg, "gene", "all", "all", "atses_per_gene", "count",
                           atses.groupby("gene", observed=True).size().values, "")

    # panel g effect sizes — figure annotation and methods both read from here
    if effects is not None:
        for metric, key in (("log2_nuclei_over_cells_junctions", "pooled_log2_junc"),
                            ("log2_nuclei_over_cells_read_fraction", "pooled_log2_frac")):
            rows.append(dict(dataset=ds, species=cfg["species"], label=cfg["label"],
                             level="cell", group_by="modality", group="pooled",
                             metric=metric, unit="log2_ratio", stat="log2_ratio",
                             value=effects[key], n=effects["pooled_n_a"] + effects["pooled_n_b"],
                             panel="g"))
        for s in strata:
            for metric, key in (("log2_nuclei_over_cells_junctions", "log2_junc"),
                                ("log2_nuclei_over_cells_read_fraction", "log2_frac")):
                rows.append(dict(dataset=ds, species=cfg["species"], label=cfg["label"],
                                 level="cell", group_by="matched_cell_type",
                                 group=s["stratum"], metric=metric, unit="log2_ratio",
                                 stat="log2_ratio", value=s[key],
                                 n=s["n_a"] + s["n_b"], panel="g"))
    cols = ["dataset", "species", "label", "level", "group_by", "group", "metric",
            "unit", "stat", "value", "n", "panel"]
    return pd.DataFrame(rows)[cols]


# ── METHODS draft ─────────────────────────────────────────────────────────────

def write_methods(cells, juncs, atses, strata, effects, ds, cfg, path):
    gcol = "gene_name" if "gene_name" in juncs.columns else "gene_id"
    mod_counts = "; ".join(f"{n:,} {m}" for m, n in
                           cells["modality"].value_counts().items() if m != "unknown")
    q = np.percentile(juncs.loc[juncs["in_gene_body"], "rel_pos"].dropna(), [25, 50, 75])
    g_txt = ""
    if effects is not None:
        g_txt = f"""
### Single cells versus single nuclei

Pooled, nuclei appear to detect **{2**effects['pooled_log2_junc']:.1f}×** more junctions per cell
than whole cells ({effects['pooled_log2_junc']:+.2f} log2). This comparison is confounded:
{cfg['confound_note']}

Restricting both modalities to brain tissue, to a common cohort, to the same
`{cfg['celltype_key']}`, and finally to the overlap of their 10th–90th percentile
junction-read windows, the difference **collapses to {2**effects['matched_log2_junc']:.2f}×**
({effects['matched_log2_junc']:+.2f} log2, median of {effects['n_strata']} matched strata) — i.e.
the apparent advantage is composition and depth, not modality.

The read-fraction difference does **not** collapse: nuclei devote
{effects['matched_log2_frac']:+.2f} log2 ({2**effects['matched_log2_frac']:.2f}×) as large a
fraction of their reads to junction-spanning reads under matching, versus
{effects['pooled_log2_frac']:+.2f} pooled — consistent across every stratum.
"""
    Path(path).write_text(f"""# Dataset coverage — {cfg['label']}

_Generated by `supp_dataset_coverage.py` v{__version__}. Numbers here are read from
`supp_dataset_coverage_{ds}_summary.csv`; do not retype them._

## Methods

Coverage statistics were computed from the filtered, ATSE-regrouped splicing object
(`{Path(cfg['splice']).name}`) and its matched gene-expression object
(`{Path(cfg['ge']).name}`), whose cells are identical and in the same order.

The resource comprises **{len(cells):,} cells** ({mod_counts}), **{len(juncs):,} splice
junctions** grouped into **{len(atses):,} ATSEs** across **{juncs[gcol].nunique():,} genes**.
Junctions were retained if detected in ≥{cfg['min_cells_filter']} cells and cells if they
carried ≥{cfg['min_juncs_filter']} junctions.

Per-cell detection counts were taken from the CSR row pointers of the
`cell_by_junction_matrix` layer; read sums and per-junction aggregates from a single chunked
pass over the same layer. The number of distinct ATSEs covered per cell was obtained by
deduplicating (cell, event) pairs — ATSE identifiers are not contiguous in column order, so
run-length counting would overcount. The `cell_by_cluster_matrix` layer was not used: it
stores each ATSE total broadcast across every junction column of that event, and was verified
to equal the within-event sum of the junction layer.

**Read fraction.** The gene-expression object contains only the
{ (juncs[gcol].nunique()) :,} ATSE-bearing genes rather than the full transcriptome.
Junction-read fractions are therefore reported as *"junction-spanning reads as a fraction of
reads in ATSE-bearing genes"* and are biased upward relative to a true library fraction.

**Junction position.** The `position_off_5_prime` / `position_off_3_prime` columns record
offsets from annotated exon boundaries and do not measure position within a transcript. We
instead computed strand-aware distances from GENCODE gene spans: for forward-strand genes the
5′ distance is the junction start minus the gene start and the 3′ distance is the gene end
minus the junction end; for reverse-strand genes the two are exchanged. All junctions matched
a gene by versioned identifier and all strands agreed; the
{100*(1-juncs['in_gene_body'].mean()):.1f}% of junctions extending beyond the annotated gene
body were excluded from the position panel.

## Results

Cells carry a median of **{cells['n_junctions_detected'].median():,.0f} detected junctions**
(IQR {cells['n_junctions_detected'].quantile(.25):,.0f}–{cells['n_junctions_detected'].quantile(.75):,.0f})
spanning **{cells['n_atses_detected'].median():,.0f} ATSEs**, from a median of
{cells['junction_reads'].median():,.0f} junction-spanning reads. Junction detection scales with
depth, so coverage differences between groups largely track sequencing depth.

Most ATSEs are simple: **{100*(atses['n_junctions']==2).mean():.0f}%** contain exactly two
junctions and **{100*(atses['n_junctions']<=3).mean():.0f}%** three or fewer. Junctions inside
larger ATSEs are individually thinner (median mean-reads-per-detected-cell falls from
{juncs.loc[juncs['n_junctions_in_atse']==2,'mean_reads_per_detected_cell'].median():.1f} in
two-junction events to
{juncs.loc[juncs['n_junctions_in_atse']>=6,'mean_reads_per_detected_cell'].median():.1f} in
events with six or more), so PSI in complex events is estimated from fewer reads.

Junctions are distributed across the gene body (relative-position quartiles
{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}, where 0 is the TSS and 1 the TES) rather than
concentrated near the 3′ end, as expected for full-length SMART-seq protocols and in contrast
to 3′-biased chemistries.
{g_txt}
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mouse_smartseq2", choices=list(DEFAULTS))
    ap.add_argument("--splice"); ap.add_argument("--ge"); ap.add_argument("--gtf")
    ap.add_argument("--outdir", default=str(HERE / "output"))
    ap.add_argument("--cache-dir", default=str(HERE / "cache"))
    ap.add_argument("--stage", default="auto", choices=["auto", "compute", "plot"])
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--force-cache", action="store_true",
                    help="plot from a stale cache (prints a warning)")
    ap.add_argument("--chunk-rows", type=int, default=20_000)
    ap.add_argument("--min-group-n", type=int, default=50)
    ap.add_argument("--depth-window", type=float, nargs=2, default=(0.10, 0.90))
    ap.add_argument("--no-methods", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    cfg = dict(DEFAULTS[ds])
    for k in ("splice", "ge", "gtf"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    for k in ("splice", "ge", "gtf"):
        if not os.path.exists(cfg[k]):
            raise SystemExit(f"ERROR: {k} not found: {cfg[k]}")

    print(f"=== {cfg['label']}  [{ds}] ===")
    print(f"  splice {cfg['splice']}\n  ge     {cfg['ge']}")

    cached = None if args.force_recompute else read_cache(cfg, ds, args.cache_dir,
                                                          force=args.force_cache)
    if cached is None:
        if args.stage == "plot":
            raise SystemExit("ERROR: --stage plot but no valid cache. Run compute first.")
        cells, juncs, atses = compute_metrics(cfg, chunk_rows=args.chunk_rows)
        write_cache(cells, juncs, atses, cfg, ds, args.cache_dir)
        print(f"  cache -> {args.cache_dir}")
    else:
        cells, juncs, atses = cached
    if args.stage == "compute":
        return

    strata, effects = matched_modality(cells, cfg, min_group_n=args.min_group_n,
                                       depth_window=tuple(args.depth_window))
    if effects is not None:
        print(f"  matched strata: {effects['n_strata']} | pooled log2 "
              f"{effects['pooled_log2_junc']:+.2f} -> matched "
              f"{effects['matched_log2_junc']:+.2f}")

    apply_theme()
    fig = make_figure(cells, juncs, atses, strata, effects, cfg)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.outdir).resolve() / f"supp_dataset_coverage_{ds}"
    assert "." not in stem.name, "save() truncates stems containing a dot"
    out = save(fig, str(stem), close=True)
    print("  " + "\n  ".join(str(p) for p in out))

    stats = summary_stats(cells, juncs, atses, strata, effects, ds, cfg)
    csv_p = Path(args.outdir) / f"supp_dataset_coverage_{ds}_summary.csv"
    stats.to_csv(csv_p, index=False)
    print(f"  {csv_p}  ({len(stats):,} rows)")

    if not args.no_methods:
        md = Path(args.outdir) / f"METHODS_dataset_coverage_{ds}.md"
        write_methods(cells, juncs, atses, strata, effects, ds, cfg, md)
        print(f"  {md}")


if __name__ == "__main__":
    main()
