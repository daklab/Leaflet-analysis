#!/usr/bin/env python
"""A58: when a cell puts ALL its reads on one junction, is that biology or PCR duplication?

The observation
---------------
Within ONE cell type (microglia), on ATSEs whose POOLED composition is genuinely mixed (top
junction between 0.2 and 0.8, so alternative splicing is definitely happening), cells are
overwhelmingly all-or-nothing:

    reads in cell    all-on-one-junction    expected if reads were independent
    5-8                    0.632                    0.054
    10-18                  0.532                    0.005
    20-48                  0.434                    0.0001
    50-198                 0.238                    ~0

At 50-198 reads independence predicts essentially never, so a cell's reads are NOT independent
draws from that cell's splicing composition.

Why it matters
--------------
Two explanations, with OPPOSITE implications for the model:

  (a) TECHNICAL -- reads are duplicates of a few cDNA molecules, so a cell looks binary even though
      its true splicing is mixed. Then a LOW concentration is the correct model, and conc_min = 2.0
      is actively harmful: it denies the model the right explanation and pushes duplication into
      phi, manufacturing structure.

  (b) BIOLOGICAL -- cells genuinely commit to one isoform, and the pooled 0.5 is a mixture of
      all-in and all-out CELLS. Then this is exactly phi-shaped and the floor is right.

The discriminator
-----------------
Technical duplication is random per cell per event. Biological commitment is a property of the
CELL, so it must show up as association with something else about that cell.

  1. coverage        does commitment track library size? (pure artefact check)
  2. donor / batch   association with mouse.id or batch means donor or technical structure
  3. age / sex       association with a biological covariate
  4. cross-event     does a cell's choice at event A predict its choice at event B?
                     CONTROLLED for the cell's overall tendency to be extreme -- a cell with few
                     molecules is all-or-nothing EVERYWHERE, which would create association across
                     events under pure duplication. So we condition on committed-at-both and test
                     WHICH junction, not WHETHER extreme.
                     Same-gene pairs are compared against random pairs: biology should link
                     functionally related events more than unrelated ones.
"""

import argparse, os, sys, collections

import h5py
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def cat(f, grp, name):
    x = f[grp][name]
    if isinstance(x, h5py.Group):
        c = np.array([y.decode() if isinstance(y, bytes) else y for y in x["categories"][:]])
        return c[x["codes"][:]]
    return np.array([y.decode() if isinstance(y, bytes) else y for y in x[:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="Brain_Myeloid_MICROGLIA")
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--min-reads", type=int, default=20,
                    help="a cell must have this many reads on the ATSE to count as committed")
    ap.add_argument("--min-cells", type=int, default=60)
    ap.add_argument("--out", default=f"{WD}/ATLAS_DM/a58_commitment_structure.tsv")
    args = ap.parse_args()

    with h5py.File(H5, "r") as f:
        ev = cat(f, "var", "event_id").astype(str)
        gene = cat(f, "var", "gene_name").astype(str)
        lab = cat(f, "obs", "tissue_cell_type").astype(str)
        meta = {k: cat(f, "obs", k).astype(str) for k in ("mouse.id", "batch", "age", "sex")
                if k in f["obs"]}
        g = f["layers/cell_by_junction_matrix"]
        jp, ji, jd = g["indptr"][:], g["indices"][:], g["data"][:]

    uq, inv = np.unique(ev, return_inverse=True)
    nA = len(uq)
    cells = np.flatnonzero(lab == args.celltype)[: args.cells]
    print(f"{args.celltype}: {len(cells):,} cells, {nA:,} ATSEs", flush=True)

    Y = np.zeros((len(cells), len(ev)), np.float32)
    for i, c in enumerate(cells):
        s, e = jp[c], jp[c + 1]
        Y[i, ji[s:e]] = jd[s:e]
    Ne = np.zeros((len(cells), nA))
    np.add.at(Ne, (np.arange(len(cells))[:, None],
                   np.broadcast_to(inv, (len(cells), len(inv)))), Y)
    libsize = Y.sum(1)

    tot_j = Y.sum(0)
    tot_e = np.bincount(inv, weights=tot_j, minlength=nA)
    p = np.where(tot_e[inv] > 0, tot_j / np.maximum(tot_e[inv], 1e-9), 0.0)

    # ---- commitment matrix: for each mixed ATSE, which junction did each cell commit to? -----
    commit = {}          # atse -> (cell_idx, chosen_junction_rank)
    for e in range(nA):
        cols = np.flatnonzero(inv == e)
        if len(cols) < 2 or not (0.2 <= p[cols].max() <= 0.8):
            continue
        n = Ne[:, e]
        k = np.flatnonzero(n >= args.min_reads)
        if len(k) < args.min_cells:
            continue
        y = Y[np.ix_(k, cols)]
        pure = y.max(1) == n[k]                     # ALL reads on one junction
        if pure.sum() < args.min_cells:
            continue
        kk = k[pure]
        commit[e] = (kk, np.argmax(y[pure], axis=1))
    print(f"mixed ATSEs with >= {args.min_cells} committed cells: {len(commit):,}", flush=True)
    if not commit:
        print("nothing to test"); return 1

    # ---- 1-3. is commitment associated with anything about the CELL? ------------------------
    rows = []
    for e, (kk, choice) in commit.items():
        if len(np.unique(choice)) < 2:
            continue
        r = dict(atse=uq[e], n_committed=len(kk),
                 gene=gene[np.flatnonzero(inv == e)[0]])
        # coverage: do the two commitment groups differ in library size?
        a, b = libsize[kk][choice == 0], libsize[kk][choice != 0]
        if len(a) > 5 and len(b) > 5:
            r["p_coverage"] = float(mannwhitneyu(a, b).pvalue)
        for key, vals in meta.items():
            v = vals[cells][kk]
            tab = pd.crosstab(v, choice)
            if tab.shape[0] > 1 and tab.shape[1] > 1 and tab.values.min() >= 0:
                try:
                    r[f"p_{key}"] = float(chi2_contingency(tab.values)[1])
                except ValueError:
                    pass
        rows.append(r)
    R = pd.DataFrame(rows)
    print("\n=== is commitment associated with a cell-level covariate? ===")
    print(f"(fraction of {len(R):,} ATSEs with p < 0.01; ~0.01 = no association)")
    for c in [c for c in R.columns if c.startswith("p_")]:
        v = R[c].dropna()
        if len(v):
            print(f"  {c[2:]:<12} {float((v < 0.01).mean()):.3f}   (n={len(v):,})")

    # ---- 4. cross-event: does the choice at A predict the choice at B? ----------------------
    # Conditioning on committed-at-BOTH removes the 'this cell has few molecules' confound: every
    # cell in the table is extreme at both events, so only WHICH junction can carry signal.
    keys = list(commit)
    cmap = {e: dict(zip(*commit[e])) for e in keys}
    gene_of = {e: gene[np.flatnonzero(inv == e)[0]] for e in keys}
    rng = np.random.default_rng(0)

    def assoc(e1, e2):
        c1, c2 = cmap[e1], cmap[e2]
        shared = np.intersect1d(list(c1), list(c2))
        if len(shared) < 40:
            return None
        x = np.array([c1[c] for c in shared]) > 0
        y = np.array([c2[c] for c in shared]) > 0
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return None
        tab = pd.crosstab(x, y).values
        if tab.shape != (2, 2) or tab.min() < 1:
            return None
        return float(chi2_contingency(tab)[1])

    same_gene, rand_pairs = [], []
    by_gene = collections.defaultdict(list)
    for e in keys:
        by_gene[gene_of[e]].append(e)
    for gname, es in by_gene.items():
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                v = assoc(es[i], es[j])
                if v is not None:
                    same_gene.append(v)
    for _ in range(min(4000, len(keys) * 4)):
        e1, e2 = rng.choice(keys, 2, replace=False)
        if gene_of[e1] == gene_of[e2]:
            continue
        v = assoc(e1, e2)
        if v is not None:
            rand_pairs.append(v)

    print("\n=== does a cell's choice at one event predict its choice at another? ===")
    print("(conditioned on committed at BOTH, so 'few molecules' cannot create the association)")
    for nm, v in (("same gene", same_gene), ("different genes", rand_pairs)):
        v = np.asarray(v)
        if len(v):
            print(f"  {nm:<16} {float((v < 0.01).mean()):.3f} of {len(v):,} pairs at p<0.01")
    print("\nreading: ~0.01 everywhere = commitment is random per cell per event -> TECHNICAL,")
    print("         and a low concentration is the CORRECT model rather than a pathology.")
    print("         Enrichment, especially same-gene > different-genes, means cells genuinely")
    print("         commit -> the structure is real and phi should capture it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R.to_csv(args.out, sep="\t", index=False)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
