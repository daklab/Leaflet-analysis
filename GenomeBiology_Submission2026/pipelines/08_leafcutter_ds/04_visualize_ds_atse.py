#!/usr/bin/env python
"""
Step 08-04: Visualize a LeafCutter DS ATSE as a two-panel ATSEviz-style plot.

Layout (top→bottom):
  Panel A  — junction lines colored by GROUP 2 (old / cell type B) PSI
  Panel B  — junction lines colored by GROUP 1 (young / cell type A) PSI
  Shared   — exon/transcript structure drawn once below both panels

Each junction is a horizontal coloured line; colour = PSI via copper colormap
(matching ATSEmapper atseviz style).  ALL junctions shown.  Junctions with
|dPSI| above --sig-dpsi are marked with ★ and the ΔΨ value.

Usage
-----
# Age DS (young vs old in one tissue_cell_type)
python 04_visualize_ds_atse.py \\
    --cluster   "chr11:ENSMUSG00000018381.15_atse_3_-" \\
    --condition  Brain_Myeloid_MICROGLIA \\
    --analysis   age_ds \\
    --lc-dir    <LEAFCUTTER_DS dir> \\
    --output    <out.pdf>

# Cell-type DS
python 04_visualize_ds_atse.py \\
    --cluster   "chr11:ENSMUSG00000018381.15_atse_3_-" \\
    --condition  Brain \\
    --analysis   celltype_ds \\
    --group1    MICROGLIA \\
    --group2    "ENDOTHELIAL CELL" \\
    --lc-dir    <LEAFCUTTER_DS dir> \\
    --output    <out.pdf>
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import gffutils

DEFAULT_DB = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "TabulaSenis/genome_files/gencode.vM19/genes/gencode.vM19.db"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_cluster(cluster_id):
    chrom, rest = cluster_id.split(":", 1)
    strand  = rest[-1]
    atse_id = rest[:-2]
    return chrom, atse_id, strand


def parse_jid(jid):
    """chr11_95833334_95833639_- → (chr11, 95833334, 95833639, -)"""
    parts = jid.rsplit("_", 1)
    strand = parts[1]
    coords = parts[0].split("_")
    return coords[0], int(coords[1]), int(coords[2]), strand


def get_atse_junctions(junc_map_path, atse_id):
    jmap = pd.read_csv(junc_map_path, sep="\t")
    rows = jmap[jmap["event_id"] == atse_id].copy()
    if rows.empty:
        raise ValueError(f"event_id '{atse_id}' not found in {junc_map_path}")
    rows[["j_chrom","j_start","j_end","j_strand"]] = rows["junction_id"].apply(
        lambda x: pd.Series(parse_jid(x))
    )
    return rows.sort_values("j_start").reset_index(drop=True)


def compute_group_psi(counts_path, metadata_path, junctions, group_col, g1, g2):
    """Return DataFrame indexed by leafcutter_junc_id with columns g1, g2, delta."""
    meta = pd.read_csv(metadata_path)
    meta["age_group"] = meta["age"].map(
        {"2m": "young", "3m": "young", "18m": "old", "24m": "old"}
    )
    g1_samples = meta[meta[group_col] == g1]["sample_name"].tolist()
    g2_samples = meta[meta[group_col] == g2]["sample_name"].tolist()

    wanted = junctions["leafcutter_junc_id"].tolist()
    counts = pd.read_csv(counts_path, sep="\t", index_col=0, compression="gzip")
    rows   = counts.loc[counts.index.isin(wanted)]

    def mean_psi(samples):
        present = [s for s in samples if s in rows.columns]
        if not present:
            return pd.Series(np.nan, index=rows.index)
        sub = rows[present]
        totals = sub.sum(axis=0).replace(0, np.nan)
        return sub.div(totals, axis=1).mean(axis=1)

    psi = pd.DataFrame({g1: mean_psi(g1_samples), g2: mean_psi(g2_samples)})
    psi["delta"] = psi[g1] - psi[g2]
    return psi


def get_transcripts(db, chrom, region_start, region_end, strand, max_tx=6):
    """Return list of (tx_name, tx_type, [(start,end),...]) for exons in region."""
    try:
        genes = list(db.region(seqid=chrom, start=region_start, end=region_end,
                               strand=strand, featuretype="gene"))
    except Exception:
        return []
    results = []
    seen = set()
    for gene in genes[:2]:
        for tx in db.children(gene, featuretype="transcript", order_by="start"):
            tx_type = tx.attributes.get("transcript_type", ["unknown"])[0]
            if tx_type not in ("protein_coding", "nonsense_mediated_decay"):
                continue
            tx_name = tx.attributes.get("transcript_name", [tx.id])[0]
            if tx_name in seen:
                continue
            seen.add(tx_name)
            exons = [(e.start, e.end)
                     for e in db.children(tx, featuretype="exon", order_by="start")]
            if exons:
                results.append((tx_name, tx_type, exons))
            if len(results) >= max_tx:
                return results
    return results


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_exons(ax, transcript_data, region_start, region_end):
    """Draw exon boxes and intron dashes. Returns final y_offset used."""
    y = 0.0
    for tx_name, tx_type, exons in transcript_data:
        for ex_s, ex_e in exons:
            xs = max(ex_s, region_start)
            xe = min(ex_e, region_end)
            if xs >= xe:
                continue
            ax.add_patch(plt.Rectangle((xs, y - 0.25), xe - xs, 0.5,
                                       color="#555555", alpha=0.75, zorder=3))
        for i in range(len(exons) - 1):
            ax.plot([exons[i][1], exons[i+1][0]], [y, y],
                    color="#555555", lw=0.8, ls="--", alpha=0.5, zorder=2)
        short_type = "pc" if tx_type == "protein_coding" else "nmd"
        ax.text(region_end + (region_end - region_start) * 0.01,
                y, f"{tx_name} ({short_type})",
                va="center", fontsize=7, color="#444444")
        y -= 0.7
    return y


def draw_junctions(ax, junctions, psi_series, cmap, norm, sig_mask,
                   region_start, region_end, direction=1):
    """
    Draw one junction per row as a horizontal coloured line.
    direction=1 → junctions stacked upward from y=0 (old panel)
    direction=-1 → stacked downward (young panel, will be flipped)
    Significant junctions get a ★ marker and ΔΨ label.
    """
    y = 0.0
    step = 1.2
    x_mid = (region_start + region_end) / 2

    for _, row in junctions.iterrows():
        jid   = row["leafcutter_junc_id"]
        j_s   = int(row["j_start"])
        j_e   = int(row["j_end"])
        psi   = psi_series.get(jid, np.nan)
        is_sig = sig_mask.get(jid, False)

        color = cmap(norm(psi)) if not np.isnan(psi) else "#cccccc"
        lw    = 3.0 if is_sig else 1.5
        alpha = 1.0 if is_sig else 0.7

        y += direction * step
        ax.plot([j_s, j_e], [y, y], color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=4)

        # PSI label at junction mid
        psi_str = f"{psi:.2f}" if not np.isnan(psi) else "NA"
        prefix  = "★ " if is_sig else ""
        ax.text((j_s + j_e) / 2, y + direction * 0.35,
                f"{prefix}{psi_str}",
                ha="center", va="bottom" if direction > 0 else "top",
                fontsize=7.5, fontweight="bold" if is_sig else "normal",
                color=color if not is_sig else "#000000")

        # Junction coordinate label on left
        ax.text(j_s - (region_end - region_start) * 0.005, y,
                f"{j_s:,}–{j_e:,}",
                ha="right", va="center", fontsize=6, color="#666666")

    return y


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_two_panel(psi_df, junctions, transcript_data,
                   chrom, atse_id, strand, gene_name,
                   g1_label, g2_label, sig_dpsi, output_path, padding=1500):

    region_start = int(junctions["j_start"].min()) - padding
    region_end   = int(junctions["j_end"].max())   + padding

    # Colormap (copper, matching ATSEviz)
    cmap = cm.copper
    all_psi = pd.concat([psi_df[g1_label], psi_df[g2_label]]).dropna()
    norm = mcolors.Normalize(vmin=0, vmax=max(all_psi.max(), 0.01))

    # Significance mask: |dPSI| >= sig_dpsi
    sig_mask = (psi_df["delta"].abs() >= sig_dpsi).to_dict()

    n_junc = len(junctions)
    n_tx   = len(transcript_data)
    panel_h = n_junc * 1.4 + 1.0
    exon_h  = n_tx  * 0.8 + 0.5
    fig_h   = panel_h * 2 + exon_h + 2.0

    fig = plt.figure(figsize=(14, fig_h))

    # Heights: [g2 panel, g1 panel, exon panel]
    gs = fig.add_gridspec(3, 1,
                          height_ratios=[panel_h, panel_h, exon_h],
                          hspace=0.05)
    ax_g2   = fig.add_subplot(gs[0])   # old / group2 on top
    ax_g1   = fig.add_subplot(gs[1], sharex=ax_g2)   # young / group1 below
    ax_exon = fig.add_subplot(gs[2], sharex=ax_g2)   # exons at bottom

    # ── Panel: group 2 (old) ─────────────────────────────────────────
    g2_psi = psi_df[g2_label]
    y2 = draw_junctions(ax_g2, junctions, g2_psi, cmap, norm,
                        sig_mask, region_start, region_end, direction=1)
    ax_g2.set_xlim(region_start, region_end + (region_end - region_start) * 0.18)
    ax_g2.set_ylim(-0.5, y2 + 1.5)
    ax_g2.set_ylabel(g2_label, fontsize=10, fontweight="bold")
    ax_g2.set_yticks([])
    ax_g2.spines[["top","right","left","bottom"]].set_visible(False)
    ax_g2.axhline(0, color="black", lw=0.5)

    # ── Panel: group 1 (young) ───────────────────────────────────────
    g1_psi = psi_df[g1_label]
    y1 = draw_junctions(ax_g1, junctions, g1_psi, cmap, norm,
                        sig_mask, region_start, region_end, direction=1)
    ax_g1.set_ylim(-0.5, y1 + 1.5)
    ax_g1.set_ylabel(g1_label, fontsize=10, fontweight="bold")
    ax_g1.set_yticks([])
    ax_g1.spines[["top","right","left","bottom"]].set_visible(False)
    ax_g1.axhline(0, color="black", lw=0.5)

    # ── Exon / transcript panel ──────────────────────────────────────
    draw_exons(ax_exon, transcript_data, region_start, region_end)
    ax_exon.set_xlim(region_start, region_end + (region_end - region_start) * 0.18)
    ax_exon.set_ylim(-(len(transcript_data) * 0.7 + 1), 0.8)
    ax_exon.set_yticks([])
    ax_exon.set_xlabel(f"Genomic position ({chrom})", fontsize=10)
    ax_exon.spines[["top","right","left"]].set_visible(False)

    # ── Shared colorbar ──────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_g2, ax_g1], orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Mean PSI", fontsize=9)

    # ── Title ────────────────────────────────────────────────────────
    sig_count = sum(sig_mask.values())
    fig.suptitle(
        f"{gene_name}  —  {atse_id}  ({chrom}, {strand} strand)\n"
        f"{g2_label}  vs  {g1_label}   "
        f"[★ = |ΔΨ| ≥ {sig_dpsi:.2f},  {sig_count}/{n_junc} junctions]",
        fontsize=11, y=0.99
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster",   required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--analysis",  required=True, choices=["age_ds","celltype_ds"])
    ap.add_argument("--lc-dir",    required=True)
    ap.add_argument("--junc-map",  default=None)
    ap.add_argument("--db",        default=DEFAULT_DB)
    ap.add_argument("--output",    default=None)
    ap.add_argument("--group1",    default=None)
    ap.add_argument("--group2",    default=None)
    ap.add_argument("--sig-dpsi",  type=float, default=0.10,
                    help="Min |dPSI| to mark junction as significant with ★ (default: 0.10)")
    ap.add_argument("--padding",   type=int, default=1500)
    args = ap.parse_args()

    junc_map  = args.junc_map or os.path.join(args.lc_dir, "junction_atse_map.tsv")
    meta_path = os.path.join(args.lc_dir, "sample_metadata.csv")

    if args.analysis == "age_ds":
        sub_dir   = os.path.join(args.lc_dir, "by_celltype", args.condition)
        group_col = "age_group"
        g1 = args.group1 or "young"
        g2 = args.group2 or "old"
    else:
        sub_dir   = os.path.join(args.lc_dir, "by_tissue", args.condition)
        group_col = "broad_cell_type"
        if not (args.group1 and args.group2):
            ap.error("--group1 and --group2 required for celltype_ds")
        g1, g2 = args.group1, args.group2

    counts_path = os.path.join(sub_dir, "counts_subset.tsv.gz")
    chrom, atse_id, strand = parse_cluster(args.cluster)
    output = args.output or f"{args.condition}_{atse_id}.pdf"

    print(f"Cluster:  {args.cluster}")
    print(f"Groups:   {g2} (top)  vs  {g1} (bottom)")

    # 1. Junctions
    print("\n[1] Junctions ...")
    junctions = get_atse_junctions(junc_map, atse_id)
    for _, r in junctions.iterrows():
        print(f"   {r['junction_id']}")

    # 2. PSI
    print(f"\n[2] Computing PSI ...")
    psi_df = compute_group_psi(counts_path, meta_path, junctions, group_col, g1, g2)
    # Reindex to ensure all ATSE junctions appear (NaN for 0-read junctions)
    psi_df = psi_df.reindex(junctions["leafcutter_junc_id"].values)
    print(psi_df[[g1, g2, "delta"]].round(3).to_string())

    # 3. Gene name
    jmap = pd.read_csv(junc_map, sep="\t")
    gene_name = jmap[jmap["event_id"] == atse_id]["gene_name"].iloc[0]

    # 4. Transcripts
    print(f"\n[3] Fetching transcripts ...")
    db = gffutils.FeatureDB(args.db)
    r_start = int(junctions["j_start"].min()) - args.padding
    r_end   = int(junctions["j_end"].max())   + args.padding
    tx_data = get_transcripts(db, chrom, r_start, r_end, strand)
    print(f"    {len(tx_data)} transcripts found")

    # 5. Plot
    print(f"\n[4] Plotting → {output}")
    plot_two_panel(psi_df, junctions, tx_data,
                   chrom, atse_id, strand, gene_name,
                   g1, g2, args.sig_dpsi, output, padding=args.padding)


if __name__ == "__main__":
    main()
