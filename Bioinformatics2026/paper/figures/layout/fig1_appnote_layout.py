"""Layout mock-up (boxes and labels only, no data) of the single Application Note figure, for Karin's approval before building.
183 mm wide (double column), 165 mm tall. python fig1_appnote_layout.py -> fig1_appnote_layout.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
MM = 1 / 25.4; W, H = 183, 165
fig = plt.figure(figsize=(W * MM, H * MM), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"
def box(x, y, w, h, lab, title, body, fc="#F3F5F6"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#6E7478", lw=0.6))
    ax.text(x - 1, y - 1.2, lab, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.text(x + 2, y + 3.5, title, fontsize=6.5, fontweight="bold", va="top")
    ax.text(x + 2, y + 8.5, body, fontsize=5.5, va="top", linespacing=1.35, color="#222222")
def strip(x, y, w, note):
    ax.add_patch(Rectangle((x + 2, y), w - 4, 9, fc="white", ec="#B79B51", lw=0.5, ls=(0, (3, 2))))
    ax.text(x + 4, y + 4.5, note, fontsize=5, va="center", color="#6E7478")
box(4, 6, 112, 52, "a", "Catalog and model",
    "left: ATSEmapper schematic, junctions -> splice graph -> ATSE\n(drawn, from DEFINITIONS.md)\n\nmiddle: catalog per atlas and event types\n(fig1.py panels 1b, 1c; ledger macros)\n\nright: LeafletFA in one line: cells x junctions = phi x psi,\nper-cell usage, cell-type mean with interval")
box(122, 6, 57, 52, "b", "Accuracy vs raw pseudobulk (10.1)",
    "error vs TEST pseudobulk,\nLeafletFA and raw, by TRAIN reads\n(1-19, 20-99, ...) and type size;\nTMS and Allen; 95% CI\n\nsource: tsv/c10_*_s0_accuracy.tsv\n(+ accuracy_units for bins)\n\nn per bin in-plot")
ex = [("c", "Gria2 flip/flop (Allen)", "linked MXE, ATSE atse_3, 7 junctions\npsi of the flip and flop junctions\nper cell group, grouped by class\n(glutamatergic, GABAergic)"),
      ("d", "Fgfr2 IIIb/IIIc (TMS)", "upstream pair, atse_3; IIIb usage\nper type, epithelial vs mesenchymal;\ncovered (raw vs LeafletFA) and\nuncovered cells shown apart"),
      ("e", "Nrxn SS4 (Allen)", "candidate: Nrxn1 atse_2 (or Nrxn2 atse_1);\nSS4 junctions per cell group,\nexcitatory vs inhibitory\n(Nrxn3 limited: downstream only)"),
      ("f", "Ptprc (TMS)", "atse_2 (MXE linked, 10 junctions);\nlead junction psi in B and T cells\nvs other immune types;\nTEST direction from 10.4")]
for i, (lab, t, b) in enumerate(ex):
    x = 4 + (i % 2) * 90; y = 66 + (i // 2) * 49
    box(x, y, 85, 43, lab, t, "")
    strip(x, y + 6, 85, "exon structure (portal structure diagram; GENCODE vM19)")
    ax.add_patch(Rectangle((x + 2, y + 17), 50, 24, fc="white", ec="#657BB5", lw=0.5, ls=(0, (3, 2))))
    ax.text(x + 4, y + 19, "psi per cell type: dot = LeafletFA\n(interval, not calibrated), ring = raw\nTRAIN pseudobulk sized by reads", fontsize=4.8, va="top", color="#6E7478")
    import textwrap; ax.text(x + 54, y + 18, "\n".join(textwrap.fill(l, 30) for l in b.split("\n")), fontsize=5, va="top", color="#222222", linespacing=1.3)
ax.text(4, 163, "Layout only, no data. 183 x 165 mm. Gallery of 10-20 further ATSEs in the Supplement; every event in the browser. Colors: LeafletFA teal, raw grey, lineages from the cell-type maps.",
        fontsize=5, color="#6E7478")
fig.savefig("fig1_appnote_layout.png", dpi=200)
