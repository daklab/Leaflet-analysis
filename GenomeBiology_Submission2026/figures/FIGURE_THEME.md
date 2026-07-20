# LeafletFA figure theme — Genome Biology submission

**Scope:** every panel in `GenomeBiology_Submission2026/figures/**` (main + supplemental),
including panels rendered from notebooks. One visual system across all figures.

**Rule zero:** never hand-set fonts, sizes, DPI, or colors in a figure script.
Import the theme and use its constants:

```python
import sys; sys.path.insert(0, "<repo>/GenomeBiology_Submission2026/figures")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, save
apply_theme()
```

`leaflet_theme.py` is the executable version of this document. If the two ever disagree,
the module is authoritative — fix this file.

---

## 1. Canvas geometry

Nature-family column widths. Never invent a width; pick one of these.

| Name | Width | Use |
|---|---|---|
| `single` | 89 mm (3.50 in) | one-column panel or small figure |
| `onehalf` | 120 mm (4.72 in) | intermediate |
| `double` | 183 mm (7.20 in) | full-width multipanel figure |

Max page height **247 mm (9.72 in)** including the caption — keep figure art ≤ 220 mm.

`figsize("double", h=3.2)` returns the inches tuple. For a grid, size the *whole figure*
to a column width and let subplots divide it — do not build panels at arbitrary sizes and
scale them later. Scaling is what makes text sizes inconsistent between panels.

## 2. Type

Sans-serif throughout: **Arial → Helvetica → DejaVu Sans** (first available).
Never mix serif and sans in one figure. Never use a condensed or display face.

| Element | Size | Weight |
|---|---|---|
| Panel label (`a`, `b`, `c`) | 8 pt | bold, lowercase |
| Axis label | 7 pt | regular |
| Tick label | 6 pt | regular |
| Legend text / annotation | 6 pt | regular |
| Legend title / axes title | 7 pt | regular |
| In-plot stat text (n, p, r) | 6 pt | regular |

**Hard floor: 5 pt after any scaling.** Nature rejects smaller. If a label does not fit at
6 pt, the panel is too dense — drop elements, don't shrink type.

Panel labels sit at the **top-left outside the axes**, lowercase bold, no parentheses and
no period: `a`, not `(A).` Use `panel_label(ax, "a")`.

Text conventions:
- Sentence case for axis labels and titles ("Junctions per ATSE", not "Junctions Per ATSE").
- Units in parentheses: `Read depth (millions)`.
- Italicise gene symbols (`$\it{Hmg20b}$`) and statistical variables (*n*, *P*, *r*).
- Report *P* as `P = 0.003`, or `P < 1 × 10⁻⁴` below that. Never `p-val` or `pval`.
- Thousands separators in counts: `12,481`.
- No figure-level suptitle in the final artwork — that belongs in the caption. Titles are
  fine while exploring; strip them before export.

## 3. Color

A **LeafletFA-specific palette** — cool-dominant, deliberately unlike stock
matplotlib/seaborn defaults and unlike the Okabe–Ito set most papers use. It is
*constructed*, not eyeballed: the five series colors sit on a CIELAB **lightness ladder**,
each ~13 L* from its neighbours.

That ladder is the whole trick. Dichromats effectively lose the red–green axis and read
lightness instead, and greyscale printing discards hue entirely — so spacing colors by
lightness is what makes a palette survive both. Hue variety alone never does.

| Token | Hex | L* | Reserved meaning |
|---|---|---|---|
| `teal` | `#006865` | 39 | **LeafletFA / the method** |
| `indigo` | `#657BB5` | 52 | primary baseline (NMF) |
| `ochre` | `#B79B51` | 65 | second baseline (scVI) |
| `rose` | `#ECB0CA` | 78 | null / shuffled / negative control |
| `olive` | `#3A4621` | 28 | ablation / third comparator (PCA) |
| `grey` | `#6E7478` | — | non-focal context only — never a compared series |

**Method colors are fixed across all figures.** LeafletFA is `teal` in every panel it
appears in; NMF is `indigo` everywhere. Never let a per-panel default cycle reassign them.

Verified by `validate_palette.py`: all 10 pairs clear ΔE ≥ 20 (normal vision), ≥ 15 under
simulated deuteranopia and protanopia, and ΔL* ≥ 8 for greyscale. **Re-run it after any
color change** — it exits non-zero on failure, so it can gate a figure build:

```bash
python GenomeBiology_Submission2026/figures/validate_palette.py           # numeric check
python GenomeBiology_Submission2026/figures/validate_palette.py --sheet   # + swatch proof
```

`--sheet` regenerates `palette_reference.png/.pdf` — the palette rendered under normal
vision, deuteranopia, protanopia and greyscale. Check both: numbers catch what the eye
misses on small marks, and the sheet catches what thresholds let through.

**Hard limit: five series colors per panel.** A sixth cannot be added without breaking
greyscale or dichromat separation — that is a property of color space, not a lack of
effort. More than five categories → facet the panel, or collapse to top-4 + `grey` "other".

`grey` is exempt from the separation checks precisely because it never encodes a compared
series. If you need grey to be *distinguishable* from a series color, it has become a
series: give it a real color or facet.

Continuous maps:
- sequential → `mako` (cool, teal-anchored, matches the palette identity)
- diverging (ΔPSI, log fold change, correlation) → `BrBG`, brown↔teal, **always symmetric
  about 0** (`vmin=-v, vmax=+v`) — use `symmetric_norm()`; an off-center diverging map
  misrepresents the data
- PSI (0–1 bounded) → `mako` with `vmin=0, vmax=1` fixed, so PSI heatmaps are comparable
  across figures
- `viridis` only to match a previously published panel

Cell-type colors must be **consistent across every figure** — define once in
`cell_type_maps/` and load it, never re-derive from a palette cycle (ordering changes
between datasets and silently swaps colors between panels).

## 4. Axes and marks

- **Despine**: remove top and right spines on every Cartesian axes. Keep left+bottom at
  0.6 pt in near-black `#222222`, never pure black.
- Ticks outward, 2.5 pt long, 0.6 pt wide. No minor ticks unless log-scaled.
- **No gridlines by default.** If a readout genuinely needs one, a single light horizontal
  grid (`#E6E6E6`, 0.5 pt, below the data) is allowed — never both axes, never vertical only.
- **No frames**: legends `frameon=False`, no boxes around panels.
- Start count/bar axes at 0. Note any truncated axis explicitly in the caption.
- Log axes get real decade ticks (`10⁰, 10¹, 10²`), not `1e0`.
- Line width 1.0 pt for data lines, 0.6 pt for reference/annotation lines
  (reference lines: grey, dashed `(0,(3,2))`).
- Scatter: `s=6`, `linewidth=0`. If >5,000 points, `alpha=0.25` **and** `rasterized=True`
  (keeps the PDF openable while axes/text stay vector).
- Error bars are 0.8 pt with no caps; state in the caption what they represent (SD, SEM, or
  95% CI) — always state it.
- Boxplots: show the underlying points when *n* ≤ 100 (strip/swarm, `alpha=0.5`); otherwise
  box + violin. Never a bare bar chart for a distribution.
- Always annotate *n* per group, in-plot at 6 pt or in the caption. Never omit it.

## 5. Legends

Prefer **direct labeling** of lines/points over a legend box — a labeled line is faster to
read and survives shrinking. Use a legend when there are >3 series or the marks overlap.

- `frameon=False`, `handlelength=1.2`, `labelspacing=0.35`, `borderpad=0.2`
- Place outside the axes only if it would occlude data; otherwise `loc="best"` inside.
- One legend per *figure* when panels share an encoding, not one per panel.

## 6. Export

Every panel is saved by `save(fig, "<path stem>")`, which writes **both**:

- `.pdf` — vector, the submission file. Fonts embedded as editable text
  (`pdf.fonttype = 42`), not outlines.
- `.png` at **600 dpi** — for review, Slack, and the notebook.

Rules:
- `bbox_inches="tight"`, `pad_inches=0.02`.
- No `transparent=True` — Genome Biology composites on white.
- Filenames: `FIGURE<N>_<topic>/figure_panels/<panel><letter>_<slug>.pdf`, e.g.
  `FIGURE2_benchmark/figure_panels/2c_pr_curve.pdf`. Lowercase, underscores, no spaces.
- Rasterize dense *data* layers only (`rasterized=True` on the artist), never whole axes —
  text and axes must stay vector.

## 7. Pre-submission checklist

Run through this before declaring a figure done:

- [ ] Figure width is exactly a column width (89 / 120 / 183 mm)
- [ ] Smallest text ≥ 5 pt at final size; nothing shrunk after export
- [ ] Panel labels bold lowercase, consistent position across all figures
- [ ] Method colors match the reserved table (LeafletFA teal everywhere)
- [ ] `validate_palette.py` passes; ≤5 series colors in any one panel
- [ ] Legible in greyscale; no red/green-only distinction
- [ ] Diverging colormaps centered on 0; PSI maps on a fixed 0–1 scale
- [ ] Typeface resolves to Arimo/Arial, not the DejaVu fallback (`check_fonts()`)
- [ ] *n* stated for every group; error bars defined
- [ ] Top/right spines gone; no stray gridlines or legend frames
- [ ] Vector PDF opens with selectable text
- [ ] Same quantity uses the same axis label wording across figures
