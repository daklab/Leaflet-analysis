# Supplementary figures

Organised **by theme**. Each theme directory holds its scripts, its hand-written methods, a
`output/` folder for generated artefacts, and (where needed) a gitignored `cache/`.

```
supplemental/
  _shared/          code reused across themes (no dataset knowledge, no plotting)
  data_quality/     what is in the resource, and is the coverage adequate?
  _archive/         superseded outputs kept for reference — do not cite
```

## `data_quality/`

There are **two distinct figures** here. Do not confuse them.

### Figure 1 — `supp_dataset_coverage.py` (8 panels, a–h)
*What is in the resource, and is the coverage adequate?* Inventory, per-cell depth × breadth,
junction-read fraction, reads per junction vs ATSE size, junctions per ATSE, coverage across
tissues, **single cells vs single nuclei (matched)**, junction position along the gene.

| output | dataset |
|---|---|
| `supp_dataset_coverage_mouse_smartseq2` | mouse — **deliberately POOLED** (TMS cells + Allen nuclei) |
| `supp_dataset_coverage_human_allen_brain` | human Allen Brain (nuclei) |

⚠️ The mouse figure is pooled **on purpose**: panel **g** is the cells-vs-nuclei comparison and
needs both modalities present. Splitting it would delete the figure's main result (pooled 4.8×
collapsing to ~1.0× under matching). Do not "fix" this.

### Figure 2 — `supp_exons_vs_atses.py` (6 panels, A–F)
*Genes without ATSEs are genes that cannot be spliced.* Exon count predicts ATSE detection
(partial ρ ≈ 0.46–0.57) far more than read depth; single-exon genes essentially never carry one.

| output | dataset |
|---|---|
| `supp_exons_vs_atses_mouse_tms` | mouse Tabula Muris Senis — **single cells** |
| `supp_exons_vs_atses_mouse_allen_brain` | mouse Allen Brain — **single nuclei** |
| `supp_exons_vs_atses_human_allen_brain` | human Allen Brain — **single nuclei** |
| `supp_exons_vs_atses_rna_atlas` | human bulk |

Here the mouse object IS split, because pooling two modalities under one "SC" label was
misleading and made mouse non-comparable to human nuclei. The split is not just a cell filter:
per-gene junction/ATSE counts are **recomputed within each subset** (via `row_mask` in
`_shared/_h5ad_scan.py`), since `var`-level counts are object-level and would otherwise report
the pooled numbers.

### Running
```bash
cd data_quality
python supp_dataset_coverage.py --dataset mouse_smartseq2      # ~80 s first run, ~3 s after
python supp_dataset_coverage.py --dataset human_allen_brain
python supp_exons_vs_atses.py  --dataset mouse_tms
python supp_exons_vs_atses.py  --dataset mouse_allen_brain
python supp_exons_vs_atses.py  --dataset human_allen_brain
python supp_exons_vs_atses.py  --dataset rna_atlas
```

### Reading Figure 2 across datasets — do NOT compare the depth panels

| dataset | exons vs #ATSEs (partial) | depth vs #ATSEs (marginal → partial) | expressed | depth IQR (log10) |
|---|---|---|---|---|
| human Allen (nuclei) | +0.47 | +0.38 → **+0.26** | 16,232 | 0.78 |
| RNA atlas (bulk) | +0.57 | +0.26 → **+0.06** | 31,324 | 1.09 |
| mouse Allen (nuclei) | +0.50 | +0.02 → **+0.01** | 11,333 | 0.82 |
| mouse TMS (cells) | +0.46 | −0.18 → **−0.14** | 9,997 | 0.60 |

**Exon count is the robust result** — +0.46 to +0.57 in all four, and barely moved by controlling
for depth. Quote this.

**The depth column is NOT interpretable across datasets and we do not have a verified
explanation.** A "range restriction" story (mouse GE pre-filtered to well-expressed genes, so
depth cannot vary) was proposed and is **refuted by these numbers**: mouse Allen has a *wider*
depth spread than human (IQR 0.82 vs 0.78) yet essentially zero depth effect, and the atlas has
both the most genes and the widest spread yet only +0.06. Gene count and depth spread are both
non-monotone with the depth effect.

What is certainly true is that the four GE objects come from **different quantification
pipelines** (mouse = `tms_ab_exons_combo` exon-level counts; human Allen and the atlas each
different), so "gene read depth" is not the same measured quantity across rows. That alone makes
the cross-dataset depth comparison invalid, independent of any biology. Report depth **per
dataset** and do not rank datasets by it.

Outputs land in `data_quality/output/` as `<stem>_<dataset>.{pdf,png}` plus a long-format
`_summary.csv` and a generated `METHODS_*.md`. The summary CSVs are designed to concatenate
across datasets:

```python
pd.concat([mouse_csv, human_csv]).query("group_by=='all' and stat=='median'") \
  .pivot_table(index='metric', columns='dataset', values='value')
```

### Caching
`supp_dataset_coverage.py` scans a 30 GB object once and caches three parquet tables in
`data_quality/cache/` (gitignored). A JSON manifest records each input's path, mtime and size;
the plot stage **refuses a stale cache** rather than silently plotting from a superseded input
file. Override only with `--force-cache`.

## `_shared/`
`_h5ad_scan.py` — memory-safe CSR primitives for very large `.h5ad` matrices (row/column
aggregates, distinct-group counts, broadcast-layer verification). Never materialises the matrix;
`anndata`'s backed mode does, which is why this exists.

## Theme compliance
`supp_dataset_coverage.py` uses `figures/leaflet_theme.py` (widths, type scale, palette, export).
`supp_exons_vs_atses.py` **predates the theme and does not comply** — hardcoded colours and sizes,
non-column width, `suptitle`, 150 dpi. Port it before submission.
