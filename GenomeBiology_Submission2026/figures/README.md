# figures/

Layout as of 2026-08-01.

```
_common/                        leaflet_theme.py · figure_common.py · validate_palette.py
FIGURE1_atlas/                  code/ data/ plots/
FIGURE2_simulation_imputation/  code/ data/ plots/ logs/
FIGURE3_leafcutter_ds/          code/ data/ plots/
FIGURE4_aging/                  code/ data/ plots/
FIGURE5_conservation/           code/ data/ plots/
supplemental/
```

Every figure directory has the same three subdirectories: `code/` builds, `data/` is read, `plots/`
is written. Nothing renders next to its source any more.

## The figures

| | content |
|---|---|
| **1** | The mouse splicing atlas — what the data is, before modelling |
| **2** | **Simulation + imputation** — recovers known structure, and imputes held-out data |
| **3** | **Full-data model: calibration vs LeafCutter, differential splicing** |
| **4** | **Aging case study in mouse** |
| **5** | **Human vs mouse brain splicing** |

`FIGURE_OUTLINE_ALL.md` is the authority on panel-level content and status.
`REPRODUCE_ALL.md` is the authority on how to regenerate everything.

## Two naming rules

**"Atlas" means the full dataset (Figure 1) and the released full-data models (`*_full_K30_*`).**
It does *not* mean the 70/30 held-out imputation work — that is `impute`/`heldout` throughout the
code. The held-out models live in `LEAFLETFA_MODELS/2026-07-30_atlas9_default`; the directory keeps
its historical name so job logs stay traceable, but `panels.py` calls it `IMPUTE_DIR`.
`2026-07-26_atlas9` is **superseded**.

**In `RESAMPLED_<n>_PERCENT`, `<n>` is percent DROPPED.** `100_PERCENT` is the most depleted file,
not the fullest. This reads backwards and has caused real errors.

## Building Figure 2

```bash
cd FIGURE2_simulation_imputation
python code/build_sim.py           # 2B-D + simulation supplements; needs only data/
python code/build_imputation.py    # 2E-F + held-out supplements + the assembled FIG2_validation
```

Both are thin entry points over `code/panels.py`, which holds every loader and panel. They are not
two copies: `figure_validation` is the assembled main figure and spans both halves, because Figure 2
*is* simulation plus imputation.

A missing input is a **failure with a non-zero exit**, not a skipped line. If a figure is listed and
does not appear, something is genuinely absent — read the message.

## What is not in git

`.gitignore` excludes `*.pdf`, `*.png` and `*.csv`. Rendered panels and result CSVs — including
`FIGURE2_simulation_imputation/data/rerun_logs/*.csv`, which the panels glob at runtime — are **not
recoverable from history**. Do not delete them expecting a `git checkout` to bring them back. Code is
tracked and is recoverable.
