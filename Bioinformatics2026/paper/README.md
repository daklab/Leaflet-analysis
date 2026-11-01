# LeafletFA paper: figures and numbers

Self-contained: everything the figures and quoted numbers need, in the SpliceVI paper's style (style code and config
conventions copied from SpliceVI-utils `manuscript/paper/`, 2026-11-01).

- `paper_config.yaml`: style, palette, method colors (shared with SpliceVI), input paths, figure sizes and layouts.
- `figures/make_figures.py --list | --figure N | --panels 2b 2c`: panels and assembled previews under `output_dir`.
  Figure modules go in `figures/panels/` (start from `_template.py`) and are registered in `make_figures.FIGURE_MODULES`.
- `ledger/results_ledger.tsv`: every number quoted in the text (value, display, source file + md5, job, record, status).
  Columns as in SpliceVI's ledger, with `record` (where the number is reported) in place of `stage5_section`.
- `scripts/make_numbers_tex.py`: ledger -> `numbers.tex` (final rows only; refuses rows whose source md5 changed), copied into
  `../manuscript/`.
- `captions/<fig>.md`, `records/FIGURE_BUILD_<fig>.md`: caption drafts and per-figure build records, as for SpliceVI.
