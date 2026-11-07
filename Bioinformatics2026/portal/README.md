# Cell-type splicing browser (static, local preview)

Static files, no server code, all computation client-side. Not published; do not deploy from here.

```bash
V=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation
python -m http.server -d $V/portal_browser/site 8000     # open http://localhost:8000
```
`fetch()` needs http; opening `index.html` from `file://` does not work. Over ssh: `ssh -L 8000:localhost:8000 <host>`.

## Rebuild
```bash
sbatch -J pb_cat -c 2 --mem 1G -t 0:30:00 run_precompute.sbatch precompute.py catalog
sbatch -J pb_TMS -c 4 --mem 8G -t 0:30:00 run_precompute.sbatch precompute.py counts TMS
sbatch -J pb_Allen -c 4 --mem 18G -t 0:30:00 run_precompute.sbatch precompute.py counts Allen
sbatch -J pb_build -c 2 --mem 2G -t 0:15:00 run_precompute.sbatch build_portal.py all     # data shards + web files
python build_portal.py web                                                                  # web/ edits only
python make_screenshots.py                                                                  # browser checks + screenshots
```
Measured peak memory: catalog 0.58 GiB, counts TMS 6.05 GiB, counts Allen 13.8 GiB, build 1.24 GiB; the requests above are peak + 30 %.
All generated data live under `$V/portal_browser/` (`precomputed/`, `site/`, `screenshots/`), not in git. `SOURCE.json` here
(copy in `site/data/`) holds input md5s, fit commits, counts and sizes.

## Fixed wording and rules
`web/config.js` is the single place for the interval label, the compare-view note, the grey rule and the default read filters.

- Differences are LeafletFA point estimates with intervals, ranked for exploration. The site makes no pairwise calls. The
  check 10 arrays `specific`, `tv`, `p_tv` are neither read nor shipped.
- Grey estimates: `CONFIG.GREY_RULE`. Default `"n_mice_lt2"` (Karin's spec, 2026-11-06): cell types with fewer than 2 mice. Under
  this rule **no type is grey at present** in TMS or Allen (every type has at least 2 mice; `SOURCE.json`). Alternative `"eligible"`:
  the saved check 10 flag is false (fewer than 2 mice, or no TRAIN cells in one of the two frozen mouse halves), which greys 5 TMS
  types and 1 Allen group.
- Junction colors: `CONFIG.JUNCTION_COLORS` (light) and `JUNCTION_COLORS_DARK`, the same eight hues stepped per surface, chosen by
  the page's color scheme. Validated with the dataviz `validate_palette.js` (2026-11-06):
  - dark on `#15191B` and `#1E2427`: all checks pass (lightness band, chroma, CVD worst adjacent ΔE 8.4, normal-vision 19.3, contrast ≥ 3:1);
  - light on `#ffffff` and `#F4F5F3`: all checks pass; contrast WARN below 3:1 for 3–4 colors, relieved by the J labels and the junction table.

## Views
- **Search**: gene name or ATSE id; gene page lists its ATSEs.
- **Event** (`#/event/<dataset>/<event_id>`): structure diagram and a ψ dot plot per junction, grouped by tissue / cell type (TMS)
  or class / official subclass (Allen). LeafletFA estimate with interval (where validated); raw TRAIN pseudobulk as an open marker sized by junction reads.
- **Compare two** (`#/compare/<dataset>/<A>/<B>`): ATSEs ranked by max |Δψ| across junctions. Default filter 20 reads and 3 cells per type.
- **One vs. all others** (`#/one/<dataset>/<A>`): ATSEs ranked by max |ψ − rest ψ| (rest: saved check 10 mean over the other types' cells),
  with the range across other eligible types.
- **Downloads**: each table as TSV, with version, fit md5 and fit commit in `#` comment lines.

## Data sources and definitions
- Estimates: `$V/c10/{TMS,Allen}_s0_estimates.npz`. Raw pseudobulk, reads, cells and mice with reads: TRAIN counts (`precompute.py counts`,
  which asserts reads, cells and mice per type against the saved arrays).
- Event class (`precompute.classify`, `mxe_pairs`): the coordinate rules of `../DEFINITIONS.md` 1.6. Linked MXE (STAGE5 s177 definition, at
  least one linked pair) is taken out of "complex". Labels are geometric.
- Junction annotation status: ATSE file `MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz`.
- Exons: GENCODE vM19 `genes.gtf`. Junction id `chrom_start_end_strand`: start = last base of the left exon, end = first base of the right exon - 1.
  The diagram draws, per junction end, the touching exon with most transcripts, plus up to 15 non-overlapping annotated exons inside the
  span that no catalog junction touches (dashed).
- Allen: "cell groups (cluster names)". Class by majority `class_label` of a group's TRAIN cells: Glutamatergic = excitatory,
  GABAergic = inhibitory, any other = non-neuronal.

## File layout (`site/data/`)
- `index.json`: version, shard layout, types per dataset, featured examples (each verified in `build_portal.featured`, or the build fails).
- `catalog/events.json`, `catalog/junctions.bin` (u32 start × J, u32 end × J, u8 status × J), `catalog/struct/<bucket>.json`. Site junction order: by event, then genomic.
- `<dataset>/types/<i>.bin`: one per type; columns in `index.json` `type_layout`; a compare loads two of these.
- `<dataset>/events/<bucket>.bin`: 64 ATSEs per bucket; an event page loads one. Arrays in order: junction reads (u32, types × junctions),
  ATSE reads (u32, types × ATSEs), TEST reads (u32), ψ, lo, hi, raw ψ (u16, types × junctions), cells with reads, mice, TEST cells with reads (u16, types × ATSEs).
- Intervals (`lo`, `hi`): check 12 (a)+(b), from `LeafletFA/scripts/validation/c12_export.py` (`$V/c12/export/<ds>_intervals.npz`). A 95 % predictive
  interval for the pseudobulk ψ of the type's held-out TEST cells at their read depth; missing (65535) where TEST has fewer than 100 reads or fewer
  than 10 cells with reads at the ATSE. The rest-ψ interval has no validated version and is always missing.
- ψ-like values: u16 = value × 10,000; 65535 = missing. All little-endian.

## Filling the "SpliceVI DS call" column
The column ships empty. To fill it, no rebuild:
```bash
python add_splicevi_ds.py CALLS.tsv --source "SpliceVI cell-type DS runs, <run id>"
```
`CALLS.tsv` columns: `dataset, cell_type_a, cell_type_b, event_id, call`. Type names as in `index.json` `name`; order does not matter;
`cell_type_b = rest` fills the one-vs-all view. `call` is shown verbatim, in the table and the TSV, and the column subtitle becomes `source: ...`.
