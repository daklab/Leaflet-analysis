# Application Note figure: caption FACTS (no caption prose; 2026-11-10)

**Build:** `paper/figures/panels/fig_appnote.py` via `make_figures.py --figure AN` (sbatch; final build 2026-11-10, 24 s).
- **Output** (`/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_bioinformatics/paper_figures/figAN/`):
  - `figAN.pdf`: vector, editable text;
  - `figAN.tiff` (350 dpi) and `figAN_review.png` (150 dpi);
  - `panels/AN_{a..f}.pdf`;
  - `source_data/figAN_*.tsv`, one or more per panel.
- **Size:** 183 × 164 mm (limit 165).
- **Fonts:** 6 pt minimum (ticks, legend, annotations); 7 pt labels; Arial or Liberation Sans from `paper_config.yaml`.
- **Colors:**
  - LeafletFA is teal: `methods.leafletfa` changed from ochre to teal on 2026-11-10, per Karin.
  - Raw pseudobulk is gray.
  - TMS / Allen use the dataset colors slate / sage (panels a, b).
  - Lineage strips use indigo / ochre / olive / light gray (labels only, not data marks).
- **Lineage colors:** no lineage color map existed in `cell_type_maps/`. The groupings below are defined in `fig_appnote.py` (`LINEAGES`); Karin to confirm or move them into `cell_type_maps/`.

## Inputs (md5)

| Input | md5 |
|---|---|
| TMS seed-0 fit `fit.npz` (junction / ATSE order) | 140c17e6af58dd9466607e864a0fb983 |
| Allen seed-0 fit `fit.npz` | 974153b331677c443933f67de77a9f9f |
| `c10/TMS_s0_estimates.npz` (point estimate) | 738f74ce8dff156fe3d9232aa8599a19 |
| `c10/Allen_s0_estimates.npz` | a1d787502bf1d53793b24ddc5f0f3e15 |
| `c12/export/TMS_intervals.npz` (check 12 interval, TEST reads/cells) | 098bc7fc97be13af92bcf025c07747d9 |
| `c12/export/Allen_intervals.npz` | 0b5c146f815ccac335b3d7b1749498b0 |
| `portal_browser/precomputed/TMS_train_tables.npz` (TRAIN reads, cells, mice) | 08ccbc1f89830c5909286c7c7a9672c0 |
| `portal_browser/precomputed/Allen_train_tables.npz` | 21a702fbb80f0e9d59a513bb062450dc |
| `catalog_junctions.parquet` / `catalog_events.json` (structure) | 81c745d3b5f29bf20e8ae9f65cf6fa56 / 3b49732b6a1a51fb457f8c6e544b8870 |
| `tsv/c10_TMS_s0_accuracy.tsv` / `c10_Allen_s0_accuracy.tsv` | baa4dd9dabda407b87128615090f82c4 / 5e123d9527f3d4859ef399f91e59687e |
| `tsv/c10_TMS_s0_ptprc_direction.tsv` | 2bab4adf8795cbede4ec313015dc6d00 |
| `fig1b_catalog.tsv` / `fig1c_event_types.tsv` (ledger sources) | 04c635ae429b7a7fa2b25763e966f7f0 / 3d85527e8a95c94dc2c2e9f1d5f3eaaa |

## a. Catalog and model
- **Counts per atlas** (TRAIN + TEST, June 2026 build; asserted equal to the `numbers.tex` macros `Catalog*TMS/Allen`):

  | | TMS | Allen |
  |---|---|---|
  | Cells | 78,135 | 54,571 |
  | Junctions | 77,562 | 77,204 |
  | ATSEs | 24,390 | 24,162 |
  | Genes | 11,210 | 11,050 |

- **Cell types / groups** (types in the check 10 estimates, ≥ 10 TRAIN cells): TMS 132, Allen 68. No ledger macro.
- **Event types of the 24,475 ATSEs** (asserted equal to `EventShare*`; geometric rules, DEFINITIONS.md 1.6):
  - alt 3′ site 5,810 (24 %);
  - alt 5′ site 4,867 (20 %);
  - skipped exon 5,708 (23 %);
  - complex 8,090 (33 %).
  - The browser's 5-class scheme splits 1,034 linked-MXE ATSEs out of "complex". The figure uses the ledger's 4 classes.
- **Model schematic:** placeholder for Karin. Text in the box: "cells × junctions ≈ φ × ψ; Dirichlet-multinomial per ATSE; K = 20".
- **Legend for panels c–f** sits in panel a, lower right (one legend per figure).

## b. Check 10.1 accuracy (pre-registered; passed in both atlases)
- **Quantity:** error = TV between the estimate and the held-out TEST pseudobulk. Plotted: LeafletFA − raw TRAIN pseudobulk, with 95 % CI. Below 0 = LeafletFA closer to held-out cells.
- **Split:** per cell (70/30, stratified by broad type × age, seed 42), not per mouse. TEST cells are new cells of the same mice.

| Atlas | Bin | Units | Types | LeafletFA − raw (95 % CI) |
|---|---|---|---|---|
| TMS | overall | 1,038,848 | 130 | −0.0072 (−0.0094, −0.0050) |
| TMS | 10–49 cells | 51,876 | 31 | −0.0194 (−0.0242, −0.0153) |
| TMS | 50–199 | 307,672 | 47 | −0.0155 (−0.0178, −0.0129) |
| TMS | 200–999 | 452,293 | 38 | −0.0065 (−0.0090, −0.0044) |
| TMS | ≥ 1,000 | 227,007 | 14 | **+0.0054** (+0.0029, +0.0083) |
| TMS | 0 TRAIN reads | 3,731 | 130 | −0.0138 (−0.0166, −0.0107) |
| TMS | 1–19 | 15,617 | 130 | −0.0625 (−0.0681, −0.0571) |
| TMS | 20–99 | 84,817 | 130 | −0.0243 (−0.0278, −0.0211) |
| TMS | ≥ 100 | 934,683 | 130 | −0.0047 (−0.0067, −0.0029) |
| Allen | overall | 763,823 | 68 | −0.0021 (−0.0039, −0.0005) |
| Allen | 10–49 cells | 39,807 | 11 | −0.0116 (−0.0143, −0.0090) |
| Allen | 50–199 | 262,244 | 30 | −0.0077 (−0.0088, −0.0066) |
| Allen | 200–999 | 224,979 | 15 | −0.0022 (−0.0038, −0.0002) |
| Allen | ≥ 1,000 | 236,793 | 12 | **+0.0057** (+0.0044, +0.0074) |
| Allen | 0 TRAIN reads | 385 | 29 | −0.0095 (−0.0169, +0.0010) |
| Allen | 1–19 | 5,111 | 68 | −0.0543 (−0.0599, −0.0482) |
| Allen | 20–99 | 145,013 | 68 | −0.0122 (−0.0134, −0.0110) |
| Allen | ≥ 100 | 613,314 | 68 | +0.0007 (−0.0010, +0.0022) |

- **The gain is largest where data are thin.**
- **Where raw does better:** in types with ≥ 1,000 TRAIN cells, raw pseudobulk is closer than LeafletFA in both atlases (+0.005 to +0.006).
- **At 0 TRAIN reads,** raw falls back to the TRAIN population ψ (the check 10 comparator rule).
- n per bin is in the table; the plot does not print it.

## c–f. Rules (fixed, from `fig_appnote.py`; none hand-picked)
- **Cell types:**
  - every type (TMS tissue-cell type; Allen cell group) with ≥ 20 TRAIN reads and ≥ 3 TRAIN cells with reads at the ATSE (the browser's range-support rule);
  - grouped by lineage;
  - within a lineage, sorted by LeafletFA ψ of the first junction shown.
- **Lineages:**
  - **Allen:** class of the group's TRAIN cells (Glutamatergic → excitatory, GABAergic → inhibitory, else non-neuronal; the browser rule).
  - **TMS *Fgfr2*:** the check 10 epithelial / mesenchymal keyword lists (`c10_catalog.EPI` / `MES`) on the broad cell type; other types = "Other".
  - **TMS *Ptprc*:** B cell / T cell / other immune (NK, macrophage, monocyte, granulocyte, dendritic, myeloid, microglia, general immune, hematopoietic progenitor) / non-immune.
- **Marks:**
  - **LeafletFA point:** check 10 point estimate, seed 0; the mean of the posterior-mean per-cell usage over the type's TRAIN cells.
  - **Interval:** the check 12 (a)+(b) 95 % predictive interval for the type's held-out pseudobulk ψ at its read depth. It is not an interval for a true cell-type ψ, and it is drawn only where TEST cells have ≥ 100 reads and ≥ 10 cells with reads at the ATSE.
    - τ²: TMS 0.01024 / 0.00256; Allen 0.00128 / 0.00032 (100–499 / ≥ 500 TEST reads).
  - **Raw ring:** raw TRAIN pseudobulk ψ = junction reads / ATSE reads, ring area ∝ log junction reads (scaled per panel).
- **Structure:** the browser's diagram.
  - GENCODE vM19 exons at junction ends filled; internal exons with no catalog junction dashed.
  - Compressed coordinates, as `app.js`: exon segments 10 + √length, introns 8 + 7 log10(length + 1). Exons and introns not to scale.
  - Shown junctions in teal (J1, J2); the other junctions of the ATSE in light gray.

### c. *Gria2* flip/flop (Allen, `ENSMUSG00000033981.14_atse_3`; linked MXE, 7 junctions, minus strand)
- **Junctions shown** (MXE pair rule, DEFINITIONS.md 1.6, both exons > 100 nt; the rule returns exactly one pair):
  - **J1** `chr3_80689351_80690403_-`, into exon 80,690,404–80,690,518;
  - **J2** `chr3_80689351_80691319_-`, into exon 80,691,320–80,691,434.
  - **Which exon is flip and which is flop: UNVERIFIED** (not checked against annotation or literature). The figure labels exons by coordinate.
- **Types shown:** 64 cell groups: excitatory 47, inhibitory 15, non-neuronal 2. 57 of 64 have an interval (excitatory 41, inhibitory 14, non-neuronal 2).
- **Medians, LeafletFA / raw:**

  | | J1 | J2 |
  |---|---|---|
  | Excitatory | 0.154 / 0.155 | 0.338 / 0.339 |
  | Inhibitory | 0.067 / 0.104 | 0.468 / 0.431 |
  | Non-neuronal | 0.420 / 0.396 | 0.041 / 0.025 |

- **Gallery row 4:** descriptive; no direction is stated (Sommer 1990: rat, abstract only).

### d. *Fgfr2* IIIb vs IIIc (TMS, `ENSMUSG00000030849.18_atse_3`; 2 junctions)
- **Junction shown:** J1 = the IIIb junction `chr7_130199905_130200982_-` (`c10_catalog.FGFR2_IIIB`). The other junction of this ATSE (IIIc) is 1 − ψ.
- **Context:** IIIb/IIIc is a split MXE (`atse_3` upstream pair, `atse_4` downstream; DEFINITIONS.md 1.6). Only `atse_3` is shown.
- **Types shown:** 31: epithelial 15, mesenchymal 11, other 5. With an interval: 13 (epithelial 10, mesenchymal 1, other 2).
- **Medians, LeafletFA / raw:** epithelial 0.919 / 1.000; mesenchymal **0.405 / 0.000**; other 0.400 / 0.001.
- **Mesenchymal types:** raw is 0 and LeafletFA about 0.4. This is the smoothing diagnosed on 2026-11-06 (cells without reads enter at the program prior; numbers.tex `FgfrUncoveredMesenchymal` 0.35–0.63 for uncovered cells).
- **Direction source:** Bebee et al. 2015 (mouse Esrp1/2 knockout epidermis) and Warzecha et al. 2009; gallery row 1.

### e. *Nrxn1* SS4 (Allen, `ENSMUSG00000024109.18_atse_2`; complex, 10 junctions)
- **Junction shown:** J1 `chr17_90991378_90992114_-`. It is the junction with the most TRAIN reads among the 5 junctions with an end at the annotated 30-nt SS4 exon (chr17:90,992,115–90,992,144; `c10_known_biology_fix.SS4`): 3 junctions enter the exon from the left, 2 leave on the right.
- **Limit:** J1's ψ is the share of the ATSE's reads on this one junction. **It is not SS4 inclusion (PSI).** The other SS4 junctions and the skipping junctions carry the rest.
- **Types shown:** 57 cell groups: excitatory 41, inhibitory 14, non-neuronal 2. With an interval: 21 (14 / 6 / 1).
- **Medians, LeafletFA / raw:** excitatory 0.541 / 0.485; inhibitory 0.510 / 0.457; non-neuronal 0.254 / 0.247.
- **Gallery row 3 direction** (Nguyen 2016, mouse: SS4 skipped in pyramidal neurons, included in PV and SST interneurons) cannot be read from this single junction. **Decision for Karin:** show SS4 inclusion as the sum of the SS4 junctions on one side, or keep the one-junction rule. The sum would need a new interval; the check 12 interval is per junction.

### f. *Ptprc* variable exons (TMS, `ENSMUSG00000026395.16_atse_2`; linked MXE, 10 junctions)
- **Junction shown:** J1 `chr1_138115642_138116953_-`. Rule: among the lead junctions of the check 10.4 Ptprc calls, the one with the most TRAIN reads.
  - The candidates were `chr1_138117025_138126302_-` (13 calls), `chr1_138115642_138116953_-` (13 calls) and `chr1_138115642_138126302_-` (1 call).
- **Types shown:** 81: B cell 12, T cell 16, other immune 35, non-immune 18. With an interval: 51 (B 11, T 12, other immune 28, non-immune 0).
- **Medians, LeafletFA / raw:** B cell **0.397 / 0.190**; T cell 0.489 / 0.442; other immune 0.516 / 0.476; non-immune 0.338 / 0.359.
- **B cells:** LeafletFA sits above raw (shrinkage toward the shared programs, as in the browser-gap diagnosis).
- **10.4 (passed):** 27 B/T calls, 25 confirmed in TEST; B cells lower than the rest at the lead junction in TEST. Calls are not shown, because 10.2 failed.
- **Gallery row 2:** descriptive (the B vs T direction is not verified).

## Limits to state in the caption
- TEST cells are new cells of the same mice (per-cell split). Generalization to new donors is untested.
- The interval exists only where held-out data are deep enough: 28 % of TMS catalog rows and 35 % of Allen rows overall.
- Seed-0 fits only (K = 20). Programs are not shown; they did not reproduce across seeds (checks 3, 8).
- No cell-type-specificity calls are shown (10.2 and 10.7 failed).
