# Application Note figure: caption FACTS (no caption prose; updated 2026-11-10 after Karin's review)

**Build:** `paper/figures/panels/fig_appnote.py` via `make_figures.py --figure AN`, sbatch. Output is in `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_bioinformatics/paper_figures/`.

| Output | Contents |
|---|---|
| `figAN/` | `figAN.pdf` (vector, editable text), `figAN.tiff` (350 dpi), `figAN_review.png`, `panels/AN_{a..f}.pdf`, `source_data/` |
| `figAN_grey_noiv/` | **Variant for comparison** (`FIGAN_GREY_NOIV=1`): in c–f, every type without a validated interval is drawn in light gray (point and ring). Same events and rules otherwise |

- **Size:** 183 × 164 mm (limit 165).
- **Fonts:** 6 pt minimum; 7 pt labels.
- **Colors:**
  - LeafletFA teal; raw pseudobulk gray.
  - **Panel b atlases in neutral black (TMS, filled square) and gray (Allen, open circle)**, not lineage or dataset colors.
  - Lineage strips use the colors of `cell_type_maps/mouse_lineage_map.json`.
- **Interval lines:** 0.4 pt at 55 % opacity in c–f.
- **Exon annotation everywhere:** **GENCODE vM19** (Ensembl 94, GRCm38; GTF header dated 2018-08-30), the annotation of the catalog build.
  - **MANE:** MANE Select exists only for human, and vM19 carries no Ensembl_canonical tag.
  - **Exon numbering:** exons are numbered in the named transcript, in transcription order. The GTF used has no `exon_number` attribute, so numbers are computed from exon order. The named transcript is the longest CCDS transcript that holds the exon (exceptions are noted per panel).

## Inputs (md5)

| Input | md5 |
|---|---|
| TMS / Allen seed-0 `fit.npz` | 140c17e6af58dd9466607e864a0fb983 / 974153b331677c443933f67de77a9f9f |
| `c10/TMS_s0_estimates.npz` / `Allen_s0_estimates.npz` (point) | 738f74ce8dff156fe3d9232aa8599a19 / a1d787502bf1d53793b24ddc5f0f3e15 |
| `c12/export/TMS_intervals.npz` / `Allen_intervals.npz` (check 12 interval) | 098bc7fc97be13af92bcf025c07747d9 / 0b5c146f815ccac335b3d7b1749498b0 |
| `TMS_train_tables.npz` / `Allen_train_tables.npz` (TRAIN reads, cells, mice) | 08ccbc1f89830c5909286c7c7a9672c0 / 21a702fbb80f0e9d59a513bb062450dc |
| `catalog_junctions.parquet` / `catalog_events.json` | 81c745d3b5f29bf20e8ae9f65cf6fa56 / 3b49732b6a1a51fb457f8c6e544b8870 |
| `tsv/c10_TMS_s0_accuracy.tsv` / `c10_Allen_s0_accuracy.tsv` | baa4dd9dabda407b87128615090f82c4 / 5e123d9527f3d4859ef399f91e59687e |
| `tsv/c10_TMS_s0_ptprc_direction.tsv` | 2bab4adf8795cbede4ec313015dc6d00 |
| `fig1b_catalog.tsv` / `fig1c_event_types.tsv` (ledger sources) | 04c635ae429b7a7fa2b25763e966f7f0 / 3d85527e8a95c94dc2c2e9f1d5f3eaaa |
| `data/figAN_event_junctions.tsv` (junctions by annotation, `paper/scripts/event_annotation.py`; log `figAN_event_annotation.log`) | 5e4bb80b606cd762305cd7ae20be3d38 |
| `cell_type_maps/mouse_lineage_map.json` (version 2026-11-10) | 03c408a3c486545e513a8fd615dbb15f |

## a. Catalog and model
- **Counts per atlas** (TRAIN + TEST, June 2026 build; asserted equal to the `numbers.tex` macros):

  | | TMS | Allen |
  |---|---|---|
  | Cells | 78,135 | 54,571 |
  | Junctions | 77,562 | 77,204 |
  | ATSEs | 24,390 | 24,162 |
  | Genes | 11,210 | 11,050 |
  | Cell types / groups (≥ 10 TRAIN cells) | 132 | 68 |

- **Event types of the 24,475 ATSEs** (asserted equal to `EventShare*`; DEFINITIONS.md 1.6): alt 3′ 5,810 (24 %), alt 5′ 4,867 (20 %), skipped exon 5,708 (23 %), complex 8,090 (33 %).
- **Linked MXEs are inside "Complex"** (checked 2026-11-10 on the catalog):
  - all 1,034 linked-MXE ATSEs have ≥ 5 junctions (range 5 to 43), and every ATSE with ≥ 4 junctions is "complex" by the rule;
  - the browser's 5 classes give complex 7,056 + linked MXE 1,034 = **8,090**, the ledger's "complex".
  - Split MXEs (e.g. *Fgfr2* IIIb/IIIc) appear as an alt 3′ and an alt 5′ pair, not in "complex".
- The model schematic is a placeholder for Karin. The panel a legend covers c–f and names the annotation (GENCODE vM19, GRCm38).

## b. Check 10.1 accuracy (pre-registered; passed in both atlases)
- **Error:** the **total variation distance** between the estimate and the held-out TEST pseudobulk usage over the junctions of an ATSE, TV = ½ Σ_j |ψ̂_j − ψ_j^TEST|. Plotted: error(LeafletFA) − error(raw TRAIN pseudobulk), with 95 % CI; below 0 = LeafletFA closer to held-out cells.
- **Axis label:** "Held-out error, LeafletFA − raw" (no "TV").
- **Split:** per cell (70/30, stratified by broad type × age, seed 42), not per mouse. TEST cells are new cells of the same mice.

| Atlas | Bin | Units | Types | LeafletFA − raw (95 % CI) |
|---|---|---|---|---|
| TMS | overall | 1,038,848 | 130 | −0.0072 (−0.0094, −0.0050) |
| TMS | 10–49 TRAIN cells | 51,876 | 31 | −0.0194 (−0.0242, −0.0153) |
| TMS | 50–199 | 307,672 | 47 | −0.0155 (−0.0178, −0.0129) |
| TMS | 200–999 | 452,293 | 38 | −0.0065 (−0.0090, −0.0044) |
| TMS | ≥ 1,000 | 227,007 | 14 | **+0.0054 (+0.0029, +0.0083)** |
| TMS | 0 TRAIN reads | 3,731 | 130 | −0.0138 (−0.0166, −0.0107) |
| TMS | 1–19 | 15,617 | 130 | −0.0625 (−0.0681, −0.0571) |
| TMS | 20–99 | 84,817 | 130 | −0.0243 (−0.0278, −0.0211) |
| TMS | ≥ 100 | 934,683 | 130 | −0.0047 (−0.0067, −0.0029) |
| Allen | overall | 763,823 | 68 | −0.0021 (−0.0039, −0.0005) |
| Allen | 10–49 TRAIN cells | 39,807 | 11 | −0.0116 (−0.0143, −0.0090) |
| Allen | 50–199 | 262,244 | 30 | −0.0077 (−0.0088, −0.0066) |
| Allen | 200–999 | 224,979 | 15 | −0.0022 (−0.0038, −0.0002) |
| Allen | ≥ 1,000 | 236,793 | 12 | **+0.0057 (+0.0044, +0.0074)** |
| Allen | 0 TRAIN reads | 385 | 29 | −0.0095 (−0.0169, +0.0010) |
| Allen | 1–19 | 5,111 | 68 | −0.0543 (−0.0599, −0.0482) |
| Allen | 20–99 | 145,013 | 68 | −0.0122 (−0.0134, −0.0110) |
| Allen | ≥ 100 | 613,314 | 68 | +0.0007 (−0.0010, +0.0022) |

- **Large types:** in types with ≥ 1,000 TRAIN cells, **raw TRAIN pseudobulk is slightly closer to held-out cells than LeafletFA**, in both atlases (+0.0054 TMS, +0.0057 Allen; both CIs above 0). LeafletFA's gain is in small types and at few reads; the largest gain is at 1–19 TRAIN reads (−0.063 TMS, −0.054 Allen).
- At 0 TRAIN reads, raw falls back to the TRAIN population ψ (the check 10 comparator rule).

## c–f. Rules (fixed; `fig_appnote.py` docstring)
- **Cell types:** every type with ≥ 20 TRAIN reads and ≥ 3 TRAIN cells with reads at the ATSE, grouped by lineage, sorted by LeafletFA ψ within a lineage.
- **Lineages:** `GenomeBiology_Submission2026/cell_type_maps/mouse_lineage_map.json`, now **one definition for the browser** (event-page "Group by: Lineage", version preview-2026-11-10) **and the figures**.
  - TMS: one lineage per broad cell type. Epithelial and Mesenchymal are asserted equal to the check 10 keyword lists.
  - Allen: majority class of the group's TRAIN cells (Glutamatergic → Excitatory, GABAergic → Inhibitory, else Non-neuronal). Inhibitory groups split by cluster-name prefix: Pvalb → PV, Sst → SST, Vip → VIP, Lamp5 → Lamp5, else Other inhibitory.
  - Each panel uses one collapse scheme of the map: c `allen`, d `tms_epithelial`, e `allen_inhibitory_split`, f `tms_immune`.
- **Junctions:** chosen by annotation only (`paper/scripts/event_annotation.py`, GENCODE vM19; no usage value read), except f, which uses the 10.4 rule below.
- **Marks:**
  - LeafletFA point: check 10, seed 0; the mean over the type's TRAIN cells of the posterior-mean per-cell usage.
  - Interval: the check 12 (a)+(b) **95 % predictive interval for the type's held-out pseudobulk ψ at its read depth**. It is not an interval for a true cell-type ψ, and it is drawn only where TEST cells have ≥ 100 reads and ≥ 10 cells with reads.
    - τ² (100–499 / ≥ 500 TEST reads): TMS 0.01024 / 0.00256; Allen 0.00128 / 0.00032.
  - Raw ring: TRAIN junction reads / ATSE reads; ring area ∝ log junction reads.
- **Compression at low coverage** (smoothing, diagnosed 2026-11-06):
  - Where a type's TRAIN cells have few or no reads at the event, cells without reads enter the type mean at their program prior. This pulls LeafletFA ψ toward the shared programs and away from raw. It is visible in d and f, and it is kept visible (Karin 2026-11-10).
  - These are also the types that mostly lack a validated interval (gray in the variant).
  - At the top-ranked events of the browser ranking, LeafletFA |Δψ| is about 0.12 below held-out TEST (median of 20 random TMS pairs, −0.121).

### c. *Gria2* flip/flop (Allen, `ENSMUSG00000033981.14_atse_3`; linked MXE, 7 junctions, minus strand)
- **Flip and flop identified from the annotation and the sequence:**
  - Exon chr3:80,690,404–80,690,518 (115 nt) is used by **Gria2-202, ENSMUST00000107745.7 (CCDS)**. It translates (reverse strand) to …**TP**VNLAVLKL**S**EQG**V**LDKLKNKWWYDKGEC**GAKDSG**SK = **flip**.
  - Exon chr3:80,691,320–80,691,434 (115 nt) is used by **Gria2-201, ENSMUST00000075316.9 (CCDS)**. It translates to …**NA**VNLAVLKL**N**EQG**L**LDKLKNKWWYDKGEC**GSGGGD**SK = **flop**.
  - Residues per UniProt P19491 (rat GRIA2): isoform Flop displayed; Flip = VSP_000111 to VSP_000114 (NA→TP, N→S, L→V, SGGGD→AKDSG).
  - Gria2-203 (ENSMUST00000192463.1, NMD) holds both exons.
  - In transcription order (minus strand) **the flop exon is 5′ of the flip exon** in this annotation.
- **Junctions:**
  - **J1** `chr3_80689351_80690403_-` = into the flip exon;
  - **J2** `chr3_80689351_80691319_-` = into the flop exon.
  - Both run from the shared downstream exon (chr3:80,689,103–80,689,351), so J1 ψ ≈ flip use and J2 ψ ≈ flop use within the ATSE.
- **Types:** 64 cell groups: excitatory 47, inhibitory 15, non-neuronal 2. With an interval: 57 (41 / 14 / 2).
- **Medians, LeafletFA / raw:**

  | | Flip (J1) | Flop (J2) |
  |---|---|---|
  | Excitatory | 0.154 / 0.155 | 0.338 / 0.339 |
  | Inhibitory | 0.067 / 0.104 | 0.468 / 0.431 |
  | Non-neuronal | 0.420 / 0.396 | 0.041 / 0.025 |

- **Gallery row 4:** descriptive; no direction stated (Sommer 1990: rat, abstract only).

### d. *Fgfr2* IIIb vs IIIc (TMS, `ENSMUSG00000030849.18_atse_3`; 2 junctions; event unchanged)
- **J1** = the IIIb junction `chr7_130199905_130200982_-` (check 10 definition), into the IIIb exon chr7:130,199,758–130,199,905 (148 nt) or its 99-nt variant ending at the same site (GENCODE vM19; CCDS transcript Fgfr2-211, ENSMUST00000120187.8). IIIc is 1 − ψ.
- **Types:** 31: epithelial 15, mesenchymal 11, other 5. With an interval: 13 (10 / 1 / 2).
- **Medians, LeafletFA / raw:** epithelial 0.919 / 1.000; **mesenchymal 0.405 / 0.000**; other 0.400 / 0.001.
- **Low-coverage compression:**
  - In mesenchymal types raw IIIb is 0, but LeafletFA sits at about 0.4 (+0.405 median).
  - Only 1 of 11 mesenchymal types has a validated interval.
  - numbers.tex `FgfrUncoveredMesenchymal` 0.35–0.63 (seed means over 12 types, uncovered cells) is the same mechanism.
  - The lineage direction (epithelial above mesenchymal) holds in LeafletFA and raw.
- **Direction source:** Bebee et al. 2015 (mouse Esrp1/2 knockout epidermis) and Warzecha et al. 2009.

### e. *Nrxn1* SS4 skipping (Allen, `ENSMUSG00000024109.18_atse_2`; complex, 10 junctions)
- **Rule (annotation; Karin 2026-11-10):** J1 = the SS4-skipping junction, i.e. the junction joining the two exons that are consecutive in the annotated transcripts lacking SS4 and that straddle the 30-nt SS4 exon (chr17:90,992,115–90,992,144).
  - Those transcripts: Nrxn1-203 (ENSMUST00000159419.1), Nrxn1-205 (ENSMUST00000160467.7), Nrxn1-210 (ENSMUST00000161637.7).
  - All three give one junction: **`chr17_90704367_91087954_-`**, joining the exon starting at 91,087,955 to the exon ending at 90,704,367.
  - 7 annotated transcripts include SS4. In them, SS4 is flanked by an 18-nt exon (90,995,461–478) and a 12-nt exon (90,991,367–378), so **the skipping junction also skips those two small exons**.
- **Not shown:** a summed SS4 estimate.
- **Types:** 57 cell groups: excitatory 41; inhibitory split PV 2, SST 3, VIP 3, Lamp5 2, other inhibitory 4; non-neuronal 2. With an interval: 21 (excitatory 14; PV 1, SST 1, VIP 2, Lamp5 1, other 1; non-neuronal 1).
- **Medians, LeafletFA / raw:**

  | Lineage | LeafletFA / raw |
  |---|---|
  | Excitatory | 0.013 / 0.011 |
  | PV | 0.019 / 0.018 |
  | SST | 0.019 / 0.022 |
  | VIP | 0.013 / 0.014 |
  | Lamp5 | 0.013 / 0.014 |
  | Other inhibitory | 0.018 / 0.031 |
  | Non-neuronal | 0.063 / 0.129 |

- **Expected direction** (Nguyen 2016, mouse): skipping higher in excitatory, lower in PV and SST. **It is not seen at this junction.** The junction carries about 1–2 % of the ATSE's reads in every neuronal group, and PV and SST are not lower than excitatory.
  - Only 2 PV and 3 SST groups pass the support rule (Pvalb, Pvalb Vipr2; Sst, Sst Chodl, Sst HPF).

### f. *Ptprc* variable exons (TMS, `ENSMUSG00000026395.16_atse_2`; linked MXE, 10 junctions)
- **J1 rule (unchanged):** among the lead junctions of the check 10.4 Ptprc calls, the one with the most TRAIN reads.
  - Candidates: `chr1_138117025_138126302_-` (13 calls), `chr1_138115642_138116953_-` (13 calls), `chr1_138115642_138126302_-` (1 call).
- **J1 by its exons:** `chr1_138115642_138116953_-` joins **exon 7 (chr1:138,116,954–138,117,025, 72 nt) to exon 8 (chr1:138,115,619–138,115,642, 24 nt)** of Ptprc-207 (ENSMUST00000183301.7, CCDS). Ptprc-207 is the only vM19 transcript with all three variable exons:
  - exon 4 = 129 nt (RA), chr1:138,122,456–584;
  - exon 5 = 147 nt (RB), chr1:138,120,020–166;
  - exon 6 = 141 nt (RC), chr1:138,117,754–894.
  - Exon 7 is itself alternative: present in Ptprc-203, -207, -213, absent from Ptprc-204.
- **What J1 ψ means:** J1 is used whether or not the variable exons are included. Its ψ is its share of all junction reads in this ATSE, which also holds the junctions into and between exons 4–6. Counting junctions in this ATSE:
  - a transcript with exons 4, 5, 6 and 7 uses 5 junctions (3→4, 4→5, 5→6, 6→7, 7→8), one of them J1;
  - a transcript with exon 7 and no variable exon uses 2 (3→7, 7→8).
  - So **higher J1 ψ means less variable-exon inclusion**, assuming equal reads per junction of a molecule. This is a structural argument, not a measurement.
- **Types:** 81: B cell 12, T cell 16, other immune 35, non-immune 18. With an interval: 51 (11 / 12 / 28 / 0).
- **Medians, LeafletFA / raw:** **B cell 0.397 / 0.190**; T cell 0.489 / 0.442; other immune 0.516 / 0.476; non-immune 0.338 / 0.359.
  - B cells lowest (more variable-exon inclusion) in raw; in LeafletFA the B-cell values are compressed upward (+0.207 median), the low-coverage compression above.
- **B-cell group composition** (TMS cell_ontology_class, all B CELL types):
  - **TRAIN:** B cell 3,034; naive B cell 1,577; precursor B cell 381; immature B cell 277; late pro-B cell 141; early pro-B cell 28; **plasma cell 17**.
  - **TEST:** plasma cell 5.
  - All plasma cells are in **Lung_B CELL** (17 of its 183 TRAIN cells, 9 %). Lung_B CELL is shown in f (LeafletFA 0.403, raw 0.193, interval present).
  - The B-cell group also includes developing B cells (pro-, pre- and immature B cells, mostly marrow).
  - No other plasma cells or plasmablasts are in any B-cell type; "plasmacytoid dendritic cell" (5 TRAIN cells, Lung) is in DENDRITIC CELL.
- **10.4 (passed):** 27 B/T calls, 25 confirmed in TEST; calls are not shown (10.2 failed).

## Limits to state in the caption
- TEST cells are new cells of the same mice (per-cell split). Generalization to new donors is untested.
- The interval exists only where held-out data are deep enough: 28 % of TMS catalog rows and 35 % of Allen rows overall.
- Seed-0 fits only (K = 20). Programs are not shown (they did not reproduce; checks 3, 8). No cell-type-specificity calls are shown (10.2, 10.7 failed).
