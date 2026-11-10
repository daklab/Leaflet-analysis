# Supplementary gallery: caption FACTS (no caption prose; 2026-11-10)

**Build:** `paper/figures/panels/fig_gallery.py` via `make_figures.py --figure SG`, sbatch.
- **Output:** `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_bioinformatics/paper_figures/figSG/` (`figSG.pdf`, `figSG.tiff`, `figSG_review.png`, `panels/SG_{a..j}.pdf`, `source_data/figSG_<gene>.tsv`).
- **Size:** 183 × 220 mm (limit 220).
- **List:** `GALLERY_KNOWN_BIOLOGY_DRAFT.md`, final (Karin 2026-11-10, 2eae6cc).
  - Every row is shown, whatever it gives.
  - Each panel title carries the row's evidence type and expected direction as written in the list.
  - No value was looked at before the list was final.
- **Panel style:** the same as the Application Note figure (structure on top, ψ per type below, check 12 interval where validated, raw TRAIN ring).
  - Cell types: ≥ 20 TRAIN reads and ≥ 3 TRAIN cells with reads at the ATSE.
  - Lineages: `cell_type_maps/mouse_lineage_map.json`.
  - Exons: GENCODE vM19 (Ensembl 94, GRCm38).
  - Rules and inputs: `FIG_APPNOTE_CAPTION_FACTS.md`.
- **Junctions by annotation:** `paper/scripts/event_annotation.py` → `data/figAN_event_junctions.tsv` (md5 5e4bb80b606cd762305cd7ae20be3d38), with the transcripts used in `figAN_event_annotation.log`.
- **Alt 3′ / alt 5′:** **no alternative 3′ or 5′ splice-site event is in the list.** No known-biology example with a stated direction was available for those classes; say so in the caption.

| Panel | Row | Event; junctions (annotation rule) | Atlas, scheme | Types (with interval) | Medians, LeafletFA / raw, by lineage |
|---|---|---|---|---|---|
| a | 1 *Fgfr2* | IIIb junction `chr7_130199905_130200982_-` (as main d) | TMS, epithelial | 31 (13) | epithelial 0.919 / 1.000; mesenchymal 0.405 / 0.000; other 0.400 / 0.001 |
| b | 2 *Ptprc* | exon 7→8 of Ptprc-207 `chr1_138115642_138116953_-` (10.4 rule, as main f) | TMS, immune | 81 (51) | B 0.397 / 0.190; T 0.489 / 0.442; other immune 0.516 / 0.476; non-immune 0.338 / 0.359 |
| c | 3 *Nrxn1* | SS4-skipping junction `chr17_90704367_91087954_-` (as main e) | Allen, inhibitory split | 57 (21) | excitatory 0.013 / 0.011; PV 0.019 / 0.018; SST 0.019 / 0.022; VIP 0.013 / 0.014; Lamp5 0.013 / 0.014; other inhibitory 0.018 / 0.031; non-neuronal 0.063 / 0.129 |
| d | 4 *Gria2* | flip `chr3_80689351_80690403_-`, flop `chr3_80689351_80691319_-` (as main c) | Allen, class | 64 (57) | flip: excitatory 0.154 / 0.155, inhibitory 0.067 / 0.104; flop: excitatory 0.338 / 0.339, inhibitory 0.468 / 0.431 |
| e | 5 *Cd44* v8–v10 | J1 Cd44s junction `chr2_102828610_102853014_-` (exon 5→16 of Cd44-201, skips v1–v10; standard isoform Cd44-203 ENSMUST00000099673.8); J2 v10→next constant exon `chr2_102828610_102831329_-` (Cd44-201 ENSMUST00000005218.14) | TMS, epithelial | 93 (52) | J1: epithelial **0.349 / 0.003**, mesenchymal 0.910 / 0.954, other 0.897 / 0.990. J2: epithelial 0.116 / 0.181, mesenchymal 0.013 / 0.007, other 0.015 / 0.002 |
| f | 6 *Enah* 11a | 63-nt exon chr1:181,911,597–659 (only in Enah-202, ENSMUST00000111024.9, appris_alternative_2, not CCDS; **11a by length, UNVERIFIED**). J1 inclusion `chr1_181911659_181913306_-`, J2 skip `chr1_181908376_181913306_-` | TMS, epithelial | 26 (4) | J1: epithelial 0.227 / 0.357, mesenchymal 0.054 / 0.000, other 0.157 / 0.000. J2: epithelial 0.553 / 0.296, mesenchymal 0.893 / 1.000, other 0.678 / 1.000 |
| g | 7 *Ctnnd1* exons 2–3 | **no ATSE of *Ctnnd1* in the catalog** (no junction with its gene id) | — | — | — |
| h | 8 *Ptbp2* exon 10 | exon 10 of Ptbp2-201 (ENSMUST00000029780.11, CCDS), chr3:119,724,612–645, 34 nt. J1 exon 9→10 `chr3_119724645_119726065_-`, J2 exon 9→11 skip `chr3_119724186_119726065_-` | TMS, neural | 39 (8) | J1: neuron (1 type) 0.448 / 0.475, glia (2) 0.457 / 0.466, non-neural (36) 0.380 / 0.387. J2: neuron 0.017 / 0.000, glia 0.030 / 0.006, non-neural 0.079 / 0.129 |
| i | 9 *Dlg4* exon 18 | exon 18 of Dlg4-203 (ENSMUST00000108588.8, CCDS), chr11:70,043,658–767, 110 nt. J1 exon 17→18 `chr11_70042743_70043657_+`, J2 exon 17→19 skip `chr11_70042743_70044275_+` | TMS, neural | 16 (3) | J1: neuron (1) 0.456 / 0.509, glia (2) 0.415 / 0.475, non-neural (13) 0.460 / 0.464. J2: neuron 0.021 / 0.000, glia 0.033 / 0.000, non-neural 0.101 / 0.083 |
| j | 10 *Pkm* M1/M2 | **no ATSE of *Pkm* in the catalog** (no junction with its gene id); the panel also holds the gallery legend | — | — | — |

**Readings to state factually, without a verdict:**
- ***Fgfr2* (a):** epithelial above mesenchymal in raw and LeafletFA. Mesenchymal types are compressed toward 0.4 (low coverage).
- ***Cd44* (e):**
  - In raw, epithelial types skip the variable region far less (Cd44s junction 0.003) than mesenchymal (0.954) and use the v10 junction more (0.181 vs 0.007): the expected direction.
  - LeafletFA compresses the epithelial Cd44s value up to 0.349.
- ***Enah* (f):**
  - Raw inclusion is higher in epithelial (0.357) than mesenchymal (0.000) types: the expected direction.
  - Only 4 of 26 types have an interval (all epithelial).
- ***Nrxn1* (c):** the expected direction (skipping higher in excitatory, lower in PV and SST) is not seen at this junction (about 1–2 % of ATSE reads in every neuronal group).
- ***Ptbp2* (h), *Dlg4* (i):** only 1 TMS neuron type passes the support rule, so the neuron vs non-neuronal comparison rests on one type.
  - Raw skipping of exon 10 / exon 18 is 0 in that neuron type and higher in non-neural types (0.129 / 0.083): the expected direction, on one neuron type.
- ***Ctnnd1* (g), *Pkm* (j):** no ATSE of either gene is in the catalog, so they cannot be shown.
