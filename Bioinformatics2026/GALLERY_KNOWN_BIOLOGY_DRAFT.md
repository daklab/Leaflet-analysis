# Supplement gallery: known-biology event list (FINAL, Karin 2026-11-10; no values looked up until Karin approves)

Status: approved with changes (Karin 2026-11-09). No LeafletFA or raw value has been looked up against this list. The gallery rule was
approved as proposed (2026-11-06; APPLICATION_NOTE_FIGURE_PLAN.md). Every citation's volume and pages must be verified against PubMed or the
journal before use (section "Citation check" below).

- Directions come from the cited papers, written before any LeafletFA or raw value was looked at.
- Citations were written from memory, without a literature search. Each needs a check (volume, pages, and that the paper
  states this direction in mouse) before use. Volume and pages verified 2026-11-09 (below); direction in mouse not yet checked.
- "ATSE" is the catalog event. Only the four main-figure events are already mapped. The rest are mapped by annotation
  (exon coordinates, GENCODE vM19) after approval, before usage is read.
- The first four are the main figure (c to f); the gallery adds the others.

| # | Gene | Event | Atlas | Expected direction | Evidence type | Citation |
|---|---|---|---|---|---|---|
| 1 | *Fgfr2* | IIIb vs IIIc exon (MXE), atse_3 | TMS | IIIb higher in epithelial types, IIIc in mesenchymal (ESRP1/2) | mouse in vivo (Esrp1/2 knockout epidermis, E18.5: full IIIb to IIIc switch) + cell lines (human) | Warzecha et al. 2009, Mol Cell 33:591–601 (PMID 19285943); Bebee et al. 2015, eLife 4:e08954 (PMID 26371508) |
| 2 | *Ptprc* | Variable exons 4–6 (CD45RA/RB/RC), atse_2 | TMS | **Descriptive:** differs between immune types (the B vs T direction is not verified; Hermiston 2003 paywalled) | not verified | Hermiston, Xu and Weiss 2003, Annu Rev Immunol 21:107–137 (PMID 12414720) |
| 3 | *Nrxn1* | SS4 exon (90 nt, chr17:90,163,792–90,163,881), `atse_7` (SpliceVI §111 definition; corrected 2026-11-11, was the note-4 30-nt exon in `atse_2`, an early Nrxn1α site, not SS4) | Allen | **SS4 skipped in excitatory (pyramidal) neurons, included in PV and SST interneurons** | mouse cell types (hippocampus and neocortex, ribosome-tagged CamK2, PV, SST neurons) | Nguyen et al. 2016, eLife 5:e22757 (PMID 27960072); Traunmüller et al. 2016, Science 352:982–986 (PMID 27174676; paywalled, abstract only) |
| 4 | *Gria2* | Flip vs flop (MXE), atse_3 | Allen | **Descriptive:** flip/flop ratio differs between neuron types | rat brain regions (CA1 vs CA3; abstract only) | Sommer et al. 1990, Science 249:1580–1585 (PMID 1699275) |
| 5 | *Cd44* | Variable exons **v8–v10** | TMS | Inclusion higher in epithelial than mesenchymal types (ESRP) | cell lines (human; v8–v10) + mouse in vivo (Esrp1/2 knockout epidermis converts Cd44v to Cd44s; all 10 variable exons as one block) | Warzecha et al. 2009 (as row 1); Bebee et al. 2015 (as row 1) |
| 6 | *Enah* | Exon 11a | TMS | Included in epithelial, skipped in mesenchymal types (ESRP) | cell lines (human) + mouse in vivo (an "Esrp enhanced" Enah event in epidermis; the exon is not named in the text) | Warzecha et al. 2009 (as row 1); Bebee et al. 2015 (as row 1) |
| 7 | *Ctnnd1* | **Exons 2 and 3** (mesenchymal p120 isoform) | TMS | **Exons 2–3 included more in mesenchymal than in epithelial types** (ESRP silences them) | cell lines (human); Bebee 2015 does not mention *Ctnnd1* | Warzecha et al. 2009 (as row 1) |
| 8 | *Ptbp2* | Exon 10 | TMS (brain) or Allen vs non-neuronal | Included in neurons, skipped (with NMD) in neural progenitors and non-neuronal cells (PTBP1) | development (mouse cell lines, embryonic neural progenitors) | Boutz et al. 2007, Genes Dev 21:1636–1652 (PMID 17606642) |
| 9 | *Dlg4* (PSD-95) | Exon 18 | TMS brain neurons vs non-neuronal | Repressed by PTBP1/2 early, included late in neuronal maturation | development (mouse embryonic cortex, cultured neurons, N2a) | Zheng et al. 2012, Nat Neurosci 15:381–388 (PMID 22246437) |
| 10 | *Pkm* | Exon 9 (PKM1) vs exon 10 (PKM2), MXE | TMS | **Descriptive:** M1/M2 exon usage differs between cell types (no tissue direction in the abstract; paper paywalled) | not verified (rat; abstract only) | Noguchi, Inoue and Tanaka 1986, J Biol Chem 261:13807–13812 (PMID 3020052) |

Decisions (Karin 2026-11-09):
- Row 2: TMS has one broad "T CELL" type per tissue and no naive vs activated/memory split (type names checked, no values), so
  the B vs T direction is kept with the CD45 review (Hermiston, Xu and Weiss 2003) instead of Oberdoerffer 2008.
- Row 3: direction stays "differs between neuron groups"; Nguyen et al. 2016 added.
- Row 4: kept, descriptive.
- Row 10: Noguchi et al. 1986 replaces David 2010.
- Row 11 (*Src* N1): **dropped** (Karin 2026-11-10). Irimia 2014 defines microexons as 3–15 nt and the *Src* exon in the catalog is 18 nt.
- No alt 3′ or alt 5′ event: the gallery caption must state this gap.

Gallery caption note (to carry into the Supplement): the list has no alternative 3′ or alternative 5′ splice-site event, because no
known-biology example with a stated direction was available for those classes.

## Citation check (2026-11-09; PubMed esummary/efetch, read-only)
- All 10 citations match PubMed in authors, year, journal, volume and pages. Volumes and pages were missing for Hermiston 2003,
  Nguyen 2016 and Noguchi 1986; they are now filled in from PubMed.
- Noguchi 1986: PubMed lists no DOI. The paper is in **rat**.
- Zheng 2012: PubMed date 2012 Jan 15 is the online date; print is vol 15(3).
- Nguyen 2016: the abstract supports the row-3 summary (principal-cell isoforms depend on Slm2; "most parvalbumin-positive (PV+)
  interneurons lack Slm2").
- Row 11 flag (Irimia 3–15 nt vs 18-nt *Src* exon): resolved by dropping the row (Karin 2026-11-10).
- Still not checked: that each paper states the listed direction **in mouse** (only volume and pages were in scope).

## Direction check: does the cited paper state the listed direction in mouse cell types? (2026-11-10)

**Method:** two read-only literature passes over full text (PMC / Europe PMC) where open, otherwise the abstract only. The *Fgfr2*, CD44, ENAH and CTNND1 quotes from Warzecha 2009 were re-checked against the saved full text (PMC2702247).

**Recent-paper rule:** Furlanis 2019, Joglekar 2021 and Joglekar 2024 were searched for every row (main text and figure legends; supplementary tables not searched). **None states a direction for any row**, so none is added.

| # | Gene | Paper (access) | What it states (verbatim, or figure) | Organism / cells | In mouse cell types? | Match |
|---|---|---|---|---|---|---|
| 1 | *Fgfr2* | Warzecha 2009 (full text) | "The FGFR2-IIIb splice variant is exclusive to epithelial cells, while FGFR2-IIIc is mesenchymal" (Introduction, citing Orr-Urtreger 1993). Data: Figs. 1E, 2A, 3, 5 | human cell lines (293T, PNT2, HMLE, MDA-MB-231). Only mouse data: Esrp1 mRNA in situ in mouse tissues (Fig. 2B), not *Fgfr2* splicing | **no** | direction and ESRP1/2: yes |
| 2 | *Ptprc* | Hermiston, Xu and Weiss 2003 (**paywalled**, abstract only) | abstract gives no B vs T direction | — | cannot check | cannot check |
| 3 | *Nrxn1* SS4 | **Nguyen 2016** (full text) | "While in CamK2 Ribo cells the cassette exon at AS4 was largely skipped there was a high level of alternative exon inclusion in PV Ribo cells." Also SST interneurons enriched for AS4+ (Fig. 2—figure supplement 1F) | **mouse** hippocampus (and neocortex, Fig. 2E); CamK2+ pyramidal vs PV+ and SST+ interneurons | **yes** | states more than "differs" (see proposal 1) |
| 3 | *Nrxn1* SS4 | Traunmüller 2016 (**paywalled**, abstract only) | abstract: SLM2 at "glutamatergic synapses in the mouse hippocampus"; no cell-type direction | mouse | cannot check | cannot check |
| 4 | *Gria2* | Sommer 1990 (**paywalled**, abstract only) | "the alternatively spliced messenger RNAs show distinct expression patterns in rat brain, particularly in the CA1 and CA3 fields" (all four AMPA subunits; *Gria2* not singled out in the abstract) | **rat** | **no** | fits "differs" (descriptive) only |
| 4 | *Gria2* | Furlanis 2019 (full text) | flip/flop stated for **Gria1**, not *Gria2*: "CamK2- and Grik4-positive glutamatergic neurons show differential usage of the mutually exclusive exons 16 and 17 in Gria1" (Fig. 3b) | mouse CA1 vs CA3 pyramidal neurons | not for *Gria2* | no |
| 5 | *Cd44* | Warzecha 2010 (summarized page read) | **CD44 not mentioned** (text, Table 1, legends; supplementary tables not checked) | human | — | wrong citation |
| 5 | *Cd44* | Warzecha 2009 (full text) | "Inclusion of several 'variable' exons of CD44 transcripts, including exons 8–10 (V8–V10), has been shown to be epithelial-specific (Ponta et al.)"; ESRP knockdown lowers v8–v10 inclusion (Fig. 3E) | human PNT2, HMLE | **no** | partly: v8–v10, not v2–v10 |
| 6 | *Enah* | Warzecha 2010 | exon 11a not mentioned (ENAH appears only as a motif location) | human | — | wrong citation |
| 6 | *Enah* | Warzecha 2009 (full text) | "ENAH contains an alternative exon, 11a, that is predominantly included in epithelial cell lines and skipped in mesenchymal cells (Pino et al.)"; knockdown lowers 11a inclusion (Fig. 3E) | human cell lines | **no** | direction: yes |
| 7 | *Ctnnd1* | Warzecha 2009 (full text) | "expresses mesenchymal specific splice variants that contain alternative exons 2 and 3 (Keirsebilck et al.)"; knockdown "induced expression of the mesenchymal isoform" (Fig. 3E); ESRPs "silence mesenchymal exons (CTNND1)" | human PNT2, HMLE | **no** | **framing inverted:** the regulated exons (2, 3) are mesenchymal, included more in mesenchymal cells; there is no "epithelial exon" |
| 7 | *Ctnnd1* | Warzecha 2010 | generic: "The ESRPs regulate the splicing of exons that yield distinct epithelial versus mesenchymal p120-catenin isoforms"; no exon named | human | no | partly |
| 8 | *Ptbp2* | Boutz 2007 (full text) | "nPTB mRNA is expressed in many cells, but a majority of these transcripts are missing exon 10 and are degraded by the NMD pathway"; abstract: "nPTB protein is specifically expressed in post-mitotic neurons, whereas PTB is restricted to neuronal precursor cells (NPC), glia, and other nonneuronal cells" | mouse cell lines (P19, N1E-115, N2a), primary embryonic mouse NPCs, development | **partly** (mouse, but not adult glia, microglia or endothelial cells) | partly |
| 9 | *Dlg4* | Zheng 2012 (full text) | abstract: "PTBP1 and PTBP2 repressed Psd-95 (also known as Dlg4) exon 18 splicing ... The loss of first PTBP1 and then of PTBP2 during embryonic development allowed splicing of exon 18 and expression of PSD-95 late in neuronal maturation" | mouse embryonic cortex E12–E18.5, cultured cortical neurons, N2a | **partly** (developmental, not neurons vs adult non-neuronal types) | partly |
| 10 | *Pkm* | Noguchi 1986 (**paywalled**, abstract only) | abstract: "two adjacent exons coded the M1- and M2-specific sequences"; **no tissue direction in the abstract** | rat | **no** | cannot check direction |
| 10 | *Pkm* | Joglekar 2021 (full text) | Pkm named for "distinguishing neurons and non-neurons" (Fig. S9d) and "neurodevelopmental exon inclusion switches in Pkm" (Fig. 6d, S12); **which exon in which cell type is not stated** | mouse brain | partly | no direction, so not added |

**Bibliographic data for the recent-paper rule:**

| Paper | PMID | DOI | Citation |
|---|---|---|---|
| Furlanis E, Traunmüller L, Fisher G, Scheiffele P (2019) | 31451803 | 10.1038/s41593-019-0465-5 | Nat Neurosci **22(10):1709–1717**. Karin's guess (22:1709) is correct |
| Joglekar A et al. (2021) | 33469025 | 10.1038/s41467-020-20343-5 | Nat Commun 12(1):463 |
| Joglekar A et al. (2024) | 38594596 | 10.1038/s41593-024-01616-4 | Nat Neurosci 27(6):1051–1063 |

**Paywalled; Karin to put PDFs in `refs/`:**
- Traunmüller L et al. 2016, Science 352:982–986 (PMID 27174676), row 3.
- Sommer B et al. 1990, Science 249:1580–1585 (PMID 1699275), row 4.
- Hermiston ML, Xu Z, Weiss A 2003, Annu Rev Immunol 21:107–137 (PMID 12414720), row 2.
- Noguchi T, Inoue H, Tanaka T 1986, J Biol Chem 261:13807–13812 (PMID 3020052; DOI 10.1016/S0021-9258(18)67091-7), row 10.
- Saved open texts: `Warzecha 2009` full text in the session scratchpad only (not in the repo).

**Proposals, not applied (each changes a row; Karin decides):**
1. **Row 3:** Nguyen 2016 states a direction in mouse cell types: SS4 (AS4) mostly **skipped in CamK2 pyramidal neurons** and **included in PV (and SST) interneurons**. Your rule was "a direction only if Traunmüller 2016 is read and states it". Traunmüller is paywalled; Nguyen states it. Option: change row 3 to "SS4 inclusion higher in PV and SST interneurons than in excitatory neurons (Nguyen 2016)".
2. **Rows 5–7:** cite **Warzecha 2009**, not 2010. The 2010 paper does not mention CD44 or ENAH 11a, and p120 only generically.
   - Row 5: narrow to v8–v10.
   - Row 7: re-sign as "exons 2–3 (mesenchymal isoform) included more in mesenchymal than in epithelial types".
3. **Rows 1, 5, 6, 7:** the ESRP evidence is from human cell lines. **No cited paper shows these directions in mouse cell types.** Options:
   - Cite the primary sources Warzecha cites (Orr-Urtreger 1993 is mouse embryo work for *Fgfr2*; Ponta 2003 for CD44; Pino 2008 is human pancreatic cancer lines for ENAH);
   - or keep the rows, labeled "direction from human cell lines".
4. **Rows 8, 9:** mouse, but developmental and cell-line evidence. Keep them, labeled "neurons vs non-neuronal and progenitor cells (development)".
5. **Row 4:** stays descriptive. Sommer is rat (abstract only), and Furlanis covers Gria1, not *Gria2*.
6. **Rows 2, 10:** unverifiable until the PDFs are in `refs/`. Noguchi is rat.

## Decisions applied (Karin 2026-11-10); list final
- Row 3: direction from Nguyen 2016 (SS4 skipped in excitatory/pyramidal neurons, included in PV and SST interneurons). This is main-figure panel e.
- Rows 5–7 cite Warzecha 2009, not Warzecha 2010. Row 5 is narrowed to v8–v10. Row 7 is re-signed (exons 2–3 higher in mesenchymal types).
- Rows 1, 5, 6: Bebee et al. 2015 (eLife 4:e08954, PMID 26371508, DOI 10.7554/eLife.08954) checked before labeling them "cell lines".
  It gives mouse in vivo evidence (E18.5 epidermis, wild type vs Esrp1/2 knockout), quotes verified against the full text (PMC4566030):
  - Fgfr2: "in DKO epidermis Fgfr2 underwent a full switch from Fgfr2-IIIb (epithelial) to Fgfr2-IIIc (mesenchymal), indicating that the
    production of the Fgfr2-IIIb isoform is fully dependent upon the Esrps in epidermis (Figure 3B)."
  - Cd44: "inclusion of 10 exons in the mouse variable region of the Cd44 transcript is associated with the epithelial isoforms (Cd44v),
    and the exclusion of these exons generates the mesenchymal Cd44s isoform"; "KO of Esrp1 and Esrp2 results in full conversion of Cd44v
    to the mesenchymally expressed Cd44s (short) isoform" (Fig. 4—figure supplement 1B).
  - Enah: "We evaluated an Esrp enhanced splicing event in Enah" (Fig. 3C); the exon is not named in the text.
  - Ctnnd1: not mentioned (0 hits; supplementary rMATS tables and figure images not searched).
  - Limit: Bebee compares wild type with Esrp knockout **within epidermis**. No mesenchymal tissue was measured, so the
    epithelial vs mesenchymal direction itself rests on the cited earlier work (Orr-Urtreger 1993 for Fgfr2; Brown 2011 for Cd44).
- Rows 8–9: labeled developmental evidence.
- Rows 2, 4, 10: descriptive. Rows 2 and 10 change only if PDFs Karin adds to `refs/` state a direction.
- Optional PDFs for `refs/` (none blocks anything): Traunmüller 2016, Sommer 1990, Hermiston 2003, Noguchi 1986.
- **Still no values looked up against this list until Karin approves it.**

## Correction (Karin 2026-11-11)
- Row 3 was drawn on `atse_2` (the check 10.4 note-4 "single annotated 30-nt exon" rule), which is an early Nrxn1α site
  (residues ~259–274, consistent with SS#1), not SS4. It is redrawn on the SS4 exon of the SpliceVI §111 event list (fixed
  2026-10-06): 90-nt exon in `atse_7`, inclusion `chr17_90162433_90163791_-`, skipping `chr17_90162433_90208322_-`.
- The expected direction (Nguyen 2016) is unchanged; it now applies to the right exon.
