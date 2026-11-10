# Supplement gallery: known-biology event list (approved by Karin with changes, 2026-11-09)

Status: approved with changes (Karin 2026-11-09). No LeafletFA or raw value has been looked up against this list. The gallery rule was
approved as proposed (2026-11-06; APPLICATION_NOTE_FIGURE_PLAN.md). Every citation's volume and pages must be verified against PubMed or the
journal before use (section "Citation check" below).

- Directions come from the cited papers, written before any LeafletFA or raw value was looked at.
- Citations were written from memory, without a literature search. Each needs a check (volume, pages, and that the paper
  states this direction in mouse) before use. Volume and pages verified 2026-11-09 (below); direction in mouse not yet checked.
- "ATSE" is the catalog event. Only the four main-figure events are already mapped. The rest are mapped by annotation
  (exon coordinates, GENCODE vM19) after approval, before usage is read.
- The first four are the main figure (c to f); the gallery adds the others.

| # | Gene | Event | Atlas | Expected direction | Citation |
|---|---|---|---|---|---|
| 1 | *Fgfr2* | IIIb vs IIIc exon (MXE), atse_3 | TMS | IIIb higher in epithelial types, IIIc in mesenchymal (ESRP1/2) | Warzecha et al. 2009, Mol Cell 33:591–601 (PMID 19285943) |
| 2 | *Ptprc* | Variable exons 4–6 (CD45RA/RB/RC), atse_2 | TMS | Inclusion higher in B cells than T cells | Hermiston, Xu and Weiss 2003, Annu Rev Immunol 21:107–137 (CD45 review; PMID 12414720) |
| 3 | *Nrxn1* | SS4 exon, atse_2 | Allen | Differs between neuron groups (a direction only if Traunmüller 2016 is read and states it) | Traunmüller et al. 2016, Science 352:982–986 (PMID 27174676); Nguyen et al. 2016, eLife 5:e22757 (SLM2 drives principal-cell neurexin isoforms; most PV interneurons lack SLM2; PMID 27960072) |
| 4 | *Gria2* | Flip vs flop (MXE), atse_3 | Allen | Flip/flop ratio differs between neuron types; no single direction stated for excitatory vs inhibitory, so descriptive only | Sommer et al. 1990, Science 249:1580–1585 (PMID 1699275) |
| 5 | *Cd44* | Variable exons v2–v10 | TMS | Inclusion higher in epithelial than mesenchymal types (ESRP) | Warzecha et al. 2010, EMBO J 29:3286–3300 (PMID 20711167) |
| 6 | *Enah* | Exon 11a | TMS | Included in epithelial, skipped in mesenchymal types (ESRP) | Warzecha et al. 2010 (as row 5) |
| 7 | *Ctnnd1* | Epithelial exon (p120 isoform) | TMS | Epithelial isoform higher in epithelial types (ESRP) | Warzecha et al. 2010 (as row 5) |
| 8 | *Ptbp2* | Exon 10 | TMS (brain) or Allen vs non-neuronal | Included in neurons, skipped in non-neuronal cells (PTBP1-driven NMD) | Boutz et al. 2007, Genes Dev 21:1636–1652 (PMID 17606642) |
| 9 | *Dlg4* (PSD-95) | Exon 18 | TMS brain neurons vs non-neuronal | Included in neurons, skipped in non-neuronal cells (PTBP1/2) | Zheng et al. 2012, Nat Neurosci 15:381–388 (PMID 22246437) |
| 10 | *Pkm* | Exon 9 (PKM1) vs exon 10 (PKM2), MXE | TMS | PKM1 higher in muscle, heart and brain; PKM2 higher in proliferating and immune cells | Noguchi, Inoue and Tanaka 1986, J Biol Chem 261:13807–13812 (rat; M1/M2 from one gene by alternative splicing; PMID 3020052) |

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
