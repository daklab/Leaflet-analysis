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
| 11 | *Src* | N1 microexon (18 nt; chr2:157,458,843–157,458,860, +; GENCODE vM19), `ENSMUSG00000027646.15_atse_3` (class complex) | TMS brain | Included in neurons, skipped elsewhere | Irimia et al. 2014, Cell 159:1511–1523 (PMID 25525873); **see the 3–15 nt flag below** |

Decisions (Karin 2026-11-09):
- Row 2: TMS has one broad "T CELL" type per tissue and no naive vs activated/memory split (type names checked, no values), so
  the B vs T direction is kept with the CD45 review (Hermiston, Xu and Weiss 2003) instead of Oberdoerffer 2008.
- Row 3: direction stays "differs between neuron groups"; Nguyen et al. 2016 added.
- Row 4: kept, descriptive.
- Row 10: Noguchi et al. 1986 replaces David 2010.
- Row 11: fixed as *Src* N1 by annotation only (no usage read; `LeafletFA_validation/gallery_annotation/src_n1.py`, job 23121076).
  - GENCODE vM19 *Src* has one 18-nt exon, chr2:157,458,843–157,458,860 (+). It is taken as N1 by length (six codons); the exon number
    was not checked against Irimia 2014.
  - The catalog has both inclusion junctions (157,458,633→N1 start; N1 end→157,462,709) and the skipping junction
    (157,458,633→157,462,709), all in `ENSMUSG00000027646.15_atse_3`, which has 5 junctions and the catalog classes as complex
    (not skipped exon: two more junctions, N1 end→157,461,942 and 157,461,975→157,462,709,
    use a second exon between N1 and the downstream exon). Row kept.
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
- **Row 11 flag:** the Irimia 2014 abstract defines microexons as 3–15 nt. The *Src* exon taken as N1 is 18 nt, outside that
  definition, and the abstract does not mention *Src*. Either check the full text for *Src* N1, cite a *Src*-specific source for
  the 18-nt N1 exon, or drop row 11. Karin to decide; nothing was looked up beyond PubMed records.
- Still not checked: that each paper states the listed direction **in mouse** (only volume and pages were in scope).
