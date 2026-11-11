# LeafletFA main.tex screen (2026-11-11; no manuscript edits)

- **Text screened:** `~/LeafletFA_overleaf/main.tex` at 78436ab (394 lines) plus `shared_data_catalog.tex` (the `\input` at `:121`). Line numbers refer to `main.tex` unless a file is named.
- **Scope:** prose only. Red `\unsure{}` text is screened only where it is a stale claim. Comments are not screened. Nothing was rewritten; each finding names the fix, not new wording.
- **Reference points:** `OUTLINE_PREP.md` (budgets, check 12 facts), `FIG_APPNOTE_CAPTION_FACTS.md` (panel facts), `RELEASE_MANIFEST.md` (fits, timings), `VALIDATION_REPORT.md` as quoted in the comments.
- **Tags:** STYLE = style tell; STALE = contradicted by a later result or decision; MISSING = a point the note needs; CITE = citation issue; NUM = number or notation that does not agree with another place.

## 1. Word counts against the OUTLINE_PREP budgets

Counted by script: prose only; comments, red text, equations, citations and section titles removed; ledger macros count as one word.

| Section | Lines | Budget | Now | Difference |
|---|---|---|---|---|
| Abstract | 68-74 | 150 | 15 | **−135** (all red) |
| Introduction | 79-115 | 200 | 256 | **+56** |
| Data and event catalog | `shared_data_catalog.tex` | 250 | 221 (file's own count); 247 by script | −29 / −3 |
| ATSEmapper | 124-142 | 150 | 156 | +6 |
| Model and fitting | 144-181 | 200 | 267 | **+67** |
| Cell-type usage and intervals | 183-205 | 150 | 77 | −73 |
| The catalog | 210-219 | 120 | 42 | −78 |
| Accuracy | 221-232 | 180 | 11 | −169 |
| Limits | 234-279 | 170 | 31 | −139 |
| Known biology and portal | 281-295 | 120 | 129 | +9 |
| Scale | 297-303 | 60 | 77 | **+17** |
| Discussion | 313-354 | 250 | 230 | −20 |
| **Total (abstract + body)** | | **2,000** | **about 1,520** | **−480** |

- **Over budget:** Introduction, Model, Scale, ATSEmapper, Known biology (+155 together).
- **Under budget:** Abstract, Accuracy, Limits, Catalog, Cell-type usage (−594 together). These are the sections that need new prose, so the real total will rise by about 600 words once they are filled to budget. **That is about 2,100, over 2,000 by about 100.**
- **Cut sources already in the text:**
  - Duplicate PCR-duplicate sentence (`:97`, `:321`): about 20 words.
  - Duplicate ATSE definition (`:135-138` and the shared text): about 40 words.
  - Duplicate counts sentence (`:218` and the shared text): about 25 words.
  - "SVI in Pyro" twice (`:150-152`, `:174`): about 25 words.
  - Programs and depth paragraphs in the Discussion (`:324-331`): about 80 words, if the note stays catalog-only.
- **Discussion is under budget only because it holds program material** that the note no longer delivers (see 3).

## 2. Section structure

- STYLE / framing: **Results and Discussion headings, three data subsections inside Results, and a 250-word Discussion are a research-article layout.** The Bioinformatics Application Note layout is not verified (OUTLINE_PREP section 1 says VERIFY; the OUP guide was not read). Settle this first, because the budgets above assume it.
- STYLE: section title `:234` "Limits of cell-type-specific calls and intervals" uses "calls" and "cell-type-specific", the words tied to the failed 10.2 rule. See "calls" in 3.
- MISSING: no subsection for the browser, though the portal is the main way a reader uses the catalog. It is half a sentence of red text (`:295`).

## 3. Sentence by sentence

### Abstract (`:68-74`)
- `:70` Motivation, `:71` Results, `:72` Availability, `:73` Contact: all red placeholders; no prose to screen.
- `:74` Supplementary line: fine.
- MISSING in the abstract once written: the interval (check 12), the atlas-only scope of the interval, the browser URL, and the Zenodo DOI. The abstract plan at `:70` ("pseudobulk is noisy for small or shallow types") is the 10.1 result, and 10.1 reverses in large types (see 4).

### Introduction (`:79-115`)
- `:83-85` "Alternative splicing (AS) enables single genes to generate multiple RNA and protein variants ... in response to developmental and environmental signals."
  - STYLE: textbook opener; participial "allowing cells to ..."; three-item list "function, localization, and stability". Cited (Baralle 2017), so the claim is supported.
- `:85-86` "AS can also impact gene dosage by altering mRNA stability, translation efficiency, or inducing nonsense-mediated decay."
  - STYLE: "impact" as a verb; non-parallel list ("altering X, Y, or inducing Z").
  - CITE: uncited.
  - Cut candidate: this sentence carries no point the note uses.
- `:91-94` "Full-length scRNA-seq methods like Smart-seq2 are required for cell-type splicing analysis, as they capture internal splicing events missed by ... 10x."
  - STYLE: "are required" overstates (3′ and long-read data also give splicing information).
  - NUM: "Smart-seq2" here, but the Allen atlas is **Smart-seq v4** (`shared_data_catalog.tex`). The same slip is at `:111` and `:216`.
- `:93-94` "Such approaches have revealed a layer of cell-type-specific splicing information, complementary to gene expression, ..."
  - STYLE: "revealed a layer", "complementary to gene expression".
  - Cites four papers (Song 2017, Lukacsovich 2019, Booeshaghi 2021, Benegas 2022); not checked that each used full-length data.
- `:97` "Reads on a junction are largely PCR duplicates (no UMIs): even deeply covered events carry only a few effective molecules."
  - MISSING: no citation and no pointer to the evidence (Supplementary Note 1 / the companion paper).
  - STALE (internal): `:168-169` calls the same floor "a modeling choice, not a measurement". The Introduction states the cause as fact. The evidence is indirect (coverage lower where γ_e sits at the floor; check 12 (a) alone fails at 0.82-0.90).
  - Repeated at `:321`.
- `:101-102` "No existing software characterizes single-cell AS across these atlas-scale datasets without relying on cell-type labels or gene-level summaries."
  - STALE: the note's own estimator averages over **cell-type labels** (`:185`); the criticism of SpliZ applies to the deliverable.
  - STYLE: absolute claim ("No existing software"). LeafCutter and scQuint, both run per cell type on these atlases, are not named (`:105` is a gap).
- `:102-103` "Tools like SpliZ require predefined cell-type annotations and provide only gene-level scores [Tabula Sapiens, SpliZ]."
  - CITE: Tabula Sapiens is the human atlas SpliZ was applied to, not a source for the tool's properties; human content was dropped elsewhere. Drop it here.
- `:103-104` "... methods such as PsiX are restricted to annotated exons and rely on gene expression to define cell similarity."
  - Fine. Keep the characterization to what the PsiX paper states.
- `:110-111` "To quantify how much alternative splicing varies across single cells in mouse, and to do so reliably despite the sparsity ..., we developed ATSEmapper and LeafletFA."
  - STALE: the note delivers cell-type usage, not variation across single cells.
  - STYLE: "reliably" overclaims (10.2 and 10.7 failed; intervals cover only 28 % / 35 % of rows).
  - STYLE: Genome Biology framing ("how much AS varies").
- `:111-112` "ATSEmapper uniformly processes existing Smart-seq2 atlases, assembling splice junctions into ATSEs across N mouse cells."
  - NUM: Smart-seq2 vs Smart-seq v4 (above).
  - STYLE: "ATSE" is used before it is defined (defined at `:136`); the abstract uses it too. "uniformly" is vague.
- **`:112-114` "LeafletFA, a Bayesian factor model, decomposes the resulting sparse junction counts into latent splicing programs without prior cell-type labels and imputes per-cell junction usage."**
  - STALE (line 113): **programs did not reproduce** (checks 3 and 8; stable for only SP5, SP7, SP8, SP15 of 20 on the K = 20 atlas fit), so "decomposes into latent splicing programs" cannot lead the note. The Discussion (`:324`) says the opposite.
  - STALE: "imputes per-cell junction usage" is not an evaluated or shipped output; the shipped output is the cell-type estimate (`:185-188`).
  - Action: say what is shipped (cell-type usage with the interval where validated); keep the programs as the model's internal representation, with a pointer to the limits.
- `:114` "Together these tools yield a harmonized, denoised single-cell splicing atlas."
  - STYLE: "harmonized, denoised"; a stated takeaway.
  - STALE: the atlas is **cell-type level**, not single-cell. "Denoised" is supported by 10.1 on average (−0.0072 TMS, −0.0021 Allen) and reverses in types with ≥ 1,000 TRAIN cells (+0.0054, +0.0057).
- `:115` red: "what this note provides". MISSING; this is where check 12, the atlas-only interval and the browser belong.

### Data and event catalog (`shared_data_catalog.tex`)
- "We used two full-length single-cell atlases of the mouse: ..."
  - Fine; ages "3, 18 and 24 months" are data descriptors, not an aging claim.
- Event-catalog sentence 3 ("An alternative transcript structure event ... singletons are not part of the catalog") is one sentence of about 80 words.
  - STYLE: split it. Content is correct.
- "The catalog contains N junctions in N ATSEs across N genes, measured in N cells ..."
  - NUM: the same counts are stated again at `:218`.
  - Note: 24,475 ATSEs is the union over both atlases (TMS 24,390; Allen 24,162).
- "Each atlas was split at random into 70 % training and 30 % test cells ... the split is per cell, not by mouse ..."
  - Fine, and it is the key limit for 10.1; keep it in the main text.

### ATSEmapper (`:124-142`)
- `:127-128` "Splice junctions were extracted from BAM files using regtools and processed using our ATSEmapper pipeline (URL)."
  - STYLE: passive; "our". URL to confirm (`:142`).
  - CITE: regtools is dated 2023 in `references.bib`; check against the paper cited.
- `:128-130` "Each identified splice junction was validated using reference genome FASTA files to confirm the presence of canonical SS motifs ..."
  - STYLE: "validated" overstates a motif check.
  - MISSING: the annotation name and version (GENCODE vM19, GRCm38) appears nowhere in the main text; the figure legend names it.
- `:129-130` "GTF annotation files were used to classify each SS as either annotated ... or novel."
  - Fine.
- `:135-140` ATSE definition and the AnnData sentence.
  - NUM / STYLE: "SS" means splice site here, and **"SS4" (`:293`) is an exon**. Spell out "splice site" or rename.
  - STYLE: the ATSE is defined here and again in the shared text; keep one.
  - `:138-139` "ATSEs capture various forms of alternative splicing including cassette exons, alternative 5′/3′ SS, and alternative first/last exon usage." conflicts with the shared text and panel a: **MXE pairs are one ATSE only if another junction links them; otherwise two ATSEs; linked MXEs are inside "complex"**. Align the list with the event types in panel a (alt 3′, alt 5′, skipped exon, complex).
- `:142` red: inputs, outputs, run time. MISSING.

### Model and fitting (`:144-181`)
- `:148-152` "LeafletFA decomposes the sparse cell-by-junction count matrix into a set of continuous, latent splicing programs (SPs). ..."
  - STALE: programs (see `:112-114`).
  - NUM: the text uses **Φ and Ψ**; the equations use **φ_c, ψ_kj, π**. One notation.
- `:150-152` "We fit LeafletFA using Stochastic Variational Inference via the Pyro ... ensuring LeafletFA is highly scalable and computationally efficient for atlas-level datasets."
  - STYLE: "ensuring ... highly scalable and computationally efficient" is an unsupported causal claim; the numbers are in the Scale subsection. "atlas-level".
  - Duplicate: SVI in Pyro again at `:174`.
- `:156-157` Data definition: fine. NUM: **N** is the ATSE total (N_ce) here and the number of cells in "N/B" at `:175`.
- `:167-168` "At m = 2 this is the Beta-binomial; for larger m it is the exact multivariate generalization, the same likelihood as LeafCutter's DM GLM."
  - NUM: **m is not defined** (junctions in the ATSE).
  - CITE: LeafCutter is cited at `:137`; cite again or say "LeafCutter's".
- `:168-171` "γ_e is the effective number of independent molecules ... floored at 2, a modeling choice, not a measurement: on the production fits 81 % of the N per-ATSE γ_e sit exactly at 2 in TMS and 68 % in Allen ..."
  - NUM: the denominator N is `\CatalogATSEsMouse` = 24,475, the union over both atlases; the percentages are per atlas (24,390 and 24,162). Use the per-atlas macros.
  - NUM: 81 % and 68 % are hand-typed, not ledger macros (comment `:166`).
  - STYLE: "sit exactly at 2" informal.
  - Keep: this is a real limit and it is stated honestly here.
- `:174-176` "Inference is stochastic variational inference in Pyro with an AutoNormal guide ... Convergence is judged on the ψ contrast trace, not the ELBO, which is noise-dominated at this scale."
  - STYLE: "ψ contrast trace" is undefined jargon. "at this scale" vague.
- `:177-178` "All reported analyses use K = 20, 16,200 steps × 8,192 cells ..., learning rate 0.025 with exponential decay to 0.1×, and float64 φ."
  - Checked against `RELEASE_MANIFEST.md` §1: matches. Add: **seed 0** is the primary fit; the figure and the browser show seed 0 only.
- `:180` red: entry point. `:181` comment: E-step cut candidate.
  - MISSING: the held-out cells are scored with frozen globals and an inferred φ (E-step, 3,000 steps); this is how TEST cells are used in check 10, and the text never says so.

### Cell-type junction usage and intervals (`:183-205`)
- `:185-188` Estimator and comparator: still correct (check 10).
  - NUM: ρ̃ uses γ_e of the event containing j; write γ_{e(j)} or say "the event e of j".
  - STYLE: fine.
  - MISSING: the sentence that decides how to read the estimate: **cells without reads at the event enter the mean at their program prior** (smoothing; top-ranked differences about 0.12 below held-out; `FIG_APPNOTE_CAPTION_FACTS.md`). It is in the comments, not the text.
- **`:191-192` "The interval is 2.5-97.5 % over 200 joint draws from q(ψ) and q(φ); in each draw, each cell type's mice are also resampled ... This is a posterior interval (not calibrated; see Supplementary)."**
  - STALE: this is the check 10 interval, **which is no longer shipped**; its coverage was 0.136-0.346 vs 0.95.
  - Replace with the check 12 interval (below).
- `:193` red "amendment 7 (draft, not approved)": STALE; approved 2026-11-06 and run.
- **Check 12 (MISSING from the text; facts only in `:194-203`):**
  - Definition: a 95 % **predictive** interval for the held-out pseudobulk ψ of the cell type at its read depth; **not** an interval for a true cell-type ψ.
  - Construction: m ± 1.96 √(v + τ²_b), clipped to [0, 1]; m and v from 200 DM simulations; τ²_b per atlas and TEST read bin, set on one mouse half.
  - Result: coverage **0.950-0.963** in both atlases, both bins, both directions; (a) alone **0.82-0.90**; check 10 interval 0.136-0.346.
  - Reach: **28 % of TMS rows and 35 % of Allen rows** (held-out ≥ 100 reads and ≥ 10 cells with reads); other rows show ψ with `interval_status`.
  - Width: wide in TMS (median 0.225 at 100-499 TEST reads).
  - **Atlas only:** users' own fits get ψ (`leafletfa.catalog.celltype_psi`) but no interval, because τ² needs held-out cells. Karin asked for this sentence in Methods and in the tutorial.
  - Shipped: export done (f879a6d, b65f8ec), so the sentence can name the files.

### Results: The catalog (`:210-219`)
- `:215-216` "To build a comprehensive catalog of AS events, we compiled splice junction data from large-scale Smart-seq2-based atlases."
  - STYLE: "comprehensive" (singletons and low-coverage junctions are excluded), "large-scale", "compiled"; duplicates the Data subsection and the Introduction.
  - NUM: Smart-seq2 vs Smart-seq v4.
- `:216-217` "These short-read, full-length transcript coverage methods are well-suited for robust splice junction identification."
  - STYLE: "well-suited for robust" (banned word); third statement of the same point (`:91-94`, `shared_data_catalog.tex`). Cut.
- `:217-218` "We obtained N junctions in N ATSEs (N genes) in mouse (Section ...)."
  - NUM: duplicate of the shared text. Cut one.
- `:219` red: which numbers lead. MISSING. Ledger options in `OUTLINE_PREP.md` section 3.

### Results: Accuracy (`:221-232`)
- `:223` "Test accuracy is measured on new cells of the same mice."
  - STYLE: "accuracy" has no object (what is compared to what). 
  - MISSING: 10.1 itself. Facts for the prose:
    - TMS **−0.0072** (−0.0094, −0.0050); Allen **−0.0021** (−0.0039, −0.0005).
    - Smallest types: −0.0194 / −0.0116. 1-19 TRAIN reads: −0.0625 / −0.0543.
    - **Types with ≥ 1,000 TRAIN cells: +0.0054 TMS, +0.0057 Allen; raw is slightly closer.** Panel b shows it. The text must say so.
  - Panel b is built; the prose is not.

### Results: Limits (`:234-279`)
- `:236` red: all prose missing. Needs, in 170 words: 10.2 and 10.7 failed; Allen donor confounding; the LeafCutter control (neither method passes); smoothing at top-ranked events; check 13 (pooled estimate fails, current estimate kept); check 12 coverage and reach.
- `:279` red "γ_e is at the floor ... parameter-only widths are below the median error; the mouse bootstrap adds little."
  - STALE: the second and third clauses describe the check 10 interval. The γ_e floor clause stays true; it duplicates `:168-171`.

### Results: Known biology and the portal (`:281-295`)
- **`:283-284` "LeafletFA's ability to recover biologically relevant splicing patterns in real data is exemplified by the Fgfr2 gene, a canonical example of mutually exclusive exon usage driven by epithelial-mesenchymal transition."**
  - STYLE: "ability to recover ... is exemplified"; "canonical".
  - STALE: **panel d contradicts it.** In mesenchymal types raw IIIb is 0 and LeafletFA sits near 0.4 (median +0.405); only 1 of 11 mesenchymal types has a validated interval. LeafletFA keeps the direction (epithelial above mesenchymal) but does not recover the switch.
  - CITE: Warzecha 2009 is human cell lines (the mouse in-vivo evidence is Bebee 2015, not cited). Also pre-2015 (see 5).
- `:286` red "By modeling the entire dataset collectively, LeafletFA leverages information across all cells and junctions to resolve splicing decisions that would be ambiguous in sparse raw data."
  - STYLE: "leverages" (banned word); a general claim.
  - STALE: it is the claim the smoothing diagnosis contradicts: for cells without reads the estimate is the program prior, not a resolved decision. Cut, or state as "borrows from the programs, and pulls toward them".
- `:287` red Fgfr2 result: MISSING. Facts: epithelial above mesenchymal in raw and in LeafletFA; mesenchymal compressed toward 0.4; uncovered mesenchymal cells 0.35-0.63 (seed mean over 12 types); covered 0.05-0.15 in 11 of 12 types.
- `:290` red "Ptprc: B cells have lower usage of the lead junction than the rest in test cells, T cells mostly higher."
  - STYLE: "lead junction" is not interpretable without the structure (J1 = exon 7 → 8; **higher J1 means less variable-exon inclusion**, a structural argument, not a measurement).
  - CITE: none; the CD45 review (Hermiston 2003) is in the gallery list and is paywalled.
  - STALE: "calls" machinery (27 calls, 25 confirmed). Panel f shows raw (B 0.190 vs T 0.442) and LeafletFA (B 0.397, compressed upward).
  - Note: the B-cell group includes plasma cells in Lung_B CELL (17 of 183 TRAIN cells).
- **`:293` "Nrxn1-3 SS4 results ... For all three genes the rule is ATSE-level: a differing pair shows that the ATSE holding SS4 differs, not that the SS4 junctions drive the difference."**
  - STALE: **panel e shows no SS4 difference** (skipping junction 1-2 % everywhere; it also skips two small exons). The sentence now needs to say what the 10.4 result is and is not. The ATSE-level wording is correct and consistent with panel e.
  - NUM: "SS4" is used as a name with no definition and clashes with "SS" = splice site. The "rule" is the call rule.
  - Depends on the panel e decision (variant A or B).
- **Gria2: MISSING entirely.** Panel c exists. Facts: flip/flop by annotation and sequence (flip = Gria2-202 exon, flop = Gria2-201 exon; J1 into flip, J2 into flop); medians flip 0.154 / 0.155 excitatory, 0.067 / 0.104 inhibitory; flop 0.338 / 0.339 and 0.468 / 0.431; gallery row 4 is descriptive, no direction stated in mouse (Sommer 1990 is rat, abstract only).
- `:295` red: portal description and URL. MISSING. Facts: preview-2026-11-10; gene, event, compare and one-vs-all pages; raw TRAIN ψ, reads, cells and mice beside each estimate; interval only where validated; downloads carry fit and interval md5; no call columns; Lineage group-by.

### Results: Scale (`:297-303`)
- `:299-300` "Fits run on GPU: 0.05 s/step for TMS and 0.14 s/step for mouse Allen on an RTX PRO 6000, i.e. 15-40 min per fit ..."
  - NUM: 16,200 × 0.05 s = 13.5 min and 16,200 × 0.14 s = 38 min; the manifest's primary fits took **10.6 min (TMS seed 0) and 31.6 min (Allen seed 0)**, and seeds 1-2 took 23 and 69 min. "15-40 min" fits neither list.
  - STYLE: "mouse Allen" is a leftover from the human dataset; the GPU name should be the full "NVIDIA RTX PRO 6000 Blackwell".
- `:301` "CPU is about 10× slower (6 s/step for TMS on 16 cores)."
  - **NUM: 6 s / 0.05 s = 120×, not 10×.** One of the two numbers is wrong or refers to something else.
- `:301-302` "At atlas size (N cells), the full 16,200-step fit took 119 min, 0.27 s/step, with a peak host memory of 40.7 GB."
  - **NUM: 16,200 × 0.27 s = 73 min, not 119 min.** If the 119 min includes the E-step (3,000 steps) it is still 86 min. Check against the 10.6 record before keeping both.
- `:303` red: STALE. scQuint (job 22725257) and LeafCutter (TMS_full array 22780123) benchmarks have run; results not in the text.

### Figure and caption (`:305-310`)
- `:307` still includes `Figures/fig1_catalog.pdf` (the June figure).
- `:308` red caption lists panels a-e (model, representation checks); the built figure is a-f (variant A) or a-e (variant B).
- MISSING: the caption (facts in `FIG_APPNOTE_CAPTION_FACTS.md`); the model schematic in panel a is a placeholder.

### Discussion (`:313-354`)
- `:318` "We developed LeafletFA, a scalable Bayesian factor model for analyzing AS at single-cell resolution."
  - STYLE: restates the Introduction; "scalable"; "single-cell resolution" (the output is cell-type level).
- `:321` "Reads on a junction are largely PCR duplicates, and the floor on γ_e permits less overdispersion than the data show."
  - STYLE: duplicate of `:97`; "than the data show" has no number behind it in the text. Point to the check 12 (a)-alone failure.
- `:324-327` "Individual programs are mostly not reproducible across seeds. Cell-type usage of programs is reproducible, while junction composition is reproducible only for the programs that meet the reproduction rule. Make program-level statements only for reproduced programs. Across 15 TMS seeds, N clusters recur in at least 60 % of fits (...)."
  - STALE vs the Introduction (`:112-114`), which leads with programs.
  - NUM: "the reproduction rule" is undefined (centred ψ match ≥ 0.8 across seeds and live; atlas K = 20: SP5, SP7, SP8, SP15 only).
  - STYLE: imperative to the reader ("Make ... statements only for ...").
  - VERIFY: "cell-type usage of programs is reproducible" against check 8 and the finding that the per-cell encoding is degenerate.
  - Budget: the whole paragraph is about 60 words that belong in the Supplement if the note is catalog-only.
- `:330-331` "Recovery fails at EasySci-like depth (counts thinned to 1.1 %; N programs recovered): do not use LeafletFA programs fitted on data that sparse."
  - CITE: "EasySci" is uncited and undefined.
  - STYLE: colon-imperative.
  - STALE: same program scope question.
- **`:334-335` "LeafCutter is anti-conservative on mouse-split nulls; LeafletFA's posterior test made no calls, consistent with its conservative behavior."**
  - STALE: the posterior test is excluded from the note (`:205`); "no calls" reflects **low power**, not better calibration (LeafletFA's own FDR was 3-4× anti-conservative); and the LeafCutter control (a7945e3) shows **neither method passes 10.2**.
  - STYLE: "calls". This sentence is in Karin's wording (2026-11-01), which predates check 10; Karin decides.
- `:338-339` "Distinguishing functional splicing variation from unproductive splicing or technical artifacts remains difficult, as many detected events may represent splicing noise rather than regulated programs [Fair 2024]."
  - STYLE: generic "remains difficult". "rather than regulated programs" ties to programs.
  - CITE: Fair 2024 is about human unproductive splicing; fine as a pointer.
- `:342` "Our analysis does not capture intron retention events; our current approach focuses on alternative SS and exon skipping events."
  - STYLE: "Our analysis ... our current approach" repeats itself.
  - NUM: the catalog also holds MXE and complex events; and junction-based ATSEs exclude singletons (stated only in the shared text). Say both.
- `:346-347` "Applying LeafletFA to emerging technologies like single-cell long-read sequencing [Shi 2023] would enable analysis of full-length isoforms and coordination ..."
  - STYLE: "emerging technologies like"; speculative future work; cut or shorten.
- **`:350-352` "In conclusion, LeafletFA provides a foundation for incorporating AS into single-cell atlases. As single-cell technologies continue to evolve, integrating splicing analysis will be essential for a complete understanding of cellular heterogeneity in health and disease."**
  - STYLE: "In conclusion", "foundation", "As X continue to evolve", "essential", "complete understanding", "in health and disease": a stated takeaway and manufactured closure. Genome Biology framing.
  - Action: end on the limits and the deliverable (catalog with intervals where validated; the browser), not a conclusion.
- `:354` red: the SpliceVI companion sentence. MISSING.
- MISSING in the Discussion: the usage sentence (atlas lookup with intervals vs own fits without intervals), and the per-cell split as the main limit of 10.1 and check 12.

### Availability, data, back matter (`:356-381`)
- `:358-361` "... all scripts required to reproduce the analyses and figures presented in this manuscript are hosted at Leaflet-analysis."
  - STALE: the analyses for this note are in the validation branch and in `Leaflet-analysis-bioinformatics`; confirm the repository and branch (`:362`, red). "all scripts required" is a strong claim.
- `:366-367` "Raw Smart-seq2 data ... Tabula Muris Senis and ... Allen Institute (GEO GSE185862)."
  - NUM: Allen is Smart-seq v4.
- **`:367-369` "The processed AnnData objects for mouse, including quantified junction counts and inferred SP activities (Φ), have been deposited on Zenodo: [DOI]."**
  - STALE: the proposed deposit is catalog TSVs with the check 12 interval, interval arrays, the event catalog, the browser site and seed-0 checkpoints (`RELEASE_MANIFEST.md` §3). Φ is not shipped. The model-input h5mu (138 GB) is Karin's Monday decision.
- `:374` Acknowledgements: fine. `:378-381` Funding: confirm (red). `:385` Conflict: fine.

## 4. Stale claims, collected

| Where | Claim | Why stale |
|---|---|---|
| `:112-114` | decomposes into latent programs; imputes per-cell usage | programs not reproducible (checks 3, 8); deliverable is cell-type usage |
| `:114` | harmonized, denoised single-cell atlas | cell-type level; raw is closer in types ≥ 1,000 cells |
| `:101-102` | no software works without cell-type labels | the note's estimator uses labels |
| `:191-193` | check 10 posterior interval; amendment 7 draft | replaced by check 12; amendment approved and run |
| `:236`, `:279` | 10.3 interval diagnostics | superseded by check 12 |
| `:283-286` | Fgfr2 exemplifies recovery; leverages information to resolve decisions | panel d: mesenchymal compressed to about 0.4 |
| `:293` | Nrxn SS4 | panel e shows no SS4 difference |
| `:303` | benchmarks not run | both ran |
| `:334-335` | posterior test conservative; LeafCutter anti-conservative | superseded by DM LRT scope and LeafCutter control; no calls in the note |
| `:350-352` | foundation; essential | framing |
| `:368` | deposit holds Φ | deposit contents changed |
| `:307-308` | figure and caption a-e | built figure is a-f or a-e |

- **Aging:** no aging claim remains in the prose. The ages "3, 18 and 24 months" and "stratified by broad cell type and age" are data descriptors. The comments at `:51`, `:90`, `:317`, `:387` record what was dropped.
- **Genome Biology framing remaining:** `:110` ("how much AS varies across single cells"), `:114`, `:318`, `:350-352`, and the Results / Discussion layout (section 2).
- **"Calls":** prose uses it at `:334` (and the title `:234`); the red 10.4 sentences (`:290`, `:293`) rest on call counts. Decision pending (OUTLINE_PREP section 4).

## 5. Citations older than 2015

| Key | Year | Where | Finding |
|---|---|---|---|
| Warzecha2009-lu (ESRP1 and ESRP2, Mol Cell) | 2009 | `:284` | the only pre-2015 citation in the main text. Human cell lines; mouse in-vivo evidence is Bebee 2015 (eLife 4:e08954), not cited |
| Pan2008, Wang2008 | 2008 | `references.bib` only | not cited in `main.tex`, the shared text or the supplement; unused entries |

- Gallery list (Supplement figure SG, when its caption cites them): Hermiston 2003, Sommer 1990, Boutz 2007, Zheng 2012, Noguchi 1986, Warzecha 2009.
  - Of these, Sommer 1990 and Noguchi 1986 are rat and abstract-only; Hermiston 2003 is paywalled (`GALLERY_KNOWN_BIOLOGY_DRAFT.md`).
  - Only Nguyen 2016 and Bebee 2015 (2015, not older) state a mouse in-vivo direction.
- Other citations are 2017 or later (Baralle 2017, Song 2017, Leafcutter 2018, Pyro 2018, Lukacsovich 2019, Tabula Muris 2020, Booeshaghi 2021, Allen 2021, Benegas 2022, SpliZ 2022, Psix 2022, Tabula Sapiens 2022, Shi 2023, regtools 2023, Fair 2024, AnnData 2024).
- CITE, separate: `references.bib` has `scquint` and `Benegas2022-zc` as two entries for the **same paper** (same title, journal, year). The Introduction cites it as Benegas; the tool is scQuint. Merge to one key before `:105` is written.

## 6. Missing points

| Point | Where it belongs | Facts on file |
|---|---|---|
| **Check 12 interval** (definition, construction, coverage, reach) | Methods `:191-193`; Limits; abstract | section 3 above; `OUTLINE_PREP.md` section 3 |
| **Atlas-only interval** (own fits get ψ, no interval) | Methods; Discussion; tutorial | `leafletfa.catalog.celltype_psi`; `TUTORIALS_PLAN.md` |
| **Browser** (URL, pages, columns, version, no call columns) | own short subsection; abstract Availability | preview-2026-11-10; URL not published |
| 10.1 result with the large-type reversal | Accuracy | panel b table |
| Smoothing at uncovered cells (about 0.12 at top-ranked events) | Methods and Limits | check 10 browser diagnosis |
| Check 13 (pooled estimate fails; current estimate kept) | Limits or Supplement | +0.25 overshoot (TMS) |
| LeafCutter and scQuint as existing tools; their benchmark | Introduction `:105`; Scale `:303` | descriptive control; jobs 22725257, 22780123 |
| Gria2 | Known biology | panel c facts |
| Per-cell split as the main limit | Discussion | `shared_data_catalog.tex` |
| ATSEmapper inputs, outputs, run time | ATSEmapper `:142` | — |
| Annotation name and version (GENCODE vM19) | ATSEmapper | caption facts |
| Zenodo DOI and deposit contents | Data availability | `RELEASE_MANIFEST.md` §3 |

## 7. Decisions this screen needs from Karin

- Section layout for the Application Note (Results / Discussion or not): verify against the OUP guide, then fix the budgets.
- Whether the Introduction keeps programs at all (`:112-114`, Discussion `:324-331`), or the note is catalog-only.
- "Calls": use the word nowhere, or only in the Limits section.
- Panel e: variant A or B (`OUTLINE_PREP.md` section 7); `:293` and the abstract depend on it.
- The three Scale numbers (`:299-302`) to be checked against the timing records before any prose is kept.
