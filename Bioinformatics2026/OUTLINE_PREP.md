# Outline session prep, Wed 11 Nov 2026 (no prose; refreshed 2026-11-11)

Source of line numbers: `~/LeafletFA_overleaf/main.tex` at 78436ab (394 lines; unchanged since the first version, no manuscript edits made). Word counts are approximate: prose only, with comments, red `\unsure{}` text, equations and citations removed (counted again 2026-11-11; the sentence-level screen is `LEAFLETFA_SCREEN.md`).

## 0. What changed since the first version of this file
- **Figure AN built** (32c3206, 2026-11-10): panels a–f, 183 × 164 mm (limit 165); gray no-interval variant in `figAN_grey_noiv/`.
  - Junctions chosen by annotation (GENCODE vM19, `paper/scripts/event_annotation.py`), no usage value read.
  - Lineages from one map (`cell_type_maps/mouse_lineage_map.json`), shared with the browser.
  - Panel a model schematic is still a placeholder.
- **Gallery built** (`figSG`, 183 × 220 mm): all 10 rows, whatever they give.
  - *Ctnnd1* (row 7) and *Pkm* (row 10) have no ATSE in the catalog; the panel says so.
  - *Nrxn1* is already row 3 (SG_c).
- **Gria2 flip/flop by annotation and sequence:**
  - flip = exon of Gria2-202 (ENSMUST00000107745.7); flop = exon of Gria2-201 (ENSMUST00000075316.9); translated and compared with UniProt P19491.
  - J1 `chr3_80689351_80690403_-` = into flip; J2 `chr3_80689351_80691319_-` = into flop.
  - Flop is 5′ of flip on this minus-strand annotation.
- **Panel e (*Nrxn1* SS4) shows no SS4 difference:**
  - J1 is the SS4-skipping junction (`chr17_90704367_91087954_-`, three vM19 transcripts without SS4).
  - It carries about **1–2 %** of ATSE reads in every neuronal group; medians LeafletFA / raw: excitatory 0.013 / 0.011; PV 0.019 / 0.018; SST 0.019 / 0.022.
  - The expected direction (Nguyen 2016: skipped more in excitatory, less in PV and SST) is **not seen**.
  - The junction also skips two small exons (18 nt and 12 nt) that flank SS4 in the seven transcripts that include it.
  - Only 2 PV and 3 SST groups pass the support rule.
  - This is consistent with the ATSE-level rule (`main.tex:293`): the 10.4 "pairs differ" result says the ATSE differs, not that SS4 drives it. It is **not** a contradiction, but the figure cannot be captioned as an SS4 switch.
  - Two variants built for Karin's decision (section 7).
- **Check 12 export done** (f879a6d, b65f8ec):
  - Intervals shipped in the catalog export for validated rows only, with `interval_status`, TEST reads and cells, `raw_psi`, `junction_reads`, mice.
  - `leafletfa/catalog.py` estimator port: unit test, and full-size check (max difference 5.9e-8 vs check 10, both atlases).
  - Zenodo catalog TSVs carry no 10.2 call columns (`tv`, `specific`, `call_eligible`); interval arrays unchanged.
  - Users' own fits get ψ but no interval (stated in the tutorial plan and in `catalog.py`).
- **Browser:** version preview-2026-11-10 (Lineage group-by, 46 headless checks pass). The 597 MB size in `RELEASE_MANIFEST.md` is for the 2026-11-07 build; not re-measured for 11-10. URL still unpublished.

## 1. Application Note limits

| Item | Value | Source |
|---|---|---|
| Length | 4 pages, about 2,600 words, or **about 2,000 words plus one figure** | Karin (`BIOINFORMATICS_PLAN.md:26`, `APPLICATION_NOTE_FIGURE_PLAN.md:3-5`); third-party guides agree, official OUP page not read: VERIFY |
| Figures | one main figure (183 × 165 mm layout); everything else in the Supplement or the browser | `APPLICATION_NOTE_FIGURE_PLAN.md` |
| Abstract | structured, budget 150 words (`main.tex:68`) | template headings are Motivation / Results / Availability and implementation / Contact / Supplementary information (`:70-74`). One guide gives "Summary" instead of Motivation + Results for Application Notes: VERIFY against the OUP author guidelines |
| Supplementary | allowed; refer to it in the abstract (`:74` does) | OUP guidelines (via search snippet): VERIFY |

## 2. Word budget vs current text

| Section | Lines | Budget | Current prose | Status |
|---|---|---|---|---|
| Abstract | 68-74 | 150 | 15 (Supplementary line only) | all red |
| Introduction | 79-115 | 200 | 256 | **over by about 56**; framing red |
| Data and event catalog | 120-122 (`shared_data_catalog.tex`) | 250 | 221 by the file's own count (247 by the script, which counts numbers and macros) | ported, shared with SpliceVI |
| ATSEmapper | 124-142 | 150 | 156 | ported; inputs/outputs missing |
| LeafletFA model and fitting | 144-181 | 200 | 267 | **over by about 67** |
| Cell-type usage and intervals | 183-205 | 150 | 77 | interval sentences **stale** |
| The catalog | 210-219 | 120 | 42 | numbers to choose |
| Accuracy (10.1) | 221-232 | 180 | 11 | facts only |
| Limits | 234-279 | 170 | 31 | facts only |
| Known biology and the portal | 281-295 | 120 | 129 | Gria2 and Nrxn1 facts now exist, no prose |
| Scale | 297-303 | 60 | 77 | over by about 17; benchmark line stale |
| Discussion | 313-354 | 250 | 230 | framing red |
| **Total (abstract + body)** | | **2,000** | **about 1,520** | Implementation sections over, Results under |

Back matter (not in the budget): Availability `:356-362`, Data availability `:364-370`, Acknowledgements, Funding `:376-381`, Conflict of interest.

## 3. Per section: what exists, what is stale, what is missing

### Abstract (`:68-74`)
- **Exists:** the Supplementary line (`:74`).
- **Missing:**
  - Motivation and Results (`:70-71`; plan content listed in red).
  - Availability: repository URLs, Zenodo DOI, portal URL (`:72`).
  - Contact (`:73`).
- **Decide:** whether Results names the interval (check 12) and the browser.

### Introduction (`:79-115`)
- **Exists, ported:**
  - Slot 1: AS (`:83-86`) and full-length scRNA-seq (`:91-94`).
  - Slot 2: duplicate reads (`:97`).
  - Slot 3: existing tools, SpliZ and PsiX (`:101-104`).
  - Slot 4: what we built (`:110-114`).
- **Check against scope:**
  - `:113-114` "decomposes ... into latent splicing programs ... and imputes per-cell junction usage". Programs did not reproduce (checks 3 and 8), and the deliverable is the cell-type catalog.
  - "a harmonized, denoised single-cell splicing atlas" (`:114`).
- **Missing:**
  - LeafCutter and scQuint as existing tools (`:105`).
  - What this note provides (`:115`).
- **Cut:** about 55 words.

### Implementation: Data and event catalog (`:120-122`)
- **Exists:** `shared_data_catalog.tex`, about 221 words in the main text (detail moved to the Supplement, Karin 2026-11-06). It includes the per-cell split, the singleton limitation, and the Allen Cre-line comment.
- **Missing:** nothing flagged.

### Implementation: ATSEmapper (`:124-142`)
- **Exists:** junction extraction and motif check (`:127-130`), splice graph and ATSE definition (`:135-140`).
- **Missing:**
  - Inputs, outputs and run time (`:142`).
  - Confirm the ATSEmapper URL (`:128`).
- **Cut candidate:** `:131`.

### Implementation: LeafletFA model and fitting (`:144-181`)
- **Exists:**
  - Overview (`:148-152`) and data (`:156-157`).
  - Generative model (`:160-164`).
  - DM and concentration floor (`:167-171`).
  - Inference recipe (`:174-178`).
- **Overlap:** `:148-152` and `:174` both say "SVI in Pyro". This is one place to cut toward 200.
- **Missing:**
  - Entry point (`--preset paper`) (`:180`).
  - E-step slot (`:181`, cut candidate).

### Implementation: Cell-type junction usage and intervals (`:183-205`)
- **Exists:** the estimator and comparator (`:185-188`), still correct (check 10).
- **Stale:**
  - `:191-192`: the check 10 posterior interval, no longer shipped. Karin rewrites these lines.
  - `:193`: "amendment 7 (draft, not approved)". It was approved and has run.
- **Check 12 facts** (comments `:194-203`):
  - **Definition:** a 95 % predictive interval for the cell type's held-out pseudobulk ψ at its read depth. It is not an interval for a true cell-type ψ.
  - **Construction:** m ± 1.96 √(v + τ²_b), clipped to [0, 1].
    - m and v come from 200 DM simulations of the held-out pseudobulk.
    - τ²_b is the model-error variance per atlas and TEST read bin, set on one mouse half and identical in both directions.
  - **τ² values:**

    | Atlas | 100–499 TEST reads | ≥ 500 TEST reads |
    |---|---|---|
    | TMS | 0.01024 | 0.00256 |
    | Allen | 0.00128 | 0.00032 |

  - **Where it exists:** where held-out cells have ≥ 100 reads and ≥ 10 cells with reads at the ATSE. That is **28 % of TMS rows** (1,829,935 of 6,478,290) and **35 % of Allen rows** (1,530,439 of 4,420,578). Other rows show ψ with no interval and an `interval_status` reason.
  - **Atlas only:** users' own fits get ψ (`leafletfa.catalog.celltype_psi`) but no interval, because the calibration needs held-out cells.
  - **Rest ψ:** has no interval.
- **Missing:** the rewritten interval sentences (lines 191-193 are unchanged and still describe the check 10 interval). A "users' own fits" sentence is also needed here and in the tutorial (Methods and tutorial both, per Karin).
- **Export is done** (f879a6d, b65f8ec): the shipped catalog carries the check 12 interval for validated rows, with `interval_status`; `leafletfa/catalog.py` reproduces check 10 to 5.9e-8. The Methods sentence can now point at a shipped file, not a plan.

### Results: The catalog (`:210-219`)
- **Exists:** `:215-218`: junctions, ATSEs and genes via ledger macros (79,768 junctions, 24,475 ATSEs, 11,233 genes, 132,706 cells).
- **Missing:** which catalog numbers lead (`:219`). Ledger options:
  - per-dataset size: TMS 78,135 cells / 132 types; Allen 54,571 cells / 68 groups;
  - event-type shares: complex 7,056; alt3 5,810; skipped exon 5,708; alt5 4,867; linked MXE 1,034;
  - reads per event and ATSEs per cell.

### Results: Accuracy against held-out pseudobulk, 10.1 (`:221-232`)
- **Exists:** one sentence (`:223`, "new cells of the same mice").
- **Facts** (`:226-230`). 10.1 passed in both atlases. Error = TV vs TEST pseudobulk; diff = LeafletFA − raw, with 95 % CI.

  | | TMS | Allen |
  |---|---|---|
  | Overall | **−0.0072** (−0.0094, −0.0050) | **−0.0021** (−0.0039, −0.0005) |
  | Smallest types (10–49 TRAIN cells) | −0.0194 | −0.0116 |
  | 1–19 TRAIN reads | −0.0625 | −0.0543 |

  - The gain is largest where data are thin.
  - Split: per cell, not per mouse.
- **Missing:** prose. Panel b is not drawn.
- **Cut candidate:** held-out fit (`:232`).

### Results: Limits of calls and intervals (`:234-279`)
- **Facts present** (comments):
  - 10.2 and 10.7 failed: TMS 0.813 vs null 0.422; Allen 0.837 vs 0.771; pairwise 0.780 vs 0.473.
  - Allen Cre-line donor confounding.
  - LeafCutter control: neither method passes.
  - Smoothing: top-ranked differences shrunk by about 0.12 vs held-out cells.
  - Check 13: the pooled estimate fails (+0.25 overshoot); the current estimate is kept.
  - Check 12 coverage table (`:256-265`).
- **Check 12 coverage at the pass rule:**
  - (a)+(b) passes at **0.950–0.963**.
  - (a) alone fails at 0.82–0.90.
  - The check 10 interval was 0.136–0.346.
  - TMS intervals are wide: median 0.225 at 100–499 TEST reads.
- **Stale:**
  - `:236` names 10.3 as the interval result; check 12 now replaces it.
  - `:279` (red): the check 10 interval diagnostics ("parameter-only widths below the median error; the mouse bootstrap adds little") describe the superseded interval. The γ floor fact stays true.
- **Missing:** all prose (`:236`).
- **Decide:** how much of calls / smoothing / check 13 goes here vs the Supplement. At 170 words it cannot hold all of them.

### Results: Known biology and the portal (`:281-295`)
- **Exists:** *Fgfr2* framing (`:283-284`); red sentences `:286` (general claim), `:290` (*Ptprc*), `:293` (*Nrxn* limits).
- **The four gallery genes** (main figure panels c–f; `APPLICATION_NOTE_FIGURE_PLAN.md`):

  | Gene | Event | Atlas | Fact on file | Status in main.tex |
  |---|---|---|---|---|
  | *Gria2* | flip/flop, linked MXE, `atse_3`, 7 junctions | Allen | no 10.4 row; descriptive only. Flip and flop identified by annotation and translation (section 0). Medians LeafletFA / raw: flip excitatory 0.154 / 0.155, inhibitory 0.067 / 0.104; flop excitatory 0.338 / 0.339, inhibitory 0.468 / 0.431. Citation Sommer 1990 (rat, abstract only) marked VERIFY | **missing** |
  | *Fgfr2* | IIIb/IIIc, `atse_3` | TMS | 10.4 PASS: epithelial min 0.704 > mesenchymal max 0.649; covered mesenchymal cells 0.05–0.15 (11 of 12 types); uncovered 0.35–0.63 (prior = posterior) | framing only; result red (`:287`) |
  | *Nrxn1* | SS4, `atse_2` | Allen | 10.4: 270 of 1,891 neuronal pairs differ (226 with the read/cell condition), ATSE-level. **Panel e: the SS4-skipping junction is 1–2 % of reads in every neuronal group; no SS4 difference** | red (`:293`); needs the panel e decision (section 7) |
  | *Ptprc* | lymphoid lead junction, `atse_2` | TMS | 10.4 PASS: 27 B/T calls, 25 confirmed in TEST; B cells lower than rest, T cells mostly higher | red (`:290`) |

  - **Note:** the 10.4 *Fgfr2* "IIIb specific in 10 epithelial types" and the *Ptprc* / *Nrxn* counts use the call machinery that failed 10.2 / 10.7. The word "calls" is tied to that decision (`:289`).
- **Gallery citations:** volumes and pages verified against PubMed (6970a68); direction-in-mouse check in section 6.
- **Browser facts:**
  - Version preview-2026-11-10 (Lineage group-by; 46 headless checks); the 597 MB figure is from the 11-07 build.
  - Pages: gene, event (structure + per-type panel), compare two, one vs. all.
  - Beside every LeafletFA estimate: raw TRAIN pseudobulk ψ, reads, cells and mice with reads, and held-out TEST reads and cells.
  - The check 12 interval appears only where validated.
  - Downloads carry fit and interval md5.
  - No call columns are shown.
  - **Not published:** the URL is a gap (`:295`).
- **Missing:**
  - *Gria2* text.
  - Results for the other three genes.
  - Portal description and URL.

### Results: Scale (`:297-303`)
- **Exists:** GPU and CPU timings, atlas-size run (`:299-302`).
- **Stale:** `:303` says the scQuint and LeafCutter catalog benchmarks are "not run yet". Both have run: scQuint job 22725257; LeafCutter TMS_full array 22780123. Results not pulled into the text.
- **Cut:** about 17 words.

### Figure (`:305-310`)
- **Stale:** the caption (`:308`) lists panels a–e (model, representation checks), and `:307` includes the old `Figures/fig1_catalog.pdf`. The built layout (figAN) is a–f:
  - a: catalog and model;
  - b: accuracy (10.1);
  - c: *Gria2*;
  - d: *Fgfr2*;
  - e: *Nrxn1* SS4;
  - f: *Ptprc*.
- **Drawn** (figAN, built 32c3206, rebuilt as variants 2026-11-11); **missing:** the caption and the panel a model schematic. Panels c–f use the check 12 interval wording.
- **Panel e is undecided** (section 7). Panel c is drawn from data in the figure theme (teal for LeafletFA), so the browser screenshots in section 5 are no longer needed as a panel.
- **Caption limits to state** (`FIG_APPNOTE_CAPTION_FACTS.md`): per-cell split, interval coverage 28 % / 35 % of rows, seed 0 only, no programs and no specificity calls shown, and that in types with ≥ 1,000 TRAIN cells raw is slightly closer to held-out cells than LeafletFA (+0.0054 TMS, +0.0057 Allen; panel b).

### Discussion (`:313-354`)
- **Exists:**
  - Opener (`:318`), duplicate reads (`:321`), programs (`:324-327`), depth (`:330-331`).
  - DS test, in Karin's wording (`:334-335`).
  - Noise caveat (`:338-339`), intron retention (`:342`), long reads (`:346-347`), conclusion (`:350-352`).
- **Missing:**
  - Lead (`:314`).
  - SpliceVI companion sentence (`:354`).
  - Possibly a usage sentence: atlas lookup with intervals vs own fits without intervals.

### Availability and Data availability (`:356-370`)
- **Missing:**
  - Versions and tags (29 Nov) and the portal URL (`:362`).
  - The Zenodo DOI (`:369`).
- **Stale:** `:368` says the deposit holds "inferred SP activities (Φ)". The proposed deposit is in `LeafletFA/RELEASE_MANIFEST.md` §3:
  - catalog TSVs with the check 12 interval and raw ψ / reads / mice;
  - interval arrays;
  - the event catalog;
  - the browser site;
  - seed-0 checkpoints.
  - The model-input h5mu (138 GB) is Karin's Monday decision.

### Funding (`:376-381`)
- **Exists:** the NSF line from the old manuscript.
- **Missing:** confirmation (`:381`).

## 4. Decisions for the session
- Abstract headings (Motivation / Results vs Summary): verify against OUP.
- Where the cuts come from:
  - Implementation is about 120 words over (Introduction, model).
  - Results has about 400 words free.
- Limits section: which of calls, smoothing, check 13, check 12 go in the 170 words vs the Supplement.
- Whether "calls" appears anywhere (10.4 rows use the call machinery).
- *Gria2*: one sentence, with or without the citation (VERIFY).
- ~~Panel e~~ **decided (Karin 2026-11-11): variant C is panel e** (SS4, SpliceVI §111). Gallery row 3 redrawn on SS4; 10.4 Nrxn rows corrected by amendment 6 note 6 (old rows relabeled as an early Nrxn site, new SS4 rows added); `main.tex:293` marked STALE.
- *Gria2* browser screenshot: now only a Supplement or portal illustration, if at all (panel c is drawn from data).

## 5. *Gria2* browser screenshots (candidates; `portal/make_pub_screenshots.py`, sbatch job 23120924)

Files in `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser/screenshots/2026-11-09_pub/`. Light theme. For the crop, the junction table, header and controls are hidden; nothing else changed.

| File | Shows | Pixels | At 183 mm wide | Smallest text at 183 mm |
|---|---|---|---|---|
| `gria2_allen_structure_panel.png` | structure + legend + all 66 cell groups by class (excitatory 47, inhibitory 15, non-neuronal 4) | 5,648 × 7,192 | 784 dpi, **233 mm tall** | **3.3 pt** (tick labels); body 4.4 pt |
| `gria2_allen_structure_panel_tooltip.png` | structure + legend + first 22 rows + tooltip (CA1, J3: ψ 0.00 [0.00, 0.04], 5,090 held-out reads, 183 held-out cells) | 5,648 × 3,508 | 784 dpi, 114 mm tall | **3.3 pt**; body 4.4 pt |

- **Both fail the 5 pt floor at double-column width:** the browser's text is 9–12 px on a page about 1,400 px wide. The full panel is also taller than the 220 mm figure art limit.
- **Options:**
  - Use them only as browser illustrations, in the Supplement or the portal page, at a size where 5 pt is not required.
  - Crop to fewer cell groups and junctions.
  - Draw panel c from the data in the figure theme, as the figure plan already says.
  - No font change was made to the browser.
  - **Color:** the browser draws LeafletFA in ochre. The figure plan and theme reserve teal for LeafletFA, so a screenshot used as a figure panel would break the method-color rule.

## 6. Gallery status (Karin 2026-11-09; built 2026-11-10)
- The list is final with Karin's changes (`GALLERY_KNOWN_BIOLOGY_DRAFT.md`):
  - *Ptprc*: B vs T, with the CD45 review.
  - *Nrxn1*: "differs", with Nguyen 2016 added.
  - *Gria2*: descriptive.
  - *Pkm*: Noguchi 1986.
  - Row 11 (*Src* N1): dropped (Karin 2026-11-10; the catalog exon is 18 nt, Irimia 2014 defines microexons as 3–15 nt).
- No alt 3′ or alt 5′ event; the caption must say so.
- **Built** (`figSG`, 10 rows, 8 drawn; *Ctnnd1* and *Pkm* have no catalog ATSE). Values were read only after the list was final.
- Readings so far (facts, no verdict): *Cd44* and *Enah* in the expected direction in raw; *Fgfr2* mesenchymal types compressed toward 0.4; *Nrxn1* not seen at the skipping junction; *Ptbp2* and *Dlg4* rest on one TMS neuron type.
- Citation volume and pages verified against PubMed (all match). Direction-in-mouse check (10 Nov): only row 3 (Nguyen 2016) is stated in mouse cell types; rows 1, 5–7 are human cell lines; 8–9 mouse developmental; 4 rat; 2 and 10 paywalled. Proposals in `GALLERY_KNOWN_BIOLOGY_DRAFT.md`.

## 7. Panel e: three variants for Karin's decision (A and B built 2026-11-11; C added 2026-11-11 after Karin's SS4 definition)
- **A, panel e kept as is** (`figAN_varA/`): the figure is unchanged; the caption states the fact.
  - Fact for the caption: the junction shown is the SS4-skipping junction; it carries 1–2 % of the ATSE's reads in every neuronal group, and the expected direction (skipped more in excitatory, less in PV and SST) is not seen; the junction also skips two small exons.
  - Cost: a main-figure panel with no visible result, in a note that says it shows known biology.
- **B, five panels without *Nrxn1*** (`figAN5_varB/`): a, b, c *Gria2*, d *Fgfr2*, e *Ptprc* (was f); 183 × 164 mm (the same height as figure AN).
  - Row 3 holds *Ptprc* alone at the left; the right half is empty.
  - *Nrxn1* stays in the supplementary gallery as row 3 (SG_c), unchanged; nothing needs moving.
  - Legend of panel a reads "Panels c–e".
  - Cost: the neuron example is gone from the main figure; `main.tex:293` and the abstract plan lose the *Nrxn* sentence or move it to the Supplement.
- Both variants use the same code and the same data as figAN (variant B adds `fig_appnote5.py` and an `AN5` entry in `paper_config.yaml`).
- **C, panel e = *Nrxn1* SS4 by the SpliceVI Fig 2 definition** (`figAN_varC/`): 90-nt exon chr17:90,163,792–881, `atse_7`; J1 inclusion `chr17_90162433_90163791_-` (scored), J2 skipping `chr17_90162433_90208322_-`; PV / SST / VIP / Lamp5 split.
  - **The expected direction is seen.** SS4 inclusion median LeafletFA / raw: excitatory 0.153 / 0.128 vs PV 0.367 / 0.371, SST 0.405 / 0.471. Skipping: excitatory 0.704 / 0.765 vs PV 0.285 / 0.270, SST 0.237 / 0.106.
  - 64 cell groups; 48 with an interval.
- **Variant A's panel e is not SS4.**
  - `atse_2` holds three small early exons of Nrxn1α (18 / 30 / 12 nt; exons 2–4 of the α transcripts; residues about 259–274 of Q9CS84, after EGF-like 1), consistent with SS#1 (Treutlein 2014: SS#1 = "Nrxn1α exons 3a and 4"; SS number not verified).
  - The check 10.4 Nrxn1–3 rows used the same 30-nt-exon rule (c10_known_biology_fix.SS4), so they scored this early site, not SS4.
  - The gallery (SG_c) and `main.tex:293` also describe that site as SS4.
