# Outline session prep (no prose; first version Wed 11 Nov 2026, refreshed 2026-11-13; fact skeleton added 2026-11-16)

## 00. Fact skeleton, in writing order (2026-11-16; fragments only, no prose)

Each section: budget (words, abstract + body total 2,000), then the facts it needs in the order they would be written,
each with its source. VR = `LeafletFA/scripts/validation/VALIDATION_REPORT.md` (LeafletFA repo, branch `leafletfa-validation`); PR = `VALIDATION_PREREG.md` (same
dir). Older detail stays in sections 0a–7 below; this skeleton supersedes their ordering, not their facts.

### Abstract — 150
- Motivation: sparse full-length scRNA-seq junction counts; per-cell-type junction usage. Source: Introduction slots (`main.tex:83-114`).
- Results, candidate facts (choose ≤ 3):
  - Accuracy 10.1: LeafletFA closer to held-out pseudobulk than raw, largest gain at 1–19 TRAIN reads (TMS −0.0625, Allen −0.0543). VR "Check 10".
  - Interval: check 12 (a)+(b) coverage 0.950–0.963 where validated (28 % TMS / 35 % Allen rows). VR "Check 12 — result".
  - DS between cell types (TMS pairs): check 18a BETTER in mice the fit never saw (VR "Check 18", 6fbc7e9); check 17 BETTER with the full-data fit (VR "Check 17", 26da281). Allen pairs with ≥ 4 shared donors: check 19 BETTER (VR "Check 19", 713883d).
  - Scale: TMS fit about 13 min, Allen about 38 min on one GPU. VR "Scaling and runtime record".
- Availability: repo URLs, Zenodo DOI, browser URL — all missing (browser: separate `daklab/leafletfa-browser` repo, not created; nothing public before the paper, Karin 2026-11-15).
- Contact: missing.

### Introduction — 200 (now 256, about 55 over)
- Slot order as ported: AS; full-length scRNA-seq; duplicate reads; existing tools (SpliZ, PsiX); what we built. `main.tex:83-114`.
- Add: LeafCutter and scQuint as existing tools (`:105`).
- Remove/reword: "latent splicing programs ... imputes" (`:113-114`): programs did not reproduce (checks 3, 8; VR "Check 3", "Check 8").
- Cut about 55 words (section 3 below lists candidates).

### Implementation: data and event catalog — 250 (now 221)
- Shared with SpliceVI word for word (`shared_data_catalog.tex`); per-cell 70/30 split; singleton limitation; Allen Cre-line comment.
- Counts: 79,768 junctions, 24,475 ATSEs, 11,233 genes, 132,706 cells (ledger macros, `main.tex:215-218`).
- Per dataset: TMS 78,135 cells / 132 types; Allen 54,571 cells (38,199 TRAIN) / 68 groups. `CATALOG_NUMBERS.md`; VR item 1 (ceb5048).
- Gap: the filter-chain table SpliceVI cites is empty (status PDF p. 2).

### Implementation: ATSEmapper — 150 (now 156)
- Junction extraction, motif check, splice graph, ATSE definition (`:127-140`).
- Missing: inputs, outputs, run time (`:142`); URL to confirm (`:128`). Run time: UNSOURCED (no ATSEmapper timing record found).
- Reconcile the ATSE definition with the shared catalog text (status PDF p. 2).

### Implementation: LeafletFA model and fitting — 200 (now 267, about 65 over)
- DM factor model, K = 20; concentration floor γ ≥ 2; SVI recipe: 16,200 steps, minibatch 8,192, lr 0.025 decaying to 0.0025, float64 φ. Run record of `TMS/full_K20_s0_f64` (LeafletFA_splicevi_baseline); `scaling/lfa_gpu_full/run_record.json` recipe block.
- Entry point (`--preset paper`, `:180`): VERIFY it exists in the release CLI.
- Cut the duplicated "SVI in Pyro" (`:148-152` vs `:174`).

### Implementation: cell-type usage and intervals — 150 (now 77, stale)
- Estimator: type mean of per-cell posterior-mean usage ρ̃ = (γρ + y)/(γ + N); comparator raw TRAIN pseudobulk. PR amendment 6.
- Interval shipped = check 12: 95 % predictive interval for held-out pseudobulk ψ at its read depth, m ± 1.96 √(v + τ²_b); τ² TMS 0.01024 / 0.00256, Allen 0.00128 / 0.00032 (100–499 / ≥ 500 TEST reads), set on one mouse half. VR "Check 12 — result"; `tsv/c12_*_tau2.tsv`.
- Where it exists: ≥ 100 TEST reads and ≥ 10 TEST cells: 28 % TMS rows (1,829,935 / 6,478,290), 35 % Allen (1,530,439 / 4,420,578). VR "Check 12 export".
- Users' own fits: ψ, no interval (`leafletfa/catalog.py`).
- Stale now: `:191-193` (check 10 interval; "amendment 7 draft").

### Results: the catalog — 120 (now 42)
- Lead numbers to choose: per-dataset size; event classes (complex 7,056; alt3 5,810; skipped exon 5,708; alt5 4,867; linked MXE 1,034). `portal/SOURCE.json`; `CATALOG_NUMBERS.md`.

### Results: accuracy against held-out pseudobulk (10.1) — 180 (now 11)
- Error = TV vs TEST pseudobulk; LeafletFA − raw (95 % CI over types). VR "Check 10" table 10.1.
- TMS overall −0.0072 (−0.0094, −0.0050); Allen −0.0021 (−0.0039, −0.0005).
- Smallest types (10–49 TRAIN cells): TMS −0.0194, Allen −0.0116. 1–19 TRAIN reads: TMS −0.0625, Allen −0.0543.
- ≥ 100 TRAIN reads: TMS LeafletFA 0.0890 vs raw 0.0937; Allen 0.0529 vs 0.0522 (tie). Check 13 table, VR "Check 13" (`c13_pooled/*_accuracy.tsv`).
- Types ≥ 1,000 TRAIN cells: raw slightly closer (+0.0054 TMS, +0.0057 Allen). `FIG_APPNOTE_CAPTION_FACTS.md`.
- Simple shrinkage comparator B (check 14): LeafletFA beats B in TMS, ties in Allen; 1–19 reads LeafletFA 0.152 / 0.146 vs B 0.168 / 0.155. VR "Check 14".
- Simulation (checks 15, 15b): recovery worse than raw and B at own truth (15b Δ raw +0.0097 / +0.0128 / +0.0125); differential AUROC 0.684 / 0.664 / 0.648; truth independent per type (least favorable). VR "Check 15", "Check 15b".
- Limit: per-cell split, TEST and TRAIN share mice.

### Results: differential splicing between cell types (NEW; budget not set — DECIDE, candidates: Limits 170 and the about 400 free Results words)
- Design (checks 16–17): calls from half-A mice' TRAIN cells; confirmation in half-B mice' TEST cells only; confirmed = held-out TV > 0.05 with the same sign at the lead junction; excess = confirmed − within-mouse permuted null; equal calls per contrast. PR amendments 12, 12 note 1, 13.
- Check 16 (type vs rest, 100 types, top 500 per bin; VR "Check 16", e205185): **FAIL**. Only miss: ≥ 500 reads vs raw, −0.028, lower bound −0.077 (margin −0.05). 1–19 reads supported (+0.157 vs raw). vs LeafCutter +0.34 / +0.27 (100–499 / ≥ 500). Uncertainty-weighted z = point ranking.
- Check 16b (VR "Check 16b", cb3b05d): design 1 (5 per type, same types) **BETTER**: vs raw +0.102 / +0.152 / +0.077 / +0.048, vs LeafCutter +0.323 / +0.311; design 2 (types where all methods call) **FAIL** (100–499 vs raw −0.081). Pre-registered reading: **"at least partly explained by concentration"**.
- Check 17 (pairs, 300 sampled, 5 per pair per bin; VR "Check 17", 26da281): **BETTER**.
  - LeafletFA − raw: +0.079 (1–19), +0.068 (20–99), +0.070 (100–499), +0.055 (≥ 500); all lower bounds > 0.
  - LeafletFA − LeafCutter (50-pair subset): +0.081 (lower bound +0.008, 41 pairs, 100–499), +0.061 (+0.004, 48 pairs, ≥ 500). Small margin.
  - LeafCutter testable: 0 pairs with ≥ 5 scorable units below 100 reads.
  - Same cell type across tissues: no gain over raw below 500 reads (CIs include 0); gain only at ≥ 500 (+0.091).
  - Rank bands, LeafletFA excess: ranks 1–5 0.27–0.29, 6–20 0.22–0.26, 21–100 0.13–0.15 (`$V/c17/rank_bands.tsv`).
- Check 18 (refit on half-A mice, 16,200 steps, 12.9 min GPU; confirmation only in half-B mice; PR amendment 14 + note 1; VR "Check 18", 6fbc7e9): **18a pairs BETTER**, **18b type vs rest BETTER**.
  - 18a LeafletFA − raw: +0.088 / +0.068 / +0.094 / +0.061 (1–19 / 20–99 / 100–499 / ≥ 500), all lower bounds > 0. vs LeafCutter: +0.156 (lower bound +0.093, 49 pairs, 100–499), +0.059 (−0.0002, 50 pairs, ≥ 500).
  - 18b LeafletFA − raw: +0.214 / +0.087 / +0.079 / +0.036 (≥ 500 lower bound −0.006).
  - The limit "the model saw the confirming mice" no longer applies to 18a / 18b. Same cell type across tissues: still no gain below 100 reads (+0.025, +0.027; CIs include 0).
  - Rank bands (18a), LeafletFA confirmed (excess): ranks 1–5 0.83–0.93 (0.27–0.39), 6–20 0.77–0.82 (0.23–0.27), 21–100 0.63–0.67 (0.16–0.17). `$V/c18/18a_rank_bands.tsv`.
  - Composite mouse labels (3_10_M/3_11_M, 3_38_F/3_39_F) joined with their mice in the check 18 split; the 10.2a halves (checks 10.2–17) had 106 of about 78,000 cells on the wrong side (PR amendment 14 note 1).
- Allen coverage: 1,094 of 2,278 pairs (48 %) have no donor contributing both groups (flag "cell type and donor not separable; differences may reflect donor or batch"); 642 have ≥ 4. `$V/c19/allen_pairs_index.tsv`.
- Check 19 (Allen pairs with ≥ 4 shared donors, within donors; PR amendment 15; VR "Check 19"): **BETTER**. LeafletFA − raw +0.029 / +0.091 / +0.060 / +0.064, all lower bounds > 0 (564 pairs). Same class at 1–19 reads +0.011 (CI includes 0). Limit: the Allen fit saw all TRAIN cells; confirming TEST cells come from the same donors.
- Allen sense checks (descriptive; VR "Allen sense checks", 9c15ab1). Claimable: *Nrxn1/2/3* SS4 expected direction in 97 / 100 / 99 % of excitatory × inhibitory pairs (median rank 16 / 94.5 / 152); Irimia 2014 microexons 48× enriched, 98 % neuronal direction; hierarchy 0.259 (neuron vs glia) → 0.079 → 0.024 → 0.004 (same subclass), cophenetic Spearman 0.81; skipped-exon frame 0.816 vs 0.478. **Not claimable:** *Grin1* exon 5 and *Gria2* flip/flop direction (UNSOURCED); frame for alt 5′ / 3′ (no enrichment).
- Earlier fails stay reported: 10.2, 10.7 (rule ≥ 0.5 and ≥ 5× null), 10.2 LeafCutter control (descriptive). VR "Check 10", "LeafCutter descriptive control".

### Results: limits of calls and intervals — 170 (now 31)
- Check 12 coverage (a)+(b) 0.950–0.963; (a) alone 0.82–0.90; check 10 interval 0.136–0.346. TMS median width 0.225 at 100–499.
- Shrinkage of top differences: top-100 error vs held-out TMS 0.224 vs raw 0.151 (browser-gap diagnosis; VR "Browser ranking vs raw Δψ"); check 13 pooled estimate fails (+0.253 overshoot). VR "Check 13".
- Allen Cre-line donor confounding (VR "Allen donor check").
- Stale: `:236`, `:279`.

### Results: known biology and the portal — 120 (now 129)
- *Gria2* flip/flop (Allen, descriptive): medians in section 3 table. Citation Sommer 1990 VERIFY. Expected excitatory vs inhibitory direction UNSOURCED (Allen sense check: rank 688): not claimable as a positive control.
- *Fgfr2* IIIb (TMS, 10.4 PASS): epithelial min 0.704 > mesenchymal max 0.649. Cite Warzecha 2009 and **Bebee2015** (now in `manuscript/references.bib`, PMID 26371508).
- *Nrxn1* SS4 (Allen, 10.4 note 6 PASS): 1,256 of 1,891 pairs; SS4 inclusion excitatory 0.153 vs PV 0.367, SST 0.405 (Nguyen 2016 direction).
- *Ptprc* (TMS, 10.4 PASS): 25 of 27 B/T calls confirmed in TEST.
- Portal: pages gene / event / compare / one-vs-rest; raw ψ, reads, cells, mice beside every estimate; check 12 interval only where validated; comparison notes now state checks 16, 16b, 17 (portal `web/config.js`); not published.

### Results: scale — 60 (now 77; numbers wrong in `:299-302`)
- Production fits, one RTX PRO 6000 Blackwell (b6k), 16,200 steps, batch 8,192: **fit 12.9 min TMS, 37.7 min Allen (seeds 1–2; Karin, 25 Nov; ledger `\FitMinutesTMS`, `\FitMinutesAllen`; source `SpliceVI-utils/manuscript/age_analysis/audit_2026_09/RUNTIME_TABLE.md` rows 26, 28; jobs 22402842 / 22402845, 22402848 / 22402851)**, 54,694 / 38,199 TRAIN cells, 0.05 / 0.14 s/step, host 9.2 / 21.3 GB. Seed 0 has no measured fit time: jobs 22400619 and 22400622 failed while saving (their log spans, 13.2 / 38.5 min, are not quoted). Check 18 refit: `\RefitHalfMinutes` 12.9 min, `\RefitHalfPeakGPU` 6.6 GB at `\RefitHalfCells` 46,479 cells. Validation numbers for checks 10.1, 12, 17, 18a, 18b, 19, the Allen sense checks and scaling are ledger macros (`paper/ledger/results_ledger.tsv`, 67 rows added 25 Nov), plus rank-band rates for the browser (`\RankBand<atlas><band><bin><Confirmed|Excess>`, 48 rows: TMS from 18a, Allen from 19).
- 10.6 / 31.6 min (old text) are projection times, not fit times.
- Peak GPU memory of the production fits: **not recorded**. METHODS 9.0 / 29.7 GB: no source, remove or replace (controlled runs: 2.5–7.0 GB GPU at 5,000–54,694 cells; atlas-size run 25.03 GB GPU).
- Controlled runs (2,000 steps): GPU 0.032–0.046 s/step, flat from 10,000 cells; CPU (16 cores) 3.27 s/step at 5,000 cells = **102×** slower. `$V/scaling/scaling.tsv` (0b355f1).
- scQuint (16 CPUs, to early stop): 13.3 / 37.3 / 72.3 / 67.0 min at 5k / 10k / 25k / 50k cells; 86.0 min at 54,694.
- Atlas-size fit: 132,706 cells, 73.2 min fit (0.271 s/step), 40.7 GB host.
- LeafCutter catalog benchmark: 19.2 task-h (77 core-h).

### Figure
- figAN a–f (183 × 164 mm): a catalog/model (schematic placeholder), b accuracy 10.1, c *Gria2*, d *Fgfr2*, e *Nrxn1* SS4, f *Ptprc*. Candidate panel b built for check 17 and for check 18a (`paper_figures/figAN/figAN_b_candidate_check1{7,8}.pdf`; c1a23ac).
- Caption limits: per-cell split; interval 28 % / 35 % of rows; seed 0; no programs; no significance flags.

### Discussion — 250 (now 230)
- Lead (`:314`) and SpliceVI companion sentence (`:354`) missing.
- Facts available for the DS paragraph: what held (thin data, pairs at equal calls, LeafCutter cannot test thin events) and what did not (16 FAIL, 16b concentration reading, same cell type across tissues, small LeafCutter margin, simulations). Sources as in the DS section above.

### Availability and data availability — outside budget
- Missing: versions/tags, browser URL, Zenodo DOI. Deposit contents: `LeafletFA/RELEASE_MANIFEST.md` §3. h5mu deposit: Karin's decision.
- Released fit: the 70 % training fit, matching what was validated (Karin 2026-11-15).

Source of line numbers: `~/LeafletFA_overleaf/main.tex` at 78436ab (394 lines). Since then: 556f14a (STALE comment at `:293`) and c17d35b (stale items wrapped in `\unsure`, facts in comments; no rewording), which shift later lines by up to about 40. **Line numbers below are the 78436ab numbers.** Word counts are approximate: prose only, with comments, red `\unsure{}` text, equations and citations removed (counted 2026-11-11, before the stale wraps; the sentence-level screen is `LEAFLETFA_SCREEN.md`).

## 0a. What changed since 11 Nov (refresh 2026-11-13)
- **Panel e decided (Karin 2026-11-11): variant C is figAN.** *Nrxn1* SS4 by the SpliceVI definition (STAGE5 §111):
  - 90-nt exon chr17:90,163,792–881, `atse_7`.
  - J1 SS4 inclusion `chr17_90162433_90163791_-`; J2 SS4 skipping `chr17_90162433_90208322_-`.
  - Panel titles "J1: SS4 inclusion" / "J2: SS4 skipping"; PV / SST / VIP / Lamp5 split; 64 cell groups, 48 with an interval.
  - **Expected direction seen.** Medians LeafletFA / raw (`figAN_e_Nrxn1.tsv`):

    | Group (n) | SS4 inclusion | SS4 skipping |
    |---|---|---|
    | Excitatory (47) | **0.153** / 0.128 | 0.704 / 0.765 |
    | PV (2) | **0.367** / 0.371 | 0.285 / 0.270 |
    | SST (3) | **0.405** / 0.471 | 0.237 / 0.106 |
    | VIP (3) | **0.432** / 0.479 | 0.193 / 0.086 |
    | Lamp5 (2) | 0.370 / 0.388 | 0.302 / 0.252 |
    | Other inhibitory (5) | 0.330 / 0.370 | 0.320 / 0.212 |
    | Non-neuronal (2) | 0.483 / 0.459 | 0.077 / 0.107 |

- **Check 10.4 *Nrxn* rows corrected** (amendment 6 note 6, committed 6d7c648 before computing; result 685df1b, job 23230218). Rule unchanged (point TV > 0.10 and non-overlapping check 10 intervals at ≥ 1 junction; 1,891 neuronal pairs):

  | Gene | SS4 ATSE | Pairs differ | With read/cell condition | Verdict |
  |---|---|---|---|---|
  | *Nrxn1* | `atse_7` | **1,256** | 1,256 | PASS |
  | *Nrxn2* | `atse_6` | **946** | 581 | PASS |
  | *Nrxn3* | `atse_7` | **1,129** | 812 | PASS |

  - Each SS4 ATSE holds both the inclusion and the skipping junction (the old *Nrxn3* "no spanning junction" limit applied to the early site only).
  - Descriptive SS4 inclusion medians, excitatory vs inhibitory: *Nrxn1* 0.153 vs 0.396; *Nrxn2* 0.185 vs 0.334; *Nrxn3* 0.361 vs 0.525.
  - Old rows kept, relabeled "early Nrxn site (not SS4)", verdicts unchanged: `atse_2`, three small early exons of Nrxn1α (18 / 30 / 12 nt), consistent with SS#1 (SS number not verified).
  - The ATSE-level caveat still holds: a differing pair says the ATSE differs, not that the SS4 junctions drive it.
- **Gallery row c rebuilt on SS4** (SG_c, same source data as figAN e, md5 ef77be6b980f9873333a221a21d261a1).
- **main.tex stale items wrapped** (c17d35b): `:113-114`, `:191-193`, `:236`, `:279`, `:283-293`, `:303`, `:307-308`, `:368` and Smart-seq2 at `:111`, `:216`, `:366`. Each has the current fact and source in a comment above; no wording changed.
- **Benchmarks pulled** into the Scale facts (section 3).

## 0. What changed between the first version and 11 Nov
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
- ~~Panel e (*Nrxn1* `atse_2`) shows no SS4 difference~~ **Superseded (section 0a):** `atse_2` is not SS4 but an early Nrxn1α site; its junction `chr17_90704367_91087954_-` carried 1–2 % of ATSE reads. Panel e is now SS4 `atse_7` (variant C).
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
  | *Nrxn1* | SS4 (SpliceVI §111), `atse_7` | Allen | 10.4 (note 6): **1,256 of 1,891** neuronal pairs differ (1,256 with the read/cell condition), PASS, ATSE-level; *Nrxn2* `atse_6` 946 (581), *Nrxn3* `atse_7` 1,129 (812), both PASS. Panel e SS4 inclusion: excitatory **0.153** vs PV 0.367, SST 0.405, VIP 0.432 (LeafletFA medians; table in section 0a) | facts as comments; old *Nrxn3* limit sentence wrapped (c17d35b) |
  | *Ptprc* | lymphoid lead junction, `atse_2` | TMS | 10.4 PASS: 27 B/T calls, 25 confirmed in TEST; B cells lower than rest, T cells mostly higher | red (`:290`); "calls" flagged stale (c17d35b) |

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
- **Stale:** `:303` says the scQuint and LeafCutter catalog benchmarks are "not run yet" (wrapped, c17d35b). Both have run. Descriptive cost, no pass rule; different hardware:

  | Tool | Job | Data | Hardware | Time | Peak memory |
  |---|---|---|---|---|---|
  | LeafletFA, TMS seeds 1–2 | 22402842, 22402845 | 54,694 TRAIN cells | GPU (RTX PRO 6000) | fit **12.9 min** (`\FitMinutesTMS`; RUNTIME_TABLE) | — |
  | LeafletFA, Allen seeds 1–2 | 22402848, 22402851 | 38,199 TRAIN cells | GPU | fit **37.7 min** (`\FitMinutesAllen`; RUNTIME_TABLE) | — |
  | LeafletFA, atlas size | 22597359 | 132,706 cells | GPU (b6k) | fit **73.2 min** (0.27 s/step); whole job 119.4 min | 40.7 GB host |
  | scQuint | 22725257 | TMS TRAIN, 54,694 cells; 51,119 introns, 21,742 groups | 16 CPUs | training **86.0 min** (early stop); job 1 h 27 min | 1.7 GB |
  | LeafCutter cell type × event catalog | 22780123 | TMS, 128 tasks | 4 CPUs per task | **19.2 task-h** (77.0 core-h allocated); longest task 48 min | 1.28 GB per task |

  - Sources: sacct; logs `lfa_c10_scquint_22725257.log`, `lfa_c10_atlasfull_c1_22597359.log`; `run_record.json`; `RELEASE_MANIFEST.md` fits table.
- **Numbers to fix in `:299-302` (not wrapped; not on the stale list):**
  - "the full 16,200-step fit took 119 min": the fit took 73.2 min; 119.4 min is the whole job.
  - "15–40 min per fit" vs the 12.9 / 37.7 min production fits (seeds 1–2; 10.6 / 31.6 min are the seed-0 projection jobs).
  - CPU "about 10× slower (6 s/step)" vs 0.05 s/step GPU is 120×: VERIFY ([M] section 4).
- **Cut:** about 17 words.

### Figure (`:305-310`)
- **Stale:** the caption (`:308`) lists panels a–e (model, representation checks), and `:307` includes the old `Figures/fig1_catalog.pdf`. The built layout (figAN) is a–f:
  - a: catalog and model;
  - b: accuracy (10.1);
  - c: *Gria2*;
  - d: *Fgfr2*;
  - e: *Nrxn1* SS4 (`atse_7`, J1 SS4 inclusion, J2 SS4 skipping; PV / SST / VIP / Lamp5 split);
  - f: *Ptprc* (J1 = exon 7→8 of Ptprc-207).
- **Current figAN** (`paper_figures/figAN/figAN.pdf`, built 2026-11-11 22:08): variant C. 183 × 164 mm (limit 165).
  - Panel a legend: "Exons: GENCODE vM19 (GRCm38)". Panel b: neutral markers, axis "Held-out error, LeafletFA − raw". Intervals 0.4 pt, alpha 0.55.
  - Other builds kept for reference: `figAN_varA/` (old `atse_2`), `figAN5_varB/` (no *Nrxn1*), `figAN_grey_noiv/`.
- **Missing:** the caption, the panel a model schematic, and copying figAN into Overleaf `Figures/` (main.tex still includes `fig1_catalog.pdf`; wrapped, c17d35b). Panels c–f use the check 12 interval wording.
- **Panel e decided** (variant C, Karin 2026-11-11). Panel c is drawn from data in the figure theme (teal for LeafletFA), so the browser screenshots in section 5 are no longer needed as a panel.
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
- Readings so far (facts, no verdict):
  - *Cd44* and *Enah* in the expected direction in raw.
  - *Fgfr2* mesenchymal types compressed toward 0.4.
  - *Nrxn1* (row 3, rebuilt on SS4 `atse_7`): expected direction seen; SS4 inclusion excitatory 0.153 vs PV 0.367, SST 0.405 (LeafletFA medians; same data as figAN e).
  - *Ptbp2* and *Dlg4* rest on one TMS neuron type.
  - ~~Stale: `captions/FIG_GALLERY_CAPTION_FACTS.md` row c~~ fixed 2026-11-16 (54bbf47): row c now lists SS4 `atse_7`, 64 types (48 with interval), and the lineage medians.
- Citation volume and pages verified against PubMed (all match). Direction-in-mouse check (10 Nov): only row 3 (Nguyen 2016) is stated in mouse cell types; rows 1, 5–7 are human cell lines; 8–9 mouse developmental; 4 rat; 2 and 10 paywalled. Proposals in `GALLERY_KNOWN_BIOLOGY_DRAFT.md`.

## 7. Panel e: three variants (history; **resolved 2026-11-11: C**, see section 0a)
- **A, panel e kept as is** (`figAN_varA/`): the figure is unchanged; the caption states the fact.
  - Fact for the caption: the junction shown (`atse_2`, later found not to be SS4) is a skipping junction; it carries 1–2 % of the ATSE's reads in every neuronal group, and the expected direction (skipped more in excitatory, less in PV and SST) is not seen; the junction also skips two small exons.
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
