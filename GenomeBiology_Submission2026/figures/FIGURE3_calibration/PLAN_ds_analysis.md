# Figure 3 — execution plan for the LeafletFA differential-splicing analysis

Written 2026-08-01, while the final full-atlas fit trains. The LeafCutter half is **done**; this is
the plan for the LeafletFA half and the join, to run the moment the models land.

Sources reconciled here: the Notion outline *"New outline for LeafletFA resubmission"* (2026-06-22,
the paper-facing figure plan), the David meeting notes (2026-06-19, the robustness asks), and
`figures/FIGURE_OUTLINE_ALL.md` (2026-07-29, the newer analysis checklist).

---

## The claim

Notion, Section 3: *"Group-wise testing identifies cell-type marker ATSEs concordant with pseudobulk
Leafcutter."* Pseudobulk LeafCutter is **the calibrated anchor**; LeafletFA adds denoised effect sizes
and rare-cell-type resolution; **the concordance between them is the keystone figure.**

---

## Already in hand

**LeafCutter anchor — complete.** `$WD/LEAFCUTTER_DS_2026-08/`
- 654 pseudobulk samples (610 TMS keyed on 18 real donors, 44 Allen keyed on region × sex), labels
  `tissue_cell_type[_sex][_age][_donor]`, `--min-cells 25`.
- **74 pairwise contrasts, all completed, zero corruption flags**: 71 TMS within-tissue pairs across
  19 tissues + 3 Allen (`EXC vs INH` 12v12, `ASTROCYTE vs EXC/INH`).
- `ds_summary_all.tsv`: 128,685 significant ATSEs. Per-contrast medians — TMS 1,157 sig
  (range 130–4,370), Allen 8,921.
- Reference case saved: `reference_atse_Fbln2.json` + per-sample PSI + plot. Fbln2 exon 9 cassette,
  Heart endothelial vs fibroblast, |ΔΨ|=0.90, p=4.5e-31, consistent in all 13 donors.
- **Denominators are provably identical to LeafletFA's** (`cell_by_cluster_matrix[c,j]` == within-ATSE
  sum of `cell_by_junction_matrix`, 0/79,768 mismatches). So ΔPSI is the same estimand on both sides —
  this is what makes the effect-size panel meaningful. State it in Methods.

**LeafletFA DS code — complete and correct.**
`LeafletFA/leafletfa/differential_splicing_groups.py`: `group_delta_psi` (averages φ over the group
first, never materialises `[S,C,J]`), `differential_splicing_junctions`, `differential_splicing_atses`
(per-sample ATSE total variation), `bayesian_fdr`, `atse_simplex_check`. It already implements the
**max-detections FDR rule** and already avoids applying `|β|>δ` after the FDP is computed — two of the
three Methods fixes the meeting notes list are therefore code-complete. Use this module only; the
older `differential_splicing.py` is superseded.

---

## Two conflicts between the source documents — resolve these before building

**1. Spearman vs slope.** Notion's panel C legend says *"with [Spearman] agreement"*.
`FIGURE_OUTLINE_ALL.md` (newer) says the opposite, emphatically: *"Report the slope, not the
correlation… Hungarian-matched Pearson r, cosine and argmax agreement are all scale-invariant and
would score ~1.0 even under systematic attenuation. This exact blind spot made an earlier
batch-equivalence check incapable of detecting effect-size shrinkage."*

→ **Headline the slope (Deming/total-least-squares, both axes noisy); report Spearman alongside as
the rank-agreement number Notion promised.** LeafletFA's β is `φ̄@ψ`, a low-rank denoised estimate, so
some attenuation is *expected* — the point is to measure it, not hide it. A Spearman of 0.8 with a
slope of 0.6 is a publishable, honest result; a Spearman of 0.8 alone is not a claim.

**2. The worked example is Allen, but Allen is the weaker anchor.** Notion names inhibitory vs
excitatory as the worked example. But Allen pseudobulk replicates are region × sex strata, not
animals, so LeafCutter's overdispersion there is between-region variance and its p-values are
anti-conservative — visible as **55.6% of tested ATSEs significant** vs 18.1% in a comparable TMS
contrast.

→ **Split the roles.** Allen `EXC vs INH` carries panels A/B/E (schematic, marker example, Fyn) where
the job is illustration. **TMS carries panel C** (the quantitative concordance and slope) because its
replicates are 18 real donors. Say this in the legend rather than letting a reviewer find it.

---

## Panel plan (Notion's 6-panel layout, with the repo checklist folded in as supplementary)

| panel | content | status |
|---|---|---|
| A | group-wise DS schematic — posterior sampling → ΔPSI → ATSE total variation → Bayesian FDP | illustration, no compute |
| B | inhibitory vs excitatory marker ATSEs (Allen) | needs LeafletFA fit |
| C | **keystone**: LeafletFA ATSE effect size vs pseudobulk LeafCutter, TMS contrasts, **Deming slope** + Spearman | needs fit; LeafCutter side done |
| D | rare-cell-type resolution by **downsampling** — where pseudobulk loses power and LeafletFA does not | needs fit + downsampling runs |
| E | Fyn mutually-exclusive-exon marker (confirmed present: 2 ATSEs, 6 and 5 junctions) | needs fit |
| F | marker catalog summary — marker ATSEs per cell type **by annotation status** | ✅ unblocked, see below |
| **G** | **empirical false-positive rate under a donor-level permutation null** | **needs a layout decision — see below** |

**Panel G is new** — Notion's A–F layout has no calibration panel, but Reviewer #1 asked for an
empirical false-positive estimate directly and the figure is titled "calibration", so it belongs in
the main figure. Letters here are provisional; rearrange once real panels exist. It is the highest
value per unit cost of anything in the figure, because the LeafletFA half needs no refit (Step 4).

Supplementary, from `FIGURE_OUTLINE_ALL.md` and the reviewer asks:
- **S-a** seed reproducibility of the cell-type DS calls (Jaccard / slope / top-N across 3 seeds).
  Should reproduce well: cell-type labels are fixed, so groups are identical across seeds and nothing
  depends on program matching.
- **S-c** simulation FDR calibration. The pure-null number is **already done**
  (`LeafletFA/METHODS.md:95-110`, 0/4,268 ATSEs) — but a pure null measures FPR, not FDR, so the
  *curve* still needs the signal-bearing sims in `pipelines/09_simulation/`.

The donor-level permutation null is **not** supplementary — it is panel B. See Step 4.

**Panel F — resolved.** `annotation_status` is not in the atlas `var`, but it is in the ATSEmapper
output and joins on `junction_id` at **100%**:

    $WD/ATSE_mapper/ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz

Over the 79,768 atlas junctions: `both` 60,603 / `five_prime` 9,645 / `three_prime` 9,414 /
`unannotated` 106. The same file also carries `splice_motif` (GT-AG 141,128, GC-AG 1,749, AT-AC 19)
and `perfect_match_5_prime` / `perfect_match_3_prime`, which together answer Reviewer #2's minor ask:
*"What types of splicing events comprise each splicing program? Do some programs have a bias toward
exon inclusion?"* Pinned by `tests/test_ds_config.py::test_annotation_status_joins_completely`, which
fails if a single junction goes unmatched.

Note the `unannotated` class is only 106 junctions — too few to carry a "novel events" claim on its
own. Report the counts; do not build a story on that bin.

---

## Steps

### Step 1 — `A26_ds_vs_leafcutter.py` (new, `pipelines/05_leafletfa/`)

`--fit <fit.npz> --h5ad <atlas> --manifest <pairwise_manifest_*.tsv> --out <dir>`. No model path
hardcoded anywhere; the manifests define the contrasts so both methods test exactly the same pairs.

For each of the 74 contrasts, call `differential_splicing_junctions` and
`differential_splicing_atses` with `groupby="tissue_cell_type"` (TMS) or `"broad_cell_type"` (Allen),
matching how the LeafCutter groups were defined.

Three implementation points:

- **ψ sampling**: draw from `psis_loc`/`psis_scale`, streaming over junction chunks — copy
  `A23_ds_reproducibility.py::sample_psi()` (lines 39–55). Never materialise `[S,K,J]`.
- **φ has no posterior samples in `fit.npz`** — only a point estimate. Since
  `Δ_sj = (φ̄_A − φ̄_B)·ψ_s[:,j]` and φ̄ averages thousands of cells, use point φ̄ and let ψ carry the
  posterior width, **plus a bootstrap over cells within each group** for φ̄ error. State this choice in
  Methods. It also keeps the comparison resting on cell-type-level structure (reproduces at r=0.90)
  rather than per-cell φ (0.65–0.87), per `A12_select_final_model.py`'s caveat.
- **Run `atse_simplex_check` once and record the number.** ATSE total variation assumes within-ATSE
  imputed PSIs sum to ~1 and LeafletFA does not enforce that. **Quote no ATSE-level number until this
  passes.**

Restrict to the junctions LeafCutter actually tested (per-contrast, from `counts_subset.tsv.gz`) so
both methods see the same universe.

### Step 2 — observed pseudobulk ΔPSI, the third track

From `leafcutter_counts.tsv.gz`: per-sample junction PSI (count ÷ within-ATSE sum — denominators are
correct thanks to the explicit zero-fill), then group-mean ΔPSI per contrast. This is the **model-free
common axis**. Plot LeafletFA β *and* LeafCutter ΔΨ against it: LeafletFA's shrinkage is only
interpretable next to LeafCutter's own slope on the same axis. Without this, a slope <1 for LeafletFA
looks like a defect rather than the expected behaviour of a low-rank estimator.

### Step 3 — depth control (meeting notes, applies to every contrast)

*"Depth confound (all comparisons, including inhibitory vs excitatory): match coverage or add
log-coverage as a covariate so a marker is not a coverage artifact."*

Cheapest sufficient version: recompute the panel-C slope with ATSEs stratified into coverage terciles
(per-contrast median ATSE total). If the slope is stable across terciles, the marker set is not a
coverage artifact and one sentence disposes of the concern. If it isn't, that is itself the finding
and needs reporting.

### Step 4 — permutation null (Reviewer 1's literal ask) — **this is panel G**

Reviewer #1, major concern 2, verbatim:

> *"Although a Bayesian FDR framework is described for differential splicing, the manuscript does
> not provide a clear empirical estimate of false positives. Given the scale of junction-level
> testing and the sparsity of single-cell data, more explicit reporting of type I error control and
> robustness (**e.g., permutation-based null models or age-label shuffling**) would strengthen
> confidence."*

So the deliverable is an **empirical false-positive count on real data**, not a better-worded rule.

#### Two nulls, answering two different questions — do not conflate them

| | what it measures | needs |
|---|---|---|
| **Donor-split permutation, real data** | **type-I error / FPR.** Every discovery is false by construction. | real coverage, real donor structure, real overdispersion |
| **Simulation with true positives** | the **nominal-vs-realised FDR calibration curve**. FDR needs a denominator containing true positives. | ground truth |

Panel B needs both: the permutation null is the honest number *on the data we actually analysed*;
the simulation is the only place a genuine FDR curve can be drawn. The pure-null simulation is
already done (`LeafletFA/METHODS.md:95-110`, 0/4,268 ATSEs) — but note a pure null also measures FPR,
not FDR, so the *curve* still needs the signal-bearing sims from `pipelines/09_simulation/`.

#### Design: donor-split within a single cell type

Take one `tissue_cell_type`, split its **donors** (never its cells) into two arms, and test arm A vs
arm B. Both arms are the same cell type, so the true ΔPSI is exactly zero and every call is a false
positive. Any signal is donor-to-donor variation being misread as a cell-type difference — which is
precisely the failure mode the reviewer is asking about, and precisely what cell-level shuffling
would hide. Per the meeting notes: *"Cell-level shuffling breaks within-donor correlation and gives a
falsely clean null."*

**Powered on TMS: 34 tissue_cell_types have ≥8 donors** (so both arms clear LeafCutter's n≥4 floor),
led by `Heart_CARDIAC MUSCLE CELL` 15, `Skin_KERATINOCYTE` 14, `Brain_Myeloid_MICROGLIA` 14, and six
at 13. Distinct balanced splits: 3,217 for the largest, ~19,000 pooled across all 34. Permutations
are not the limiting resource.

Secondary flavour, on a **real** contrast rather than a synthetic one: paired **sign-flip**. For each
donor independently, randomly swap which arm its two cell types are assigned to. This preserves the
marginal group sizes and exercises the actual contrast machinery. The Fbln2 contrast has 13 donors
contributing to both arms → 2¹³ = 8,192 flips.

#### Allen: same design, with sex × tissue as the exchangeable unit

Allen has no donor, so **pair on `tissue × sex`** — the 6 brain regions × 2 sexes give 12 strata, and
those strata are the exchangeable unit in place of donors. Everything above carries over unchanged.

| Allen cell type | strata | balanced splits |
|---|---|---|
| EXCITATORY NEURON | 12 (6 regions × 2 sexes) | 462 |
| INHIBITORY NEURON | 12 | 462 |
| ASTROCYTE | 9 | 63 |

And the worked example is **perfectly paired**: all 12 strata contribute a sample to *both* EXC and
INH, so the sign-flip null on `EXCITATORY_NEURON vs INHIBITORY_NEURON` is an exact paired permutation
test with **2¹² = 4,096 flips**. That means panel G can report the false-positive rate for the very
contrast panel B illustrates, on the same 12 strata — a much tighter statement than a null computed
on some unrelated cell type.

**The honest caveat.** A region × sex stratum pools multiple animals, so between-stratum variance is
*smaller* than between-animal variance. The Allen null therefore under-represents biological
variability and its FPR is a **lower bound** — optimistic by construction. This is the same fact that
makes Allen's LeafCutter p-values anti-conservative (55.6% of tested ATSEs significant vs 18.1% in a
comparable TMS contrast), seen from the null side. So: **TMS carries the calibration claim; Allen's
null is a within-figure consistency check for the worked example, labelled as a lower bound.**

#### The cost asymmetry, and why it is a result in itself

**LeafletFA's null is nearly free, and LeafCutter's is not.** LeafletFA never sees cell-type labels
during training — it is unsupervised — so permuting labels afterwards requires **no refit and leaks
nothing**. Draw the ψ posterior samples *once* and reuse them across every permutation; each
permutation then costs one group-mean φ̄ (`[S,K]`) plus one `einsum` over junctions. LeafCutter's
Dirichlet-multinomial GLM must be **refit per permutation** (~16 min per contrast per permutation).

→ Budget **~1,000 permutations for LeafletFA** (a dense, smooth FPR curve) against **~30 for
LeafCutter** (a point estimate with a bootstrap CI). Report the two at their own resolutions and say
why they differ; do not throttle LeafletFA to match. That a label-free generative model gives an
exact permutation null for free, while the pseudobulk GLM needs a compute budget for the same
statement, is a genuine methodological point for the Methods section.

#### What panel B reports

- x-axis nominal threshold (posterior expected FDP for LeafletFA; BH-adjusted p for LeafCutter),
  y-axis realised false-positive rate under the donor-split null, with the diagonal marked.
- The headline sentence Reviewer #1 asked for: *"at a nominal FDP ≤ 0.05, the realised false-positive
  rate under a donor-level permutation null was X% across N splits of M cell types."*
- Both methods on the same splits, so the comparison is paired.

#### Watch-outs

- **Never shuffle cells.** Permute whole exchangeable units — donors on TMS, `tissue × sex` strata on
  Allen. Cell-level shuffling breaks within-unit correlation and gives a falsely clean null.
- **Allen's null is a lower bound**, because its strata pool animals. Report it as such; do not put
  the headline calibration number on it.
- **Use S ≥ 500 posterior draws here.** At S=100 `prob_de` is quantised to 0.01, so the smallest
  non-zero lfdr is 0.01 and any FDP below that is unresolvable — and this panel lives in exactly that
  tail. Pinned by `tests/test_ds_config.py::test_probability_resolution_at_S`.
- **Stratify or report by cell type.** A single cell type's FPR is one draw from a distribution; the
  claim should be the pooled rate across the 34, with the per-cell-type spread shown.

### Step 5 — downsampling for panel D

Subsample cells per cell type before pseudobulking — ladder ~500/250/100/50/25/10 per donor×type —
re-run both methods, plot recovery of the full-data marker set vs cells per type. The claim to
support is Notion's: *"markers recovered by LeafletFA in rare cell types where pseudobulk testing is
underpowered."* Two honest framings, both worth showing:
- the **curve** (LeafletFA degrades more gracefully), and
- the **hard floor** — LeafCutter simply cannot run below ~4 pseudobulk replicates, which is why 61 of
  96 TMS tissue_cell_types were testable and 35 were not. That list is already in the manifests.

### Step 6 — figure assembly

`figures/FIGURE3_calibration/` (already created). Join on `junction_atse_map.tsv`:
`adata.var["junction_id"]` ↔ `leafcutter_junc_id`, `event_id` ↔ LeafCutter `cluster`. Style with
`figure_common.py` + `leaflet_theme.py`. **Watch the ±1 coordinate convention**: junction coordinates
are regtools *intron* boundaries, so the skip junction's donor is the last base of the upstream exon
and its acceptor is one base before the downstream exon starts. An off-by-one in the join fails
silently — validate against `reference_atse_Fbln2.json`, which has known-good values.

### Step 7 — Methods text

`METHODS_calibration_leafcutter.md`, recording:
- the pairwise design and per-contrast covariate selection;
- **Allen replicates are region × sex strata, not animals** — anti-conservative, hence TMS carries the
  calibration claim;
- Allen `age` is a hardcoded constant (`allen_brain_mouse.py:73`), never used as a covariate there;
- point-φ̄ + bootstrap, and the simplex-check number;
- LeafCutter is a *reference tool*, not ground truth — the panel-C axis label must not say "accuracy";
- the whitespace-parsing fix, so the corrupted 2026-06-20 results are never reused.

Plus the two remaining Methods-text fixes from the meeting notes (the FDR one is already code-correct):
restore the sample index in the β equation (R4), and soften *"ensures at most 5% false positives"* to
*"controls the posterior expected FDP."*

---

## Acceptance checks before anything is quoted

1. `atse_simplex_check` passes on the released fit (within-ATSE imputed PSI sums ≈ 1.0).
2. The **Fbln2 reference case reproduces**: LeafletFA gives the same sign and roughly the same
   magnitude (skip junction ~+0.90 endothelial over fibroblast), and shows the split in all 13 donors
   across all three age groups. If it appears at some ages and not others, that is a red flag —
   LeafCutter says it is present in every donor at every age.
3. Panel C reports a **slope**, not only a correlation.
4. Donor-permutation FPR is at or below nominal α.
5. Panel-C slope is stable across coverage terciles.
6. Seed reproducibility (S-a) computed on the *group-vs-group* analysis — **not** the factor-level one.
   The factor-level contrast reproduces poorly (Jaccard 0.51, slope 0.72) and that is the wrong
   analysis for this figure; it is a caveat for the program-characterisation section instead.

## Open question for the run

Notion's meeting notes carry an unanswered ask: *"is the Python/Pyro `leafcutter-ds` port validated
against the R version?"* We are using the Python port
(`miniconda3/envs/LeafletSC/bin/leafcutter-ds` → LeafCutter2). If David has no validation, the cheap
check is to run 2–3 contrasts through R leafcutter and correlate the p-values — but note the R
package is **not currently installed** (the R 4.1 library has leafcutter but is missing `rstan`, and
the conda copies predate R 4.0), so this costs an install. Worth raising before it becomes a reviewer
question.
