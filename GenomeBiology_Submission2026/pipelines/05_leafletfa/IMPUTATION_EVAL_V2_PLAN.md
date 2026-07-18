# Imputation Eval v2 — pure-prior + permutation, grounded in LeafletFA

Rigorous coverage-robustness validation for the released imputed resource.
Supersedes 04e for the **final** figure (04e stays as a coarse first pass).
Based on the Claude Science harness, adapted to how LeafletFA is actually built/trained.

## Design principle: use the data we already have (full + downsampled), no resampling

We already have full-coverage test cells + downsampled versions at 10/25/50/75/90/99/100%.
The convincing argument is **relative** — does LeafletFA's learned prior carry per-cell,
within-cell-type splicing information that methods with no learned structure lack? — built
directly from these layers.

**Headline avoids the read-overlap artifact by using PURE priors (no count blend).** Because
`phi` is inferred from `obs`, and `obs ⊂ target` (downsampled reads are a subset of the
full-coverage target), any `+obs_junc` blend (`lf_bb`) gets a read-overlap correlation with
the target that a train-derived prior (`meanCT`) lacks — so `lf_bb > meanCT_bb` on shared
reads is partly an artifact (confirmed in the Claude Science simulation). The clean claim is
the pure-prior comparison + within-cell-type permutation null (see Pipeline §5). `lf_bb` with
each model's own `bb_conc` is kept as a SECONDARY characterization, high-dropout only. A read
split would remove overlap fully but needs resampling — optional supplement, not required.

## The fixes that matter vs 04e (all use existing data)
1. **Feature-selection leak:** 04e computes HVJ from the *test* file. → compute HVJ (and any
   ATSE/threshold choices) on **train only**, apply to test. Cell-type labels come from obs
   (given, not derived) so no leak there.
2. **Within-cell-type scoring + permutation null** on the PURE priors (`lf`/`meanCT`/`NMF`) —
   the hard, artifact-free axis (Pipeline §5).
3. **Reads/cell x-axis**, emphasize the sparse regime.
(The count-fair BB blend is retained only for the secondary `lf_bb` panel, not the headline.)

## LeafletFA objects we plug in (no re-derivation)
- Per junction: `junc_ratio = cell_by_junction_matrix / cell_by_cluster_matrix` (PSI).
  `cell_by_cluster_matrix` = ATSE total (shared across an ATSE's junctions, per cell).
- Frozen globals from `leafletfa_model.pkl.gz`: `psi` (K×J), `pi`, `dir_conc`, and each
  model's **own learned `bb_conc`**. Per-junction `a,b` = the junction-specific Beta PRIOR on
  psi (`junc_specific_prior=True`); psi is FROZEN so that prior is already baked into `psi`.
- **Stay true to each model** — use its OWN learned `bb_conc` (it is part of the trained
  model). BB-trained models → native `lf_bb` with their `bb_conc`. Binomial-trained models →
  no `bb_conc`; native prediction is just `lf = phi@psi` (counts entered via phi) — do NOT
  force a blend they never had.
- Test cells: the 30% held-out split. **Input** = the existing RESAMPLED downsampled
  layers at each rate; **target** = `junc_ratio_original` (full coverage). No new resampling.
- E-step: the real variational inference from `04e/infer_phi_chunk` (freeze psi/pi/dir_conc,
  `input_conc_prior=inf`, infer phi). NOT a MAP placeholder.
- NMF: trained on TRAIN junc_ratio (as in 04d), projected onto test via `nmf.transform`.

## Pipeline (per model run being characterized)

1. **Train-only setup**
   - **HVJ computed on TRAIN** junc_ratio variance (NOT the test file as 04e currently does —
     this is the fix), expanded to full ATSEs via `var/event_id`.
   - Train per-junction mean PSI and per-**broad_cell_type** mean PSI (for `mean`/`meanCT`).
   - Fit/reuse NMF on TRAIN.

   **Two scoring sets — report both:**
   - **(a) HVJ-expanded ATSEs** — the ~7.8k ATSEs / ~31k junctions containing a top-variance
     junction (train-selected). The hard, interesting cases.
   - **(b) ALL junctions / ALL ATSEs** (~79k junctions) — the full-transcriptome picture;
     dominated by constitutive/low-variance junctions, so scores will look higher — report it
     for completeness and to avoid cherry-picking, but the HVJ set is the discriminating one.

2. **For each existing dropout level** (10/25/50/75/90/99%):
   - **Input** = downsampled counts `obs_junc, obs_atse` (RESAMPLED layers).
   - **Target** = `junc_ratio_original` (full-coverage PSI).
   - **Primary score mask = `atse_original > 0`** (target must be defined). Use the FULL-
     coverage `atse_original`, NOT the downsampled `atse_obs` — cells that lost all reads
     (`atse_obs=0` but `atse_original>0`) are the key hard-imputation cases; keep them.
   - **Sensitivity score mask = `atse_original ≥ 10`** (reliable targets only) — at 1–2 reads
     the target PSI is ~binary/noisy and inflates metric noise. Report BOTH; conclusions
     should hold under both. (Honest no-resampling stand-in for the read-split noise ceiling.)
   - Record **median retained reads/cell** at this level → the x-axis.

3. **Infer phi** from `obs_junc/obs_atse` via the real E-step (fixed psi) → `phi_test`
   (this is what 04e already does).

4. **Priors (no counts):** `lf=phi_test@psi`, `NMF=W@H` (project downsampled through NMF),
   `mean` (train per-junction mean), `meanCT` (train per-cell-type mean).

5. **HEADLINE = pure-prior comparison (no count blend), scored WITHIN cell type + permutation
   null.** This avoids the shared-read artifact entirely (see note below):
   - Priors: `lf = phi@psi`, `meanCT` (train cell-type mean), `NMF = W@H`. (`naive` = 0 ref.)
   - **Within-cell-type** per-junction correlation, averaged over broad_cell_type.
   - **Permutation Δ:** matched vs within-CT-permuted → cell-specific info.
   - Report vs reads/cell; emphasize the sparse regime.
   - Honest expectation: strong cross-type structure, modest within-type — that IS the result.

6. **SECONDARY (characterization): native imputation `lf_bb`** with each model's own
   `bb_conc` (Binomial models → `lf`). Show `lf_bb` vs `meanCT_bb` (meanCT blended with the
   SAME model's `bb_conc`) but **only interpret at HIGH dropout**, where `obs_junc` is tiny so
   read overlap is minimal. Do NOT headline `lf_bb > meanCT_bb` across all dropout.

### Read-overlap note (why the headline is pure-prior)
Because `phi` is inferred from `obs`, and `obs ⊂ target` (downsampled reads are a subset of
the full-coverage target), any `+obs_junc` blend gives `lf_bb` a read-overlap correlation
with the target that `meanCT_bb` lacks — so `lf_bb > meanCT_bb` on shared reads is partly an
artifact (confirmed in the Claude Science simulation). The pure-prior comparison has no
`+obs_junc` term; residual overlap is only the weak phi→obs path (and `lf`≈0 anyway), so it
is the clean, defensible claim. A read-split would remove overlap fully but requires
resampling — deferred as an optional supplement only.

7. **Controls**
   - **Donor/batch hold-out** metric (group by `mouse.id`/batch): report within-donor too,
     since by-cell pooling can be flattered by donor structure.
   - (Optional supplements only if challenged) read-split reliability ceiling; generative
     upper bound (simulate from fitted model, re-infer, score).

## Outputs
- Table: within-CT per-junction correlation per method × reads/cell (pure priors).
- Fig A (headline): within-CT recovery vs reads/cell — `lf` vs `meanCT` vs `NMF` (`naive`=0).
- Fig B (headline): permutation Δ for `lf` vs reads/cell (shaded = cell-specific info).
- Fig C (secondary): native `lf_bb` vs `meanCT_bb` (same model's `bb_conc`), HIGH dropout only.
- **Claim statement:** state where `lf` carries within-CT cell-specific signal (Δ>0) and how
  it degrades with reads/cell. If within-type Δ is ~0, report that honestly — it's the
  "cross-type >> within-type" finding, not a failure.

## Scope
- Run on the **top classification candidates only** (e.g. run_91 K=50, run_61/85 K=20,
  run_11 K=10) — this is characterization, not model selection. Not all 96.
- New script `04f_evaluate_imputation_withinCT.py`; 04e keeps running for the coarse pass.

## Integration checklist (the only real correctness items)
- [ ] Headline is PURE-PRIOR (`lf`/`meanCT`/`NMF`) within-CT + permutation Δ — no shared-read
      artifact. `lf_bb` (native `bb_conc`) is secondary, high-dropout only.
- [ ] Each model's OWN learned `bb_conc` used for its native `lf_bb`; Binomial models → `lf`.
- [ ] E-step concentration: `inf` (sharp, old-05 + 04e default). Optional sensitivity check
      vs the model's learned `bb_conc` (BB models only).
- [ ] Real E-step wired in (not MAP).
- [ ] Per-junction `a,b` (psi prior) already in frozen psi — do NOT feed separately.
- [ ] HVJ / means / NMF all fit on TRAIN, applied to TEST. (04e currently selects HVJ on
      TEST — must switch to TRAIN in 04f.)
- [ ] Score BOTH sets: (a) HVJ-expanded ATSEs and (b) all junctions/ATSEs.
- [ ] Per-junction scoring (LeafletFA models each junction BB — NO Dirichlet-multinomial).
- [ ] Reads/cell as the x-axis; within-CT correlation vs `meanCT_bb` as the headline.
- [ ] No new resampling — reuse existing full + downsampled layers.

_See `THESIS.md` for how this serves the release claim; `memory/project_imputation_eval_diagnosis.md` for background._
