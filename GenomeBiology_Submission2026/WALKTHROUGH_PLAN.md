# Planned: an HTML walkthrough of the whole LeafletFA workflow

Not built yet — captured now because much of the reasoning below currently exists only in commit
messages and one long working session, and it will be expensive to reconstruct later.

Intended as a single scrollable page a reader can follow from "what is an ATSE" to "here is the
differential-splicing result", with the real numbers and the dead ends included.

## Sections

**1. Background: what is being measured**
- A splice junction is a read spanning an exon-exon boundary; the count is how many reads support it.
- An ATSE (alternative transcript structure event) groups junctions that share a splice site, so
  their usage is a *composition* — the counts divide a fixed total, `sum_j y_j = N_e`.
- Worked cassette-exon picture: inclusion junctions J1/J2 versus the skip junction J3, PSI as the
  skip fraction. This is where the diagram earns its place.
- Why compositionality is not a detail: 10,677 of 24,475 atlas ATSEs have exactly 2 junctions, and
  scoring those junctions independently double-counts the same observation.

**2. The data**
- Mouse atlas: 132,706 cells x 79,768 junctions, 24,475 ATSEs, two datasets (tabula muris senis,
  allen brain).
- Sparsity is the central fact: 8.35M cell-junction entries have 1-2 reads. Any method must treat a
  1-read PSI of 0 as weak evidence, not as PSI = 0.

**3. The model**
- Generative model as written in `METHODS_dirichlet_multinomial.md`.
- Why Dirichlet-multinomial and not Beta-Binomial: the BB scores each junction against a shared
  denominator, inflating effective sample size and narrowing the posterior. Coverage measured at
  m=2: BB 80.7% vs multinomial 94.7% against a nominal 95%.
- Why the within-ATSE simplex: without it psi's within-event scale is unidentified (sums drift to
  ~1.47); with it, 100% of program-ATSE pairs are compositions versus 86.9% before.
- Verified against LeafCutter's `dm_glm.stan` term for term: -114.362812 vs -114.362793.

**4. Training, and what went wrong**
- The recipe and the measurement behind each choice.
- The honest part: three silent bugs (an alpha floor that never fired, a NaN rollback that skipped
  every constrained parameter and saved corrupt fits while reporting success, Adam state
  re-poisoning restored values within two steps), a learning-rate divergence traced by a repro that
  *failed* to reproduce, and a phi regression caused by the concentration collapsing before any
  structure formed.
- Why the ELBO cannot judge convergence here, and what replaced it.

**5. Validating on simulation**
- Compositional simulator, effect-size sweep, three seeds.
- Program recovery, psi recovery, seed agreement, and the imputation-is-denoising result
  (r rises 0.79 -> 0.92 with read depth; a model scoring 1.0 at 1-2 reads would be fitting noise).

**6. Differential splicing**
- The two tests, what each uses from the model, and why neither replaces the other.
- The design lesson worth showing: three null constructions failed for three different reasons
  before the realisation that the *hypothesis* was wrong, not the null. `Delta = 0` is essentially
  never true; truth is defined by magnitude.

**7. Against LeafCutter**
- Matched-null calibration, power at matched realised FDP (not matched nominal q — a miscalibrated
  test calls more by construction).
- What LeafletFA can do that LeafCutter cannot: DS across programs found without labels.

**8. What we would not claim**
- Effective K is 5-8 regardless of nominal K, so the K sweep is robustness, not model selection.
- phi reproduces across seeds at ~0.74 while psi reaches 0.97 — the per-cell encoding stays the
  weak direction.
- Every calibration caveat, stated rather than buried.

## Notes on building it

- Use the Artifact tool; inline everything (strict CSP, no external assets).
- Diagrams: the ATSE/junction picture and the model plate diagram are the two that genuinely help.
- Include the dead ends. The three failed nulls and the refuted phi-corner mechanism are more
  instructive than the successes and stop the next person repeating them.
- Source material: `METHODS_dirichlet_multinomial.md`, `METHODS_differential_splicing.md`,
  `ATLAS_TRAINING_DECISIONS.md`, `figures/DS_COMPARISON/PLOTTING_PLAN.md`.
