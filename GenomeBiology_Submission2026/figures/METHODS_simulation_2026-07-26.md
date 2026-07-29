# Methods — simulation of single-cell alternative splicing with known ground truth

*(draft for the manuscript; all counts verified against the 2026-07-26 regeneration logs)*

## Rationale

Evaluating factor recovery requires a dataset in which the true splicing programs are known. Rather
than simulate cells from scratch — which would require modelling capture efficiency, depth and
dropout, and would make the benchmark a test of those assumptions — we **retained the real coverage
structure of the data and simulated only the splicing outcome**. Concretely, for every (cell, event)
pair we kept the observed total read count exactly as measured, and re-drew only *which* junction of
that event each read supports. The simulation therefore inherits the real data's sparsity, depth
distribution and zero-inflation, and the only quantity under our control is the splicing signal we
are trying to recover.

## Source data and cell selection

Simulations were built from the mouse splicing AnnData released with this study
(`splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad`). We restricted to a single
tissue, **heart**, giving **6,986 cells × 79,768 junctions**, so that the simulated dataset is of a
size that all benchmarked methods can be run on exhaustively across a parameter sweep.

## Restriction to cassette exons

We restricted the simulation to **cassette (exon-skipping) events**, the simplest and most
interpretable alternative transcript structure event (ATSE):

```
upstream exon ---J1---[ cassette exon ]---J2--- downstream exon
              \________________J3 (skip)_______/
```

A cassette comprises exactly three junctions — two inclusion junctions (J1, J2) and one skipping
junction (J3) — so the entire event collapses to a single interpretable quantity per program: the
skipping usage Ψ. Filtering proceeded as follows:

1. Junctions covered in fewer than 50 cells were discarded (**57,726 junctions retained**).
2. Among three-junction ATSEs (**n = 5,566**), cassettes were identified from genomic coordinates
   rather than from the ordering of junctions within the event: the skipping junction is the one
   spanning the other two (`start = min`, `end = max`), and the two inclusion introns must abut it
   and not overlap each other. **4,268 of 5,566 events (76.7%) are canonical cassettes**; the
   remainder (alternative 5′/3′ splice sites, mutually exclusive exons) were excluded rather than
   assigned cassette roles they do not have.
3. An ATSE was retained only if all three of its junctions survived coverage filtering.

The final simulated object contains **6,986 cells × 12,804 junctions across 4,268 cassette events**.

## Generative model

Let *K* = 5 be the number of splicing programs.

**Cell loadings.** Each cell was assigned a primary program seeded by its annotated broad cell type
(the *K*−1 most frequent types map to their own program; all remaining types collapse into the last
program), and its loading vector drawn as φ*ᵢ* ~ Dirichlet(α), with α = 20 on the assigned program
and 1 elsewhere. This yields cells that are predominantly but not exclusively one program (mean
maximum loading ≈ 0.9), matching the graded usage seen in real data. Cell-type labels are used
**only** to seed the simulation; no method is given access to them.

**Program splicing profiles.** For each cassette *a*, a baseline skipping usage was drawn
Ψ₀ ~ Beta(0.5, 0.5), a U-shaped prior placing mass near 0 and 1 as observed empirically. Each
cassette was then independently designated differential with probability *f* (the
`--frac-differential` parameter):

- **non-differential events**: Ψ*ₖₐ* = Ψ₀ for every program *k*;
- **differential events**: Ψ*ₖₐ* = (1 − e)·Ψ₀ + e·Beta(0.5, 0.5), drawn independently per program,
  with effect size *e* = 1 throughout (programs fully independent at differential events).

Designation is at the level of the **event**, not the junction: a cassette is either differentially
spliced or it is not, and all three of its junctions inherit that status, since their usages are
constrained to sum to one within the event.

Junction-level probabilities follow from the cassette structure:

  P(J3) = Ψ*ₖₐ*,  P(J1) = P(J2) = (1 − Ψ*ₖₐ*)/2,

which sum to 1 within each event (asserted at runtime).

**Definition of the effect size ΔΨ.** For each cassette *a* the ground-truth effect size is the
**range of skipping usage across programs**,

  ΔΨ*ₐ* = max*ₖ* Ψ*ₖₐ* − min*ₖ* Ψ*ₖₐ*,

i.e. the difference between the program that skips that exon most often and the program that skips
it least often. It is a range, not an average over program pairs. For example, a cassette with
per-program skipping usage (0.73, 0.65, 0.84, 0.86, 0.85) has ΔΨ = 0.86 − 0.65 = 0.21: the exon is
skipped 86% of the time under one program and 65% under another, a 21-percentage-point difference.
Non-differential cassettes have ΔΨ = 0 exactly, since every program shares the same Ψ₀.

Because ΔΨ is a range it is driven by the two extreme programs, so a cassette in which four programs
agree and one differs receives the same ΔΨ as one in which usage is spread evenly. The estimator is
defined identically on the fitted profiles (max*ₖ* Ψ̂*ₖₐ* − min*ₖ* Ψ̂*ₖₐ*), so the truth and the
estimate are measured on the same footing.

Where a single number summarises a whole simulation — for instance the x-axis of the effect-size
sweep — it is the **median ΔΨ over the differential cassettes only** (roughly 1,280 of 4,268 at
*f* = 0.3), i.e. the typical effect size that dataset contains. The four sweep levels correspond to
median ΔΨ of 0.084, 0.210, 0.421 and 0.841: shifts of roughly 8, 21, 42 and 84 percentage points
between the most- and least-skipping programs. Reported differential splicing in real data is
typically 5–30 percentage points, i.e. the lower two levels.

**Counts.** The per-cell splicing probability at junction *j* is *p*ᵢⱼ = φᵢ ᵀ Ψ·ⱼ. Junction counts
were drawn as

  *n*ᵢⱼ ~ Binomial(*T*ᵢₐ, *p*ᵢⱼ),

where ***T*ᵢₐ, the total read count for cell *i* at event *a*, is taken verbatim from the real
`cell_by_cluster_matrix`** and is not simulated. Cells with no coverage of an event receive no
counts, exactly as in the observed data. An overdispersed variant is available in which
*q*ᵢⱼ ~ Beta(*p*ᵢⱼ·c, (1−*p*ᵢⱼ)·c) is drawn first and *n*ᵢⱼ ~ Binomial(*T*ᵢₐ, *q*ᵢⱼ); as *c* → ∞ the
Beta collapses to a point mass and the Binomial is recovered exactly (the implementation
short-circuits the Beta draw in this limit, so the two code paths are provably identical). Results
reported here use the Binomial emission.

## Parameter sweeps

Two orthogonal quantities govern how much signal the simulation contains, and both were swept:

**Effect size (primary).** The parameter *e* sets *how different* the programs are at a
differential cassette, interpolating between "identical to a null event" (*e* = 0) and "independently
drawn per program" (*e* = 1). We swept *e* ∈ {0.1, 0.25, 0.5, 1.0} at a fixed 30% differential
fraction with three seeds, giving **twelve datasets**. The corresponding median true effect sizes are
|ΔΨ| ≈ 0.085, 0.21, 0.42 and 0.85. This is the axis reported in the main figure, because it spans the
range over which recovery actually changes: real differential splicing is typically |ΔΨ| ≈ 0.05–0.3,
i.e. the lower half of this range, whereas *e* = 1 produces differences (|ΔΨ| ≈ 0.85, a cassette going
from almost fully included to almost fully skipped) that every method resolves easily. Results are
plotted against median |ΔΨ| rather than *e* so the axis is directly interpretable.

**Overdispersion of the counts (likelihood ablation).** The sweeps above draw counts from a
Binomial, so a Binomial likelihood is correctly specified and a Beta-Binomial can at best match it.
To ask whether modelling overdispersion is worthwhile, the same simulation was regenerated with a
finite Beta-Binomial concentration — *c* = 20 (mild) and *c* = 2 (matching the concentration of
≈1.96 the atlas models learn) — at a fixed effect size (*e* = 0.25) and 30% differential cassettes,
three seeds each, and both likelihoods were fitted to each dataset. At *c* = 2 the marginal variance
is inflated roughly fivefold relative to a Binomial at the observed read depths.

**Differential fraction (secondary, supplementary).** The fraction of cassettes made differential
was varied over *f* ∈ {0.1, 0.5, 0.9} with three seeds at *e* = 1, giving **nine datasets**. Because
designation is an independent Bernoulli draw per event, realised counts vary around the target
(428–455 of 4,268 events at *f* = 0.1). This axis changes *how many* events carry signal rather than
how strong it is, and recovery is comparatively insensitive to it; it is reported in the supplement.

## Choice of likelihood

The two likelihoods are indistinguishable when the counts are not overdispersed: on Binomial-truth
simulations the Beta-Binomial and the Binomial agree to within 0.006–0.021 on every metric with no
significant difference (*P* ≥ 0.30), so the extra concentration parameter costs nothing.

When the counts *are* overdispersed the Beta-Binomial pulls ahead, and it does so monotonically with
the degree of overdispersion. At *c* = 20 (mild) effect-size recovery is 0.490 versus 0.396
(*P* = 0.038) and the silhouette 0.631 versus 0.504 (*P* = 0.055); at *c* = 2 — the level the atlas
exhibits — the Binomial retains almost no quantitative signal, with effect-size recovery 0.055 versus
0.275 (*P* = 0.001) and silhouette 0.007 versus 0.249 (*P* = 0.025). Notably Ψ recovery does not
separate the two at any level (−0.008 to +0.003, all *P* ≥ 0.55, and exactly 0.925 for both at
*c* = 2), a further illustration that profile correlation alone cannot distinguish a model that has
captured program structure from one that has not.

Since the atlas models consistently learn a concentration of ≈1.96 across every K and seed —
i.e. substantial overdispersion — the Beta-Binomial likelihood is used throughout.

## Choice of Ψ prior

The junction-specific Beta prior on Ψ behaves the same way: it is inconsequential except in the
regime the atlas occupies. On Binomial-truth simulations it is indistinguishable from a single shared
prior on every metric (differences +0.004 to +0.021, all *P* ≥ 0.46), and under a Binomial likelihood
it does nothing at any overdispersion level (+0.003 or less, all *P* ≥ 0.30) — expected, since a
Binomial has already discarded the extra variance the prior would regularise against.

Under a Beta-Binomial likelihood on overdispersed counts (*c* = 2, ten seeds) the junction-specific
prior improves Ψ recovery from 0.821 ± 0.034 to 0.930 ± 0.008 (*P* = 0.011) and reduces the
run-to-run standard error roughly fourfold: individual fits with a shared prior occasionally
converge poorly, whereas none of the junction-specific fits did. The improvement is specific to Ψ
recovery and to stability — effect-size recovery (+0.032, *P* = 0.11) and silhouette (+0.020,
*P* = 0.53) do not reach significance even at this sample size. The junction-specific prior is used
throughout on that basis.

## Ground truth retained

Each simulated object stores the complete truth: per-program skipping usage per cassette
(`uns['true_psi_skip']`), the effect size ΔΨ per cassette (`uns['atse_delta_psi']`), the true cell
loadings (`obsm['true_phi']`), the differential/non-differential label (`var['is_differential']`),
junction roles (`var['junction_role']`), and the full parameter set with source path and timestamp
(`uns['sim_params']`).

## Benchmarking

All methods were fitted at the true *K* = 5 on each of the nine datasets.

**LeafletFA** was fitted using the same minibatch stochastic variational inference implementation
used for the atlas models (subsampling plate, batch size 2,048, learning rate 0.05 with decay,
maximum 5,000 steps with early stopping on the evidence lower bound), under both Beta-Binomial and
Binomial likelihoods and with either junction-specific or shared Beta priors on Ψ. Each run asserts
that the traced local variable is batch-shaped, so that subsampling cannot silently be inactive.

**NMF** was fitted at the same rank (`scikit-learn`, `nndsvda` initialisation, 300 iterations) to
the per-cell junction usage matrix. Because NMF has no model for missing data, uncovered entries
must be filled; both conventions are reported — zero-imputation (which incorrectly asserts that an
uncovered junction is fully skipped) and mean-imputation (each junction's mean over covered cells).
LeafletFA models coverage explicitly and performs no imputation.

**Evaluation metrics.** Ψ recovery is the median Pearson correlation between fitted and true
skipping-usage profiles after optimal one-to-one matching of programs (Hungarian algorithm on
|correlation|). Effect-size recovery is the Spearman correlation between true and estimated ΔΨ
across differential cassettes. Cell-level structure is quantified by the silhouette coefficient of
the fitted loadings labelled by true program, and by the adjusted Rand index between argmax-assigned
and true programs. Detection performance is summarised by AUROC and average precision for ranking
differential against non-differential cassettes, and by precision at a calling threshold of
ΔΨ > 0.10. Panels report the mean over the nine datasets (three differential fractions × three
seeds) with standard errors; comparisons between methods use Welch's *t*-test.

## Code availability

`pipelines/09_simulation/00_simulate_splicing.py` (simulation),
`figures/FIGURE2_benchmark/sim_benchmark.py` (fitting and evaluation),
`figures/FIGURE2_benchmark/build_figures.py` (figures).

---

# Methods — model fitting, inference and held-out evaluation

## Inference

LeafletFA is fitted by stochastic variational inference in Pyro. Cells are conditionally independent
given the global variables, and this is expressed with an explicit subsampling plate
(`pyro.plate("cells", N, subsample_size=B)`); the variational family is a mean-field normal guide
constructed over the same plate (`AutoNormal` with `create_plates`), so the guide subsamples the
local variables jointly with the model and Pyro scales the local terms by *N*/*B* while leaving the
global terms at weight one. Optimisation uses ClippedAdam (gradient norm clipped at 10) with an
exponentially decaying learning rate.

Training runs to a step *cap* rather than a fixed budget: an exponential moving average of the
evidence lower bound is monitored, and training stops when it fails to improve by more than a
relative tolerance (10⁻⁴) for eight consecutive checkpoints, at which point the best (lowest-EMA)
parameter state is restored. The tolerance is relative because the ELBO is of order 10⁶–10⁸ at
atlas scale, where an absolute tolerance is numerically inert. The same checkpointing doubles as a
guard against non-finite losses: if a step returns NaN the last finite state is restored and
training stops, so a single bad gradient cannot destroy a completed run.

**Reliability.** On the simulation benchmark this implementation converges to the same solution
regardless of seed, differential fraction or batch size: across the nine simulated datasets Ψ
recovery was 0.996 ± 0.001 (mean ± s.d., range 0.995–0.997). We note this explicitly because an
earlier implementation of the same model — full-batch, without the subsampling plate — reached the
same optimum on only about a third of runs (Ψ recovery 0.80 ± 0.18, range 0.40–0.99, bimodal), so
the variability reported by that implementation reflected optimisation failures rather than
statistical uncertainty. A factor-by-factor ablation (learning-rate decay, number of ELBO particles,
early stopping, batch size, and all of them jointly) reproduced the improved result under every
combination of the older hyperparameters, localising the difference to the inference implementation
rather than to any tuning choice.

## Reproducibility on the atlas

The simulation establishes that the trainer recovers known programs reproducibly (Ψ recovery
0.996 ± 0.001 across seeds). That argument cannot be made directly on the atlas, where there is no
ground truth — but it can be made *between* models: fits differing only in random initialisation
should agree with one another if the programs are a property of the data rather than of a single
optimisation run.

Comparing the three models at each *K* pairwise, with programs matched one-to-one by correlation,
gives a **median matched correlation of 0.97** (0.961–0.983 across all nine pairwise comparisons).
Every program exceeded *r* = 0.9 at *K* = 10 (10/10 in each pair); 18 of 20 did at *K* = 20 and
29–30 of 30 at *K* = 30. Independent initialisations therefore recover the same splicing programs
from the real data.

Agreement is lower further down the pipeline, for a reason that is mechanical rather than a
deficiency of the model. Ψ is estimated from every training cell and is correspondingly well
determined. A cell's loadings φ*ᶜ* are estimated from that cell's reads alone, which at ~13.5%
junction coverage is a sparse estimate: after matching programs, per-cell loading correlations
between fits have a median of 0.65–0.87, with 24–46% of cells above 0.9. A discrete Leiden
clustering of φ compounds that noise further, and cluster identity agrees only at ARI ≈ 0.31 between
fits.

### How many programs are actually determined by the data? (effective *K*)

The explanation above — that φ is noisier than Ψ because each cell's loadings come from that cell's
reads alone — is incomplete, and we can show it is. If sparse per-cell evidence were the cause,
agreement should improve with sequencing depth. It does not: median per-cell agreement between seeds
by depth quintile is 0.833, 0.919, 0.808, 0.912, 0.853 — flat across a 450-fold range of read depth.

Matching programs one-to-one before comparing φ also assumes each program in one fit corresponds to
exactly one program in another. If two fits divide the same variation differently, that assumption
fails and the disagreement it reports is partly an artefact of the matching. **Canonical correlations
between the two φ matrices** ask the question the matching cannot — do the fits span the same
subspace, however they happened to divide it? They are invariant to any invertible linear map of
either side, so they are blind both to program ordering and to any redivision among programs
(`A10_effective_K.py`; requires φ for the same cells under different seeds, which the held-out E-step
caches provide).

At *K* = 10 the spectrum has a sharp cliff:

```
seed 0 vs 1:  1.00  0.998  0.996  0.984  0.954 │ 0.882  0.810  0.600  0.330  0.116
seed 0 vs 2:  1.00  0.996  0.994  0.988  0.980 │ 0.968  0.891  0.810  0.298  0.033
seed 1 vs 2:  1.00  0.997  0.992  0.981  0.976 │ 0.926  0.721  0.653  0.077  0.052
```

Five to six directions are essentially identical between fits; the remainder are near-orthogonal,
i.e. not determined by the data at all, so each fit picks its own. Counting canonical correlations
above 0.9 gives an **effective *K***. The fraction is taken against ***K* − 1**, not *K*: φ lies on the
simplex, so *K* programs span only *K* − 1 directions and the *K*-th singular value of the centred φ is
exactly zero. Reporting against *K* would understate reproduction — at zero overdispersion in
simulation, 4 of 5 is complete reproduction, not 80%.

| nominal *K* | effective *K* | fraction reproducing |
|---|---|---|
| 10 | 4.7 | 52% |
| 20 | 10.0 | 53% |

The fraction is stable rather than saturating: doubling *K* doubles the reproducible dimensions, so
the atlas has not exhausted its recoverable structure by *K* = 20, and the extra programs are
capturing real signal rather than fitting noise. This is consistent with the lineage separations
sharpening with *K* (above), and is the quantitative reason to prefer higher *K* for downstream
biology while reporting the reproducible subspace rather than all *K* programs.

The threshold of 0.9 is a convention; the *K* = 10 spectrum has an unambiguous cliff, but the *K* = 20
spectrum declines more gradually, so the count is more threshold-sensitive there and the full spectrum
is reported in `A10_effective_K.csv`.

**What is and is not reproducible.** Two candidate explanations for the unshared directions can be
excluded. There is **no tail of lightly used programs**: π is close to uniform at every *K*
(perplexity 9.75 of 10 and 19.03 of 20, i.e. 97% and 95% of the even value), no program in any seed
falls below half the uniform share, and the sorted usage profile itself reproduces across seeds at
*r* = 0.92–0.94. And **usage does not predict reproducibility**: the lowest-usage half of programs
have Ψ matched at median *r* = 0.972 against 0.982 for the highest-usage half at *K* = 10, with the
same negligible separation at *K* = 20 and 30 (Spearman ρ = +0.25 to +0.38, *P* < 0.04 — significant
but small).

So the model is not carrying dead factors, and every program's *profile* is well determined. What is
degenerate is the **encoding**: how an individual cell distributes its mass across programs. This is
the expected signature of overlapping, non-orthogonal programs — a cell can be reconstructed by more
than one combination of them, so the per-cell decomposition is not unique even where every component
is. It is also the single mechanism behind the whole reproducibility hierarchy: Ψ (0.97) and π (0.93)
are stable, cell-type *mean* φ transfers to ~4% because averaging over cells cancels the degeneracy,
per-cell φ agrees at only 0.65–0.87 because the degeneracy is per-cell, and a discrete clustering of
φ agrees least of all (ARI 0.31) because discretising a degenerate encoding is the worst case.

This also explains the earlier observations rather than sitting alongside them: the strong,
biologically interpretable blocks in the program × cell-type map live in the shared subspace — which
is why the lineage orderings replicate 8–9 times out of 9 — while the finer within-block texture lives
in the unshared directions and looks different in every seed.

We report all three levels because either alone is misleading: unstable clusters are not evidence of
an unstable model, and reproducible programs do not imply reproducible per-cell assignments. Those
Leiden partitions are not technical artefacts — their agreement with sequencing-depth quintile
(ARI 0.012), batch (0.056) and dataset (0.056) is negligible — they are simply a discretisation of a
continuous representation that is itself only moderately determined per cell.

## How the programs relate to cell identity, and whether that relationship generalises

Agreement against a discrete clustering is a poor summary of a continuous representation, and an ARI
against annotated cell type additionally penalises any splicing structure that does not coincide with
cell identity — the labels come from gene expression, so partial alignment is the expected result
rather than a failure. We therefore report the relationship descriptively: the **mean loading of each
program within each annotated cell type**, computed separately for the cells used in training and for
held-out cells whose φ was inferred by the frozen-globals E-step. Because both maps come from the
same model, the programs are the same objects and no matching is required; the comparison asks
directly whether cells the model never saw use the programs as training cells do.

A cell type enters the map only if it has ≥50 cells on **both** sides, since a mean over a handful of
cells is noise. At *K* = 10 that is 27 of the 39 annotated types — 98.3% of cells on each side. The
12 excluded types are present in both splits and are excluded on count alone.

**Cell-type program usage reproduces on held-out cells to within 4.4% relative error** (3.9–4.9%
across the three seeds), measured on the entries with a loading above 0.05 — the ones carrying actual
signal. Rank agreement over the whole map is *ρ* = 0.987, the mean absolute deviation is 0.005 on
entries averaging 0.100, and the largest single disagreement anywhere in the map is 0.098.

**Every one of the 27 cell types loads most strongly on the same program in held-out cells as in
training cells — 100% in all three seeds.**

We also report the Pearson correlation between the two maps (*r* = 0.998, 0.999, 0.999), but only
against a reference, because on its own it is a flattering statistic: the map is mostly near-zero
entries with a handful of large ones, so *r* is carried by whether the large blocks agree. Its value
lies in the comparison below, not in its magnitude.

### Controls: what the agreement is measured against

An agreement figure means nothing without a scale, so three reference points are computed
(`A9_transfer_controls.py`), all on the same maps:

| comparison | *r* | what it establishes |
|---|---|---|
| **sampling floor** — the same model and the same *training* cells, subsampled to the held-out set's size | 0.9995 | how much two maps differ for no reason but sample size |
| **train vs held-out** (transfer) | 0.9984 | transfer costs **0.0011** relative to that floor |
| a **different seed's** held-out map, programs matched | 0.7486 | the map is model-specific — held-out φ is shaped by *its own* model's Ψ, not generic |
| held-out map with **cell-type labels shuffled** | 0.2435 | the agreement is carried by cell identity, not by a shared marginal |

The first two are the substantive comparison: generalising to cells never seen costs about as much as
looking at a smaller sample of cells already seen. The last two rule out the two ways that could have
been vacuously true.

Held-out φ is also **not prior-dominated**, which would have produced a good-looking map from no
information: held-out cells commit to programs as sharply as training cells (median perplexity 2.37
vs 2.37 of *K* = 10, against 9.75 for the prior alone; median maximum loading 0.65 vs 0.66; 80.5% of
held-out cells place over half their mass on one program).

This is a cell-type-level statement and should be read alongside the per-cell figures above. There is
no per-cell correspondence between training and held-out cells — they are different cells — so the
comparison is necessarily between **mean φ per cell type**, and averaging the tens to hundreds of
cells within a type cancels per-cell noise. The aggregate structure transfers within ~4% while
individual cells agree more modestly (median per-cell *r* 0.65–0.87 between seeds). Both are
properties of the same representation, and neither substitutes for the other.

The map is also biologically coherent, without any label having entered fitting. At *K* = 10 the
programs organise the atlas along recognisable lineages:

| program | cell types loading it most | interpretation |
|---|---|---|
| 1 | smooth muscle (0.91), fibroblast (0.90), mesenchymal stem (0.87), endothelial (0.86), stromal (0.81), skeletal muscle (0.79), cardiac muscle (0.65) | mesenchymal / structural |
| 2 + 3 | excitatory neuron (0.53, 0.32), hippocampal neuron (0.52, 0.32) | excitatory neuronal |
| 5 + 6 | inhibitory neuron (0.47, 0.27) | inhibitory neuronal |
| 4 | granulocyte (0.81), myeloid immune (0.65), monocyte (0.63), macrophage (0.61), and the lymphoid types | haematopoietic, broadly |
| 4 + 8 | NK (0.56, 0.37), B (0.55, 0.38), T (0.51, 0.42) | lymphoid — program 8 is what separates them from myeloid cells, which load it at ≤ 0.14 |
| 7 | intestinal (0.55), hepatocyte (0.52), epithelial (0.41), basal (0.39), pancreatic islet (0.39), keratinocyte (0.28) | epithelial / secretory |
| 10 | astrocyte (0.76), oligodendrocyte (0.65) | macroglial |

Two features are worth noting because they are not what a naive expectation would give. Excitatory and
inhibitory neurons are assigned **disjoint** program pairs (2/3 versus 5/6) rather than sharing a
generic "neuron" program, and the two excitatory types are nearly indistinguishable from each other.
And microglia do **not** load the macroglial program 10; they load programs 1 and 4 (0.74, 0.21),
consistent with their myeloid rather than neuroectodermal origin. Neither pattern could have been
induced by the annotations, which are never seen during fitting.

### Do these lineage relationships replicate?

Read off a single heatmap, the above are anecdotes. We therefore restated each as an **ordering
between two profile similarities** and re-tested it on all nine models (`A8_lineage_consistency.py`).
The relational form is what makes the test meaningful: program identity is arbitrary and not
comparable between fits, so an index-based test would require cross-model matching, and it would not
be comparable across *K* either — asking whether two cell types share their top two programs becomes
strictly harder as *K* grows, so a model that resolves astrocytes and oligodendrocytes into separate
programs (the finer, better answer) would be scored as a failure. Similarity is the Pearson
correlation between two cell types' mean-loading profiles across programs.

| claim, as an ordering | holds | mean gap (min) | *P* |
|---|---|---|---|
| excitatory types resemble each other more than they resemble inhibitory neurons | **9/9** | +0.618 (+0.192) | 0.0039 |
| microglia resemble myeloid types more than they resemble macroglia | **8/9** | +0.547 (−0.072) | 0.0078 |
| B/T/NK resemble each other more than they resemble monocyte/macrophage/granulocyte | **9/9** | +0.188 (+0.003) | 0.0039 |

*P* from a Wilcoxon signed-rank test on the nine paired gaps. All three orderings therefore hold
across random initialisations and across *K*, and are properties of the data rather than of one fit.

The single exception is instructive rather than troubling: the one model that fails the microglia
ordering (*K* = 10, seed 1, −0.072) is also the model with the weakest lymphoid/myeloid separation
(+0.003). The two gaps widen monotonically with *K*:

| | *K* = 10 | *K* = 20 | *K* = 30 |
|---|---|---|---|
| microglia gap (mean, min) | 0.498, −0.072 | 0.572, 0.328 | 0.572, 0.455 |
| lymphoid gap (mean, min) | 0.067, 0.003 | 0.149, 0.092 | 0.348, 0.232 |

At *K* = 10 there are not enough programs to separate the immune compartment, and which sub-lineages
get resolved depends on the initialisation; by *K* = 30 the separation is unambiguous in every seed.
The neuronal ordering runs the other way (gap 0.83 → 0.46) but never approaches zero, because at
higher *K* neurons subdivide across more programs, lowering the within-excitatory correlation without
disturbing the excitatory/inhibitory distinction. This is a concrete argument for reporting the
higher-*K* models when the biology of interest is a fine lineage distinction.

Computed by `A7_program_celltype_map.py`; plotted as `FIG2_supp_program_celltype`.

## Atlas models

Models were trained on the 70% training split of the mouse atlas (`train_splice_model_ready_70pct`)
with *K* ∈ {10, 20, 30} and three random initialisations each (nine models). Training used a
minibatch of 8,192 cells, learning rate 0.05 decaying by a factor of 10 over the run, a cap of 5,000
steps with early stopping as above, a Beta-Binomial likelihood with learned concentration, and
junction-specific Beta priors on Ψ. **Initialisation was random; no waypoint or cell-type-informed
initialisation was used anywhere in the pipeline**, so no cell-type label enters model fitting.

Fitted global parameters were consistent across initialisations — for *K* = 10, mean Ψ was 0.3003,
0.3005 and 0.3003 against an observed splicing rate of ≈0.30, and the learned Beta-Binomial
concentration was 1.95, 1.98 and 1.99 — and cells used a median of 2–3 of the available programs.

## Held-out evaluation

The remaining 30% of cells were never seen during training. Applying a trained model to them is a
projection onto the fitted dictionary: every global variable (Ψ, π, α, the Dirichlet concentration
δ and the Beta-Binomial concentration) is held fixed and only the per-cell loadings φ are inferred.
Freezing Ψ alone would not be sufficient, because π and δ together form the Dirichlet prior on φ;
allowing them to be re-estimated on the held-out cells would infer those loadings under a different
prior from the one the training cells were fitted under, confounding any train-versus-held-out
comparison. This E-step uses the same implementation as training. With Ψ fixed the problem
decomposes into independent *K*-dimensional per-cell problems, and it is correspondingly stable: on
simulated data with a known dictionary the inferred loadings recovered the true loadings with mean
per-program correlation 0.997 and were reproducible across random seeds (mean absolute difference
0.004), with the frozen globals verified to be absent from the variational family.

The E-step was previously carried out by a separate, older implementation of the same model. We
unified it with the training code so that the pipeline contains a single inference implementation;
the change is a consistency measure and not a correction. Running both implementations on the same
trained models gave held-out structure metrics differing by 0.005–0.02 with no consistent direction
(for example, adjusted Rand index against broad cell type 0.203 versus 0.209), i.e. within run-to-run
variation. This is expected: the optimisation difficulties that affected joint training do not arise
once Ψ is fixed, because the per-cell problems are then independent and near-convex.

**Imputation.** Read counts in the held-out cells were thinned to 10, 25, 50, 75, 90, 99 and 100% of
their original depth. Loadings were inferred from the thinned counts under the frozen dictionary and
the reconstruction φΨ scored against the *original* full-depth junction usage, separately for entries
lost to thinning and for all covered entries. Accuracy is summarised as the median per-junction
Pearson correlation across cells, and calibration as the slope of a regression of observed on
predicted usage (1 indicates calibration; values above 1 indicate predictions compressed toward the
mean). Baselines are the junction mean, the cell-type-stratified junction mean (which uses cell-type
labels and is therefore an upper bound rather than a fair comparator), and NMF fitted on the
training cells and projected onto the held-out cells under both zero- and mean-imputation.

**Latent structure.** Held-out loadings were clustered with Leiden and compared to annotated cell
types by adjusted Rand index, cluster purity and silhouette coefficient, alongside the same NMF
projections. Cell-type labels are used only for evaluation, never for fitting.
