# LeafletFA for dummies

A plain-language walk through the model as it ships (2026-08-15). Written so that every symbol in the
Methods has a sentence here saying what it is, why it is there, and what would go wrong without it.
Nothing below is new; it is the model, explained.

---

## 0. The data, and why it is hard

Smart-seq2 gives us, for every cell, how many reads cross each splice junction. Junctions that share
a splice site are grouped into an **ATSE** -- an alternative transcript structure event. A cassette
exon is an ATSE with three junctions: skip, include-5', include-3'. Within one ATSE the junction reads
of a cell are a **composition**: they partition that cell's reads over the event, so they sum to the
event total. That sum-to-one is not a modelling choice; it is how the data is constructed.

Three facts about the data drive every design decision:

1. **It is sparse.** Only 13.8% of (cell, junction) entries carry any reads. Most cells see most
   events not at all.
2. **Reads are not independent.** No UMIs, so a cell's 48 reads over an event are PCR copies of ~2-3
   cDNA molecules. We measured this: 68% of (cell, ATSE) pairs put *every* read on one junction at
   a median of 48 reads, which independent draws would essentially never do. Effective molecules:
   1.7-3.1 per event. (Full numbers in `NOTE_duplicate_reads.md`.)
3. **It is bimodal at the cell level.** Because a cell has ~2 molecules, its observed usage of a
   junction is very often exactly 0 or exactly 1, even on events whose population-level usage is 0.5.
   Look at Gna13 in Figure 3: every cell at 0 or 1, mean 0.4.

A model that treats reads as independent draws (Binomial, or BetaBinomial with a large
concentration) is ~20x overconfident about each cell and gets everything downstream wrong: DS
p-values too small, per-cell posteriors too narrow, and the sparsity prior asked to explain
technical all-or-nothing as biological program commitment.

---

## 1. The idea in one line

**Every cell is a mixture of a small number of splicing programs; every program is a full set of
junction compositions; a cell's expected usage of any junction is its mixture-weighted average of
the programs' usages.**

    rho_cj  =  sum_k  phi_ck * psi_kj

That is the whole model. Everything else is (a) making phi and psi behave, and (b) connecting rho to
noisy, sparse, duplicated counts honestly.

---

## 2. Each latent variable, one at a time

### psi -- what a program IS
`psi_kj` = program k's usage of junction j. **Global**: one value per (program, junction), shared by
every cell. This is the dictionary. [PENDING 2026-10-25: the aging program is under review -- age is confounded with sequencing batch; see AGING_SUPPLEMENT_DRAFTS.md.] When we say "program 5 is an aging program", we mean psi_5 has a
particular pattern of junction usages.

Within each ATSE, psi_k is a composition -- it sums to 1 over the event's junctions. Enforced by
parameterising through a softmax over unconstrained logits eta:

    psi_kj = exp(eta_kj) / sum_{j' in same ATSE} exp(eta_kj')       eta_kj ~ Normal(0, 1)

Why softmax and not independent Betas per junction (the old model): an ATSE with m junctions has
only m-1 free numbers. Independent Betas treat it as m free numbers, so the model can put "usage"
that doesn't add up (we measured 13% of program-ATSE pairs summing to anything from 0.0 to 4.4 under
the old model). Worse, the likelihood then counts m observations where only m-1 exist -- a 2x
over-count at m=2, which is 44% of the atlas. The softmax removes both problems at once.

The Normal(0,1) prior on eta is weak: it says "usages are not astronomically extreme" and nothing
else. It is what makes the whole thing identifiable rather than a source of shrinkage.

### phi -- what a CELL is
`phi_ck` = how much of program k cell c uses. **Local**: one vector per cell, sums to 1 over the K
programs. It is a mixture, not a cluster assignment: a cell can be 0.6 program 3 + 0.3 program 7 +
0.1 the rest.

    phi_c ~ Dirichlet(delta * pi)

`pi` is the global average program usage (also a composition, Dirichlet(1/K) prior -- flat).
`delta` is how tightly cells are pulled toward pi. Both matter more than they look:

- **delta pinned at K.** The Dirichlet parameter for program k is delta*pi_k. When delta was learned
  it fell so that every delta*pi_k was 0.19-0.37. A Dirichlet with all parameters below 1 is
  *sparsity-inducing*: it pushes every cell into a corner of the simplex, i.e. "you are exactly one
  program". With ~2 molecules of evidence per event, which corner is essentially random, so cells
  ended up committed to programs at random and phi carried no cell-type information. Pinning
  delta = K makes the average delta*pi_k = 1 -- flat, no corner-seeking. This is the "phi collapse"
  fix, and it is a modelling decision, not a tuning knob.
- **pi is learned**, and tells you which programs are broadly used. It is also frozen when applying
  the model to new cells (imputation), so held-out cells get phi under the same prior training
  cells had.

phi is the poorly-determined direction of the model. Per cell it is noisy -- necessarily, given ~2
molecules per event -- and averaging cells does not fix it (cross-seed agreement is flat from n=1 to
n=1000; the disagreement is systematic). Programs are interpreted at cell-type level, DS is done
between groups of cells, and phi is never used as a per-cell label.

### gamma_e -- how much to trust ONE cell on ONE event
The concentration. This is the parameter that connects the model to the sparse, duplicated data.

Under the Dirichlet-multinomial (next section), gamma_e is the Dirichlet's total pseudo-count on
event e -- literally *the number of independent draws the model believes a cell contributes to that
event*. So it has a physical meaning: the effective number of molecules.

    gamma_e ~ Gamma(2, 0.1),   floored at 2.0,   one per ATSE

Why one per ATSE and not one global value: events differ in how duplicated / how committed they
are; a single number spanning 0.01 to 4,750 (what we saw) is not one parameter, it is an average of
incompatible things.

Why floored at 2: the measured effective molecule count is 1.7-3.1. Left free, gamma_e collapses to
0.52 -- *fewer than one molecule*, which is not a possible amount of evidence. It was not the model
discovering low overdispersion; it was a free parameter falling into an impossible regime to explain
all-or-nothing counts. The floor sits inside the measured range. `>= 2` is defensible; `= 2.0
exactly` is not (the estimate assumes p ~ 0.5 on the events used to derive it).

What it does in practice: 48 duplicated reads enter the likelihood as ~2 molecules of evidence,
whatever the read count. Overconfidence removed at the source. The composition estimate stays
informative -- a duplicated molecule's composition is still that molecule's composition -- so
duplication removes *precision*, not *signal*.

The Gamma(2, 0.1) prior has mean 20 and is broad; the floor does the work, the prior just keeps the
posterior from wandering to 4,750 on events with too little data to say.

---

## 3. The likelihood -- connecting rho to counts

    y_c^{(e)} | N_ce, rho_c^{(e)}, gamma_e  ~  DirichletMultinomial( N_ce,  gamma_e * rho_c^{(e)} )

Read it as: cell c's counts over event e are a multinomial draw of N_ce reads, from a composition
that is itself uncertain -- drawn from a Dirichlet centred on the model's rho with total pseudo-count
gamma_e. Small gamma_e = the true composition can be far from rho = counts are overdispersed relative
to a plain multinomial. This is exactly the behaviour PCR duplication produces.

Two things that are true and worth knowing:
- At m=2 this IS the BetaBinomial. For m>2 it is its exact multivariate version. So we did not throw
  away the old model; we generalised it to compositions and stopped double-counting.
- It is the same likelihood as LeafCutter's `dm_glm.stan`, verified term for term (-114.362812 vs
  -114.362793 on the same input). Our differential-splicing LRT is literally LeafCutter's test run
  on single cells.

The `alpha_j = gamma_e * rho_cj` is floored at 0.01: the DM gradient goes as 1/alpha, and junctions
the model thinks are ~never used would otherwise blow up. Tested: it moves nothing for any junction
with reads.

---

## 4. Why DM + simplex beats BetaBinomial + independent Betas

| | old (BB + Beta psi) | shipped (DM + simplex psi) |
|---|---|---|
| psi sums to 1 within ATSE | 86.9% of program-ATSE pairs; sums 0.0-4.4 | 100%, max deviation 5e-4 |
| evidence counted per ATSE | m junctions treated as m trials | m-1 free dims, counted once |
| coverage of a 95% interval (pooled, no VI) | 80.7% at m=2 | 94.7% at m=2 |
| effect size TV | picks up pure gauge when psi scale differs across programs | well-defined |
| DS FDP on simulations | 3-4x anti-conservative | at/below nominal from effect 0.30 up |
| training cost | pyro BetaBinomial, ~5 lgamma per entry | 2 lgamma per entry -- 0.21x the time |

The last row surprises people: modelling ATSEs properly is *cheaper*, because the ATSE index is a
flat integer array and grouping is one `index_add`.

---

## 5. Variational inference -- how we fit it

We cannot compute the posterior over (eta, phi, pi, gamma) exactly. Variational inference picks a
simple family of distributions q and finds the member closest to the true posterior by maximising
the ELBO (evidence lower bound). We use Pyro's `AutoNormal`: an independent Normal for every
unconstrained parameter. So the fit gives, for each parameter, a location and a scale -- and
"posterior draws" of psi or phi are just draws from those Normals pushed through softmax (psi) or
stick-breaking (phi) back onto the simplex.

Two consequences that shape everything downstream:
- **The posterior IS the product.** DS, imputation and effect sizes all draw from q. That is why the
  m/(m-1) "tempered likelihood" that improved point estimates of phi was NOT shipped: it would make
  q too narrow by construction and every posterior probability would be wrong.
- **q's widths are honest, and we checked.** On simulated null events, the spread of fitted psi
  across programs is 0.8x the posterior sd. q(psi) is if anything slightly conservative.

Getting phi for a NEW cell (imputation) is the same machinery with the globals frozen: fix eta, pi,
delta, gamma at their trained values, and run VI on that cell's phi alone. That E-step reproduces the
trained phi on training cells at r = 0.974.

---

## 6. Minibatching -- how it scales to 132,706 cells

Cells are conditionally independent given the globals. So the ELBO is a sum over cells, and an
unbiased estimate of it is: draw 8,192 cells, compute their terms, multiply by 132,706 / 8,192. Pyro's
subsampling plate does exactly this. Each step updates the drawn cells' phi AND the globals; the
globals see every cell over the run.

    16,200 steps x 8,192 cells   lr 0.025 -> 0.0025 (exponential decay)   Adam, clip 10

Things learned the hard way, all in the code now:
- The old scheme (10 passes, extract-and-reinitialise globals between minibatches) was not SVI; it
  was sequential refitting, and it needed waypoint initialisation to be reproducible. The single
  plate does not.
- Every atlas fit before this schedule hit a NaN. Now a non-finite loss rolls back to the last
  checkpoint instead of killing the run.
- **The ELBO cannot tell you when you're done.** 100% of its drop lands in the first ~50 steps and
  the tail slope flips sign between fits. What still moves for thousands of steps is the
  between-program contrast of psi -- that is the convergence criterion.
- Batch size sets a recovery ceiling; use the largest that fits.

---

## 7. What the model can and cannot say, and why

- **Programs (psi) are reproducible.** Matched across seeds on psi correlation, 11-13 of 20 come back
  at r >= 0.9, and 14 of 20 survive changing K to 30. Matching is on psi and never phi.
- **Per-cell phi is not a label.** ~2 molecules per event caps per-cell resolution. Group-level phi
  (cell type means) is stable; per-cell phi is noisy in a systematic, not averaging-out, way.
- **DS is between groups, at the ATSE level, with a stated minimum effect.** TV = 0.5 * sum_j
  |dphi . psi_j|, and we call P(TV > delta) with delta = 0.15 -- derived, not chosen: it is the 99th
  percentile of TV on permuted negative-control ATSEs (null by construction, loadings intact),
  validated on simulations where the true nulls are known. Below delta ~0.10 the test cannot see,
  because psi's own posterior sd is 0.02-0.09 per junction.
- **The posterior DS test needs a coverage gate.** TV = dphi.psi never looks at the event's own
  counts, so on an event 14 cells cover it reports the imputation prior. Gate at >= 25 covered cells
  per arm. Found by drawing the example (Usp54) and looking.
- **The LRT is the LeafCutter head-to-head, the posterior test is the model's claim.** The LRT uses
  phi only as a covariate and refits compositions from raw counts; it is better calibrated than
  LeafCutter at every event size (1.15-1.93x vs 3.2-11.4x). The posterior test is the only arm that
  uses the denoised psi -- and it is conservative and precise rather than sensitive.

---

## 8. Symbol table, for reference

| symbol | what | global/local | prior | shipped setting |
|---|---|---|---|---|
| eta_kj | logit of program k's usage of junction j | global | Normal(0,1) | learned |
| psi_kj | softmax(eta) within ATSE | global | (derived) | learned |
| pi | average program usage | global | Dirichlet(1/K) | learned |
| delta | pull of cells toward pi | global | -- | **pinned at K** |
| phi_c | cell c's program mixture | local | Dirichlet(delta pi) | learned per cell |
| gamma_e | effective molecules per event | global | Gamma(2, 0.1) | learned, **floor 2** |
| rho_cj | sum_k phi_ck psi_kj | derived | -- | -- |
| y_c^(e) | junction counts of cell c in ATSE e | data | DirMult(N_ce, gamma_e rho) | -- |
| K | number of programs | -- | -- | 20 (10, 30 robustness) |
| delta (DS) | minimum effect to call | -- | -- | 0.15, derived |
