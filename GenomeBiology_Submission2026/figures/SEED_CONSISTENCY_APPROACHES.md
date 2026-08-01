# Getting one consistent model out of a stochastic fit

> **CLOSED — 2026-07-30.** No initialisation tried beat the Pyro default on the atlas, which is
> the arm that decides (simulation mispredicted the atlas twice). Waypoints and jitter both work
> mechanically at scale and neither improved cross-seed reproducibility. The variant branches were
> removed from `A1`, and the 70/30 models were retrained from the default path as
> `2026-07-30_atlas9_default`. Seed variation is now **reported, not engineered away** — see
> `FIGURE_OUTLINE_ALL.md` § "Cross-cutting: initialisation and reproducibility".
>
> Kept for the evidence, and because a reviewer will ask what was tried. Companion document: `INIT_COMPARISON_REPORT.md` (the measured results).

**Question.** Independently initialised LeafletFA fits agree on Ψ (*r* ≈ 0.97) and π (*r* ≈ 0.93) but
only on ~30–50% of φ directions. We want one released model, and to be able to say the structure is
learned consistently — without waypoint initialisation, which reads cell-type labels and so undercuts
the claim of finding structure *without* prior annotations.

This documents what we measured, what the standard approaches are, and what I recommend.

---

## 1. First, what kind of problem this is

Three different things get called "seed instability", and they have different fixes. Separating them
is most of the work:

| | what it is | fix |
|---|---|---|
| **Label switching** | fit A's program 7 is fit B's program 22 | alignment (Hungarian). Not a real problem. |
| **Multimodality / poor symmetry-breaking** | fits leave a symmetric start in different directions | data-driven init, restarts, annealing |
| **Posterior degeneracy** | the data does not determine some directions at all | **no fix exists** — only honest reporting, or averaging |

**We measured which one we have, and it is the third.** Evidence:

- Simulating the identical design at *c* = ∞ / 20 / 2 gives **100% / 83% / 42%** reproducible
  directions (as a fraction of the *K* − 1 that exist — φ is on the simplex, so *K* programs span only
  *K* − 1 dimensions and the *K*-th canonical correlation is numerically arbitrary),
  and **recovery against known φ degrades in step** (0.990 → 0.959 → 0.788). Fits do not merely
  disagree with each other; at high dispersion they are all further from the truth. (`A13`)
- The learned `bb_conc` ≈ 1.97 means Beta-Binomial precision saturates at *c*+1 ≈ 3, so **each
  junction yields ~3 reads' worth of information regardless of sequencing depth**. That predicts the
  otherwise puzzling observation that per-cell agreement is flat across a 450-fold depth range.
- Two candidate mechanical causes were tested and **refuted**: doubling the training budget
  (1,334 → 2,584 steps) and dropping the junction-specific prior each moved effective K by ±1.0
  against a within-arm spread of 2. The prior arm is the stronger result — it *did* reduce program
  correlation (+0.876 → +0.807) and reproducibility did **not** improve. (`A11`)

**But that is only half the story, and an earlier version of this document stopped there and was
wrong to.** Overdispersion sets a *ceiling*; it does not follow that the current setup reaches it. It
does not. The default initialisation puts every program at exactly Ψ = 0.5 and every cell at φ = 1/K
(§3), which is a perfectly symmetric point that only gradient noise can break — and at high
overdispersion there is barely any signal to break it with. Supplying a data-driven starting point
raises accuracy against KNOWN φ from 0.699 to 0.847 at atlas-like overdispersion (`A15`).

So the honest statement has two parts:

- the missing directions cannot be recovered by any modelling change, because the information is not
  in the data; **and**
- we were nonetheless well short of that ceiling, for a reason that *is* fixable.

---

## 2. What the field does

Standard approaches, and whether each applies here.

**(a) Multiple restarts, keep the best ELBO.** The default in variational inference, and what our
Methods currently claims. **It does not do what we want**: on our models the best-ELBO seed disagreed
with the most central seed at K = 20 and K = 30. ELBO selects for fit quality, not for
representativeness, and in a degenerate posterior those come apart. Keep restarts; change the
selection rule.

**(b) Consensus / ensembling over replicates.** Run many fits, align them, and combine. This is
exactly what **cNMF** (Kotliar et al., *eLife* 2019) does for single-cell NMF: many replicates, pool
all components, filter outliers by local density, cluster into K groups, take the median of each. It
is the closest precedent to our situation and it is well accepted in this field. The theoretical
argument is just the posterior mean: if each fit is `shared + independent error`, averaging N of them
shrinks the error by √N. **Measured on our atlas, averaging two aligned fits already predicts a
held-out third better than either alone, at every K** (+0.010, +0.022, +0.016).

**(c) Data-driven initialisation.** NMF's standard NNDSVD (Boutsidis & Gallopoulos, 2008; sklearn's
default) derives a starting point from the data matrix itself. It is also fully deterministic, which
is *why* NMF looked so reproducible in our earlier clustering comparison — worth remembering, since it
means that comparison was not measuring what it appeared to.

The useful property for us is **not** the determinism (our baseline is already deterministic — §3) but
that it starts the K programs APART and near the data. Our default starts them identical and at
Ψ = 0.5. Measured effect at atlas-like overdispersion: effective K 1.2 → 3.0 of the 4 achievable
(30% → 75%), consistency +0.187, and accuracy against known φ +0.148. Uses only the splicing matrix, no labels.

**(d) Label-switching corrections.** A mature literature for Bayesian mixtures (e.g. Stephens, *JRSS-B*
2000) on relabelling draws to a common frame. Our Hungarian alignment is the simple version. Necessary
for any of this, but it only fixes permutation, not degeneracy.

**(e) Annealing / KL warm-up.** Down-weighting the KL term early (Bowman et al. 2016; Sønderby et al.
2016) helps escape poor optima in VAEs. Worth trying only if multimodality were our problem. It is not.

**(f) Identifiability constraints** — anchor features, orthogonality or sparsity penalties, archetypal
analysis. These make a solution unique by adding assumptions. **Our centred programs are already
maximally decorrelated** (correlations exactly −1/(K−1)), so an orthogonality penalty has nothing to
act on; and an anchor assumption we cannot verify would be buying uniqueness with an unjustified prior.

**(g) Fix the seed and report one run.** Common, and honest only if paired with a reproducibility
analysis. On its own it hides the issue rather than addressing it.

---

## 3. What "random initialisation" actually was — a correction

Measured, not assumed: LeafletFA's default is Pyro's `init_to_feasible`, which maps **zeros** through
each site's constraint transform, ignoring both the data and the distribution's parameters. The result:

| | initial value | sd across entries | varies by seed? |
|---|---|---|---|
| Ψ | **exactly 0.5** everywhere | 0.00000 | **no** |
| φ | **exactly 1/K** everywhere | 0.00000 | **no** |

So the baseline was never "random". It is a fixed, perfectly SYMMETRIC point, identical for every seed,
and the seed-to-seed variation comes entirely from minibatch ordering and stochastic gradients.

Two consequences, and they sharpen the mechanism:

1. Ψ starts at 0.5 while the data's mean is ~0.30 (atlas) / ~0.33 (simulation) — systematically
   mis-centred.
2. **Every program starts identical and every cell starts uniform**, so the programs are exactly
   symmetric at initialisation and symmetry can only be broken by gradient noise. At high
   overdispersion the likelihood signal that would break it is weak (~3 effective reads per junction),
   so symmetry breaks slowly and arbitrarily -- which is why seeds diverge more as c falls.

This also RETRACTS a concern recorded in an earlier version of this document: that a deterministic init
would deliver reproducibility "by construction" rather than identifiability, and so be circular. That
rested on the false premise that the baseline was random. Both initialisers are equally deterministic;
NNDSVD simply supplies a better, symmetry-broken starting point. The consistency gain is therefore a
genuine improvement, and the accuracy gain against known phi (+0.148, where the initialiser alone
scores only 0.32) confirms it is not a trick.

There is also a failure mode to guard against: **an init that makes every run agree on the same wrong
answer.** Our test therefore reports recovery against known φ alongside consistency, so we can tell
"agreeing and right" from "agreeing and wrong". A first attempt at the initialiser illustrates the
danger — reweighting by soft NNDSVD weights produced programs correlated at +0.991 (essentially
identical to one another), which would have started every program on top of every other. Hard
assignment gives contrast 0.111 and correlation +0.683, against fitted values of ~0.08 and ~+0.87.

---

## 4. Recommendation

**Goal (set by the user, 2026-07-28): ONE released model, with evidence that the programs are
consistent -- not an ensemble.** Consensus was an earlier suggestion of mine and it does add real
machinery: aligning latents, maintaining N fits, and explaining a merged model in the paper. If the
initialisation fix works, none of that is needed.

**Primary: fix the initialisation, then use seeds as VALIDATION rather than as an ensemble.**

Train three times from independent initialisations and report how well the programs agree. That is
what a reader wants to see anyway -- "we ran it three times and recovered the same programs" -- and it
ships a single model.

The fix targets a specific, measured defect. Pyro's default sets every variational MEAN identically
(psi = 0.5, phi = 1/K, spread across programs exactly 0.000000). Sampling still happens at every SVI
step, but the per-draw spread across programs is only ~0.021 against the ~0.08 contrast a fitted model
reaches -- so the model must amplify a small random fluctuation roughly fourfold, and at c ~ 2 the
likelihood is too weak to overrule which fluctuation it amplifies. Different seeds amplify different
noise. Breaking that symmetry at initialisation is the intervention.

Which form of symmetry-breaking is pending `A17`:

- if a fixed, information-free JITTER recovers most of the benefit, use that -- it breaks symmetry for
  all K programs by construction, needs no SVD, and has no scale-dependent failure;
- if the DATA content matters, a data-driven initialiser is needed that survives high K. NNDSVD hard
  assignment does NOT: at K=30 on the atlas it left 22 of 30 groups below the size floor, so those
  programs all started at the global mean -- identical, i.e. the very symmetry it was meant to break.
  A balanced assignment (top-N cells per component) or unstratified max-min sampling would be the
  repair.

**Secondary: report effective K alongside nominal K.** Even with a good init, simulation at c = 2
reached 75% of the K-1 achievable directions, not 100%. The overdispersion ceiling is real, so a single fit remains a
partly-arbitrary draw and should be reported as such. That is a reporting obligation, not a reason to
ship an ensemble.

**Fallback, only if agreement is too low to defend one fit: consensus over replicates** (`A14`).
Measured to work -- effective K 9.2 -> 12.2 from 1 to 9 seeds on the atlas -- but it is the more
complicated product and should not be adopted unless needed.

**Also:** if a single fit must be chosen from several, use the medoid, not best ELBO (they disagreed at
K = 20 and K = 30).

**Do not:** train longer (tested twice, null), reinstate waypoints (uses labels), or add a
decorrelation penalty (centred programs are already at the -1/(K-1) floor).

---

## 5. Status

| analysis | script | result |
|---|---|---|
| Effective K on the atlas | `A10_effective_K.py` | 4.7/10, 10.0/20, 12.3/30 (70/30 models) |
| Is it convergence or the prior? | `A11_seed_stability.py` | **both refuted** |
| Is it overdispersion? | `A13_overdispersion_vs_reproducibility.py` | **yes** — 80%/67%/33% at c=∞/20/2 |
| Consensus model | `A14_consensus_subspace.py` | merging beats any single fit at every K |
| Init vs minibatch variance, in simulation | `A15_seed_consistency_sim.py` | **running** (job 19347469) |
| Converged full-atlas K=30, 10 seeds | — | **running** (19347421, 19347436) |

`A15` is the one that decides whether deterministic init is worth adopting: it splits the seed
variance into initialisation and minibatch components at three dispersion levels, and reports accuracy
against known truth at each so a spurious "everyone agrees" cannot pass as success. It also produces
the tidy CSV for the supplementary figure on φ consistency across seeds in simulation.

---

## 7. Does deterministic init change the overdispersion-arm conclusions?

The main Fig 2 sweeps are at bb-conc=inf where init makes no difference (A15: 0.989 vs 0.988),
so they are untouched. The overdispersion arm sits in the affected regime and is re-run here.
```

--- random init ---
                                   psi_r    sil    ari  auroc  spearman_delta
sim_bb_conc likelihood junc_prior                                            
2.0         bb         False       0.805  0.284  0.016  0.803           0.217
                       True        0.915  0.310  0.040  0.837           0.248
            binom      False       0.921  0.017  0.108  0.703           0.066
                       True        0.923  0.017  0.105  0.704           0.068
20.0        bb         False       0.933  0.641  0.306  0.911           0.391
                       True        0.967  0.663  0.381  0.943           0.448
            binom      False       0.972  0.527  0.860  0.872           0.374
                       True        0.973  0.513  0.852  0.882           0.390

--- deterministic init ---
                                   psi_r    sil    ari  auroc  spearman_delta
sim_bb_conc likelihood junc_prior                                            
2.0         bb         False       0.789  0.345  0.000  0.788           0.184
                       True        0.906  0.372  0.003  0.832           0.222
            binom      False       0.918  0.026  0.143  0.702           0.078
                       True        0.921  0.026  0.139  0.703           0.081
20.0        bb         False       0.952  0.668  0.122  0.893           0.317
                       True        0.968  0.694  0.242  0.938           0.405
            binom      False       0.972  0.522  0.857  0.871           0.357
                       True        0.973  0.522  0.853  0.880           0.385
```
