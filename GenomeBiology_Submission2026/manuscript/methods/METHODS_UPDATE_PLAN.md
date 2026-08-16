# Methods update plan — patch, don't rewrite

`methods_PREVIOUS_SUBMISSION.tex` is frozen. `methods_CURRENT.tex` starts as a copy and every step
below is a bounded edit to a named subsection. Rule: touch a subsection only where the shipped model
or a shipped analysis differs from what the text says. Data processing, ATSE mapping, gene
expression, aging regression, RBP mediation and conservation are UNCHANGED and are not touched.

Source of truth for each edit is a METHODS_*.md note in the LeafletFA repo, cited per step, so the
tex is a condensation of measured, committed material -- not new prose.

| step | subsection | action | source | size |
|---|---|---|---|---|
| **1** | *LeafletFA Beta-Dirichlet Factor Model* → rename **LeafletFA Dirichlet-Multinomial Factor Model** | REPLACE the psi prior + likelihood blocks. psi becomes a within-ATSE simplex via softmax of logits, `psi_logit ~ N(0,1)`; likelihood becomes DM over the ATSE's junction vector with `alpha = conc_e * rho`; per-ATSE `conc_e ~ Gamma(2, 0.1)` clamped at 2.0. Keep phi hierarchy, but state `delta` PINNED at K (learned delta drove `delta*pi_k` to 0.19-0.37, the corner-seeking regime). Keep the KEY equation `rho = phi @ psi` -- it is unchanged | `METHODS_dirichlet_multinomial.md` §1-3 | ~35 lines out, ~40 in |
| **2** | *Table: LeafletFA Model Parameters* | REWRITE rows: drop a, b, a_j, b_j (no Beta prior); `psi_kj` → simplex row; add `conc_e`; `y_c,e` → DM; `T_cj` → `N_ce`; note delta pinned | same | table only |
| **3** | NEW paragraph inside step-1 subsection: **Concentration as effective sample size** | ADD ~8 lines: A_e = conc_e is the Dirichlet's total pseudo-count; Smart-seq2 reads are not independent (68% of (cell,ATSE) pairs all-on-one junction at median 48 reads; solving P(all-on-one) gives 1.7-3.1 molecules); free fit collapses to 0.52 (< 1 molecule); floor 2.0 sits inside the measured range | `METHODS_dirichlet_multinomial.md` §"effective sample size" | +8 |
| **4** | *Multi-pass Mini-batch Training* → **Mini-batch Stochastic Variational Inference** | REPLACE. Single Pyro subsampling plate; local phi per minibatch, globals shared; 16,200 steps, batch 8,192, lr 0.025, gamma 0.1 (`num_steps` sets the schedule); NaN-guard checkpoint/rollback; convergence by `psi_contrast`, NOT ELBO (100% of ELBO drop in first 50 steps). Drop "10 passes / 5,000 / 0.95 decay" entirely | `METHODS_MINIBATCH.md` | ~15 out, ~15 in |
| **5** | *Model Fitting and Initialization Using Waypoints* | DELETE waypoint content; keep a 3-line SVI paragraph. Plate fix removed the need for waypoints at K<=20 | `project_leafletfa_convergence` / METHODS_MINIBATCH | −25 |
| **6** | *Bayesian Factor-Specific Differential Splicing* → **Differential splicing between cell groups** | REPLACE the statistic. Group contrast: `Delta_j = dphi . psi_:,j` with dphi = group-mean phi difference DRAWN from q(phi); `TV_e = 0.5 sum_j |Delta_j|` per posterior draw; `P_e = P(TV_e > delta)`; Bayesian FDR unchanged in form (keep those equations). State delta = 0.15 and how derived (step 8). Drop the factor-vs-rest junction test and delta = 0.1 | `METHODS_differential_splicing.md` | ~30 out, ~30 in |
| **7** | NEW subsection: **Dirichlet-multinomial likelihood-ratio test** | ADD ~15 lines: per-ATSE DM GLM, H0 shared composition vs H1 composition varies with covariate; chi-square with (m-1)*K df; covariate = cell-type indicator for the LeafCutter head-to-head. Same likelihood as LeafCutter's `dm_glm.stan` (verified term-for-term) with single cells as the unit | `project_leafletfa_ds_test_lrt` + A45 docstring | +15 |
| **8** | NEW subsection: **Choosing delta: permuted negative-control ATSEs** | ADD ~15 lines: permute count vectors of 10% of ATSEs across cells keeping N_e attached; refit; those events null by construction while the rest pin dphi; delta = p99 of their TV; validated on sims (permuted vs true-null ratio 0.93-1.02); atlas floor max 0.13 → delta 0.15 | A64 / `_permute_null.py` docstrings | +15 |
| **9** | *Simulating Junction Usage Ratios* | PATCH, not rewrite: heart not brain; emission is Dirichlet-Multinomial (`q ~ Dir(conc*p)`, `y ~ Mult(N_e, q)`) so `sum_j y_j = N_e` exactly; effect parameter `psi_k = (1-e) psi0 + e psi_k^indep`; concentration 50 (main) and 2 (matched to measured); NMF baseline at its best gauge. Keep the J1/J2/J3 construction sentence -- unchanged | `00_simulate_splicing.py` docstring | ~6 changed lines |
| **10** | NEW subsection: **Held-out imputation** | ADD ~12 lines: 70/30 split (92,894 / 39,812), model refit on 70% only, downsampling of held-out reads to 10-100%, phi inferred by E-step with psi/pi/conc FROZEN, imputed value = DM posterior mean `(alpha_j + y_j)/(A_e + N_e)`, baselines (NMF at rank K, per-junction mean, cell-type mean as an upper bar that uses labels) | `A3_eval_imputation.py` docstring; `IMPUTATION_EVAL_V2_PLAN.md` | +12 |
| **11** | NEW subsection: **Comparison with LeafCutter** | ADD ~12 lines: pseudobulk 7-vs-8 per contrast; each method thresholded at its OWN realised FDP 0.05 (LeafCutter q<=1.39e-3, LeafletFA q<=3.16e-2); the null-asymmetry caveat (LeafCutter's clean, ours contaminated → upper bound) | A50/A52 docstrings | +12 |
| **12** | NEW short subsection: **Program reproducibility** | ADD ~8 lines: Hungarian match on |corr(psi)| across seeds and across K; `r >= 0.9` a stated cutoff (distribution is a smooth gradient, not bimodal); consensus DS set = called in >=2 of 3 seeds | A61 docstring | +8 |

**Order:** 1-3 first (they define the symbols everything else uses), then 4-5, then 6-8, then 9-12.
Steps 1-8 are ~1 hour of condensation. Steps 9-12 depend on numbers still landing (imputation
sweep, conc=2 arm) -- write them with `\textbf{XX}` placeholders and fill on landing.

**Do NOT add:** the m/(m-1) tempered arm (an ablation, supplement text only); the `--design phi`
LRT; any collapsed-phi DS number; A53's "0.76%".
