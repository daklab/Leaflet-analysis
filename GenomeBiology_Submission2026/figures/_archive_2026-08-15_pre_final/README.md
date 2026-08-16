# Archived 2026-08-15 — pre-final figure material

Everything here was superseded by the shipped model (Dirichlet-multinomial + within-ATSE simplex,
per-ATSE concentration floored at 2.0, delta_dir = K, no likelihood weight, K=20). It is kept because
several of these figures are referenced by dated notes and reports; none of them should be
regenerated or placed in the paper.

| moved | why it is out |
|---|---|
| `DM_TEST/` | the DM-vs-BetaBinomial factorial exploration. Its purpose was to CHOOSE the model; the model is chosen. `Q6_overdispersion` and `Q5` data still feed FIG2 panels b/c from `DM_FRESH/*.tsv`, not from these plots |
| `FIGURE2_.../FIG2_simulation`, `FIG2_validation` | validate the BetaBinomial and Binomial models. Neither ships. Replaced by `FIG2_final` |
| `FIGURE2_.../FIG2_imputation`, `FIG2_EF_imputation` | scored on models trained on ALL cells (100% of the test split was in training). Replaced once the 70/30 A3 sweep lands |
| `FIGURE2_.../FIG2_supp_*` (draft, heldout_structure, seed_consistency_sim, cellcell_consistency) | BetaBinomial-era supplements; the seed/consistency claims they made were superseded by A60/A61 |
| `FIGURE2_.../code/{build_sim,build_imputation,panels,prep_panels,sim_benchmark,ablate_trainer,collect_inputs,refresh_nmf*,step*,eff*,od*,pr*}` | the BetaBinomial figure pipeline. `build_fig2_final.py`, `bench_dm_vs_nmf.py`, `check_nmf_scale_fairness.py`, `build_regime_phi_panels.py` stay |
| `FIGURE3_leafcutter_ds/FIG3_supp_Fbln2_program_decomposition_*` except `_FINAL_K20_seed0` | per-configuration Fbln2 panels from the recipe search (conc_min2/5, global_conc, PROD2, cmin2_16k_noHO, DMsimplex, seed1). One survives as the worked example |
| `FIGURE3_leafcutter_ds/FIG3_supp_atlas_phi_umap*` except `_FINAL` | UMAPs of superseded fits |
| `FIGURE3_leafcutter_ds/FIG3_calibration_draft.*` | drawn from the collapsed-phi model |
| `FIGURE3_calibration/` | a single Fbln2 reference panel from the released BetaBinomial |
| `MORNING_REPORT_*`, `RUN_LOG_*`, `INIT_COMPARISON_REPORT`, `SEED_CONSISTENCY_APPROACHES`, `METHODS_simulation_2026-07-26`, `FIGURE2_CAPTION`, `REPRODUCE_FIGURE2` | dated working notes describing the pre-DM pipeline |
