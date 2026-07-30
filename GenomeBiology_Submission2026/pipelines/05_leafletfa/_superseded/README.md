# Superseded — kept for provenance, not for use

These drove the atlas before the Pyro-plate rewrite. They are retained because published numbers were
once produced with them, not because they should be run again.

| file | replaced by | why |
|---|---|---|
| `run_training.py` | `A1_train_atlas.py` | rebuilt a fresh `LeafletFA` and guide on every 5,000-cell chunk (~186 times for the atlas), warm-starting only Psi's mean. `A1` keeps ONE model, guide and optimiser alive under `pyro.plate(subsample_size=B)`, so Pyro scales the local terms by N/B automatically and the Psi posterior reflects all cells rather than one chunk. |
| `01_generate_params.py`, `02_submit_jobs.py` | `parameter_combinations.json` + `train.sh` | the grid is now a JSON file read directly by `A1`. |
| `03_collect_results.py`, `08_collect_evaluation.py`, `aggregate_eval_by_config.py` | the eval sbatch writes its own CSVs | separate collection steps are no longer needed. |
| `04_evaluate_imputation.py`, `05_submit_evaluation.py` | `A3_eval_imputation.py` | `A3` uses the frozen-globals E-step and the cached NMF baselines. |
| `06_consistency_check.py` | `A6`, `A10`, `A24` | the current suite reports every latent with matched AND matching-free measures. |
| `09_retrain_best_on_full.py`, `10_submit_full_training.py` | `LEAFLETFA_MODELS/2026-07-*_full_*` runs | full-atlas training is now a plain `A1` invocation on the full h5ad. |

Still live and NOT superseded: `A1`-`A14`, `A19`, `A23`-`A25`, plus `estep_minibatch.py`,
`mean_bayes.py` and `nmf_transfer.py` (imported by A3/A4/A5/A7).

`08_compare_batch_equivalence.py`, `07*_classify_splicing_programs`, `00b_visualize_psi_pca.py` and
`plot_eval_summary.py` are left in place: the batch-equivalence check is still an open TODO (it needs a
scale-sensitive slope metric added), and the others are utilities with no current replacement.
