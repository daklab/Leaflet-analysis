# Archived 2026-08-04 evening

Superseded, not wrong. Kept because the underlying fits are still on disk and these are the
record of what was looked at before the sweep.

- `simqc_*`  data QC for the first two DM simulations (effect 0.25 and 0.5, seed 0 only).
  Still valid for those two, but the sweep now covers effect {0.1, 0.25, 0.5, 1.0} x 3 seeds.
- `model_examples_*`, `model_population_*`  model-vs-truth views built from the ORIGINAL
  factorial fits, which varied only the MODEL seed on a single simulation (sim seed 0). The
  evening's panels use the sweep, where the simulation itself is regenerated per seed.

Superseded by, in `../panels/`:
  `truth_data_model_*`        adds the raw-data layer between truth and model
  `seed_consistency_*`        same ATSEs across independent simulations, programs Hungarian-matched
  `fig2style_recovery_sweep`  recovery over the four-point effect axis

## Second archive pass (same evening)

`truth_data_model_*` moved here too. Not wrong, but built from a single model seed on a single
simulation, before the model-seed grid was completed across all four effect sizes. The regenerated
set in `../panels/` covers effect {0.1, 0.25, 0.5, 1.0} with three model seeds each.

Also archived earlier in this pass: `seed_consistency_*` WITHOUT a `_model`/`_sim` tag in the
filename. Those used independent simulations per column, which is incoherent for a per-event
comparison -- the same event_id has different true psi in each simulation, so a designed-null
event in one column can be genuinely differential in another. Replaced by
`seed_consistency_model_*`, which holds one simulation fixed and varies only the training seed.
