# Morning report — 2026-07-30

The initialisation investigation is closed: nothing beat Pyro's default on the atlas, so the
variant branches were removed from A1 rather than left switched off. This report checks that
the cleanup left the published numbers intact, and retrains the 70% atlas from scratch.

## 1. Does the cleaned code reproduce the published simulation numbers?
```
8 matched configs

             psi_r: max |diff| 0.0000   mean 0.0000
               sil: max |diff| 0.0000   mean 0.0000
               ari: max |diff| 0.0000   mean 0.0000
             auroc: max |diff| 0.0000   mean 0.0000
    spearman_delta: max |diff| 0.0000   mean 0.0000

  REPRODUCED — cleanup changed nothing  (largest deviation 0.0000)
```

## 2. Retrained 70% atlas — health of all 9 models
```
=== 1. is each fit healthy? ===
  K  seed  meanPsi  contrast  bb_conc  perplex   steps  stopped  progCorr
 10     0   0.3004    0.0789     1.94     2.24    2201    early    +0.867
 10     1   0.3006    0.0751     1.93     2.18    2901    early    +0.876
 10     2   0.3001    0.0744     1.95     2.13    2651    early    +0.876
 20     0   0.2995    0.0829     2.00     2.84    2201    early    +0.874
 20     1   0.2997    0.0798     1.94     2.38    2502    early    +0.879
 20     2   0.2983    0.0825     1.92     2.57    2651    early    +0.878
 30     0   0.2988    0.0803     1.96     3.28    1865    early    +0.885
 30     1   0.3007    0.0782     1.95     2.62    2292    early    +0.892
 30     2   0.2990    0.0774     1.95     2.65    2119    early    +0.895
  -> 0 model(s) flagged; all fits look healthy

=== 2. reproducibility per K ===
  K=10: effective K = 4.7/10 (47%)   program correlation +0.873
  K=20: effective K = 9.7/20 (48%)   program correlation +0.877
  K=30: effective K = 12.3/30 (41%)   program correlation +0.890

=== 3. which seed is most representative? (medoid, not best ELBO) ===
  K=10:
      seed 0  centrality 0.674  loss 9.7e+08  <- MEDOID (best ELBO)
      seed 1  centrality 0.658  loss 9.718e+08  
      seed 2  centrality 0.671  loss 9.711e+08  
  K=20:
      seed 0  centrality 0.668  loss 9.704e+08  (best ELBO)
      seed 1  centrality 0.679  loss nan  <- MEDOID
      seed 2  centrality 0.656  loss 9.717e+08  
      NOTE: medoid and best-ELBO disagree -- selecting on loss would pick the less representative fit
  K=30:
      seed 0  centrality 0.645  loss nan  (best ELBO)
      seed 1  centrality 0.661  loss nan  <- MEDOID
      seed 2  centrality 0.649  loss nan  
      NOTE: medoid and best-ELBO disagree -- selecting on loss would pick the less representative fit

=== RECOMMENDATION ===
  K = 30, seed 1  ->  /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-29_atlas9_clean/mb_run_7/fit.npz
  chosen as the medoid (subspace centrality 0.661), not on ELBO; psi_contrast 0.0782, bb_conc 1.95, perplexity 2.62 of 30, stopped at 2292 steps.
  CAVEAT: only ~41% of phi directions reproduce across seeds, so lean downstream conclusions on aggregate cell-type-level structure rather than individual per-cell loadings.

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-29_atlas9_clean/A12_{health,perK,selection}.csv
```

## 3. DS-call reproducibility across seeds (the number that matters)
```
--- K=10 ---
  seed 0: 111,595 calls at P(|beta|>0.1) > 0.9
  seed 1: 109,006 calls at P(|beta|>0.1) > 0.9
  seed 2: 108,714 calls at P(|beta|>0.1) > 0.9
  seed0_vs_seed1: Jaccard 0.550  slope 0.702 (r 0.725)  top-2000 overlap 33.1%
  seed0_vs_seed2: Jaccard 0.579  slope 0.775 (r 0.795)  top-2000 overlap 44.2%
  seed1_vs_seed2: Jaccard 0.546  slope 0.755 (r 0.750)  top-2000 overlap 38.0%

=== summary ===
  calls per fit: [111595, 111595, 109006, 109006, 108714, 108714]
  Jaccard 0.558   effect-size slope 0.744 (1.0 = no attenuation)   top-2000 overlap 38.4%

  Slope, not correlation, is the attenuation test: correlation is scale-invariant and would read ~1.0
  even if one fit shrank every effect systematically.

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-29_atlas9_clean/A23_ds_K10.csv
--- K=20 ---
  seed 0: 185,645 calls at P(|beta|>0.1) > 0.9
  seed 1: 0 calls at P(|beta|>0.1) > 0.9
  seed 2: 171,773 calls at P(|beta|>0.1) > 0.9
  seed0_vs_seed1: Jaccard 0.000  slope nan (r nan)  top-2000 overlap 0.1%
  seed0_vs_seed2: Jaccard 0.494  slope 0.681 (r 0.695)  top-2000 overlap 39.1%
 ** On entry to DLASCL parameter number  4 had an illegal value
 ** On entry to DLASCL parameter number  4 had an illegal value
 ** On entry to DLASCL parameter number  4 had an illegal value
 ** On entry to DLASCL parameter number  4 had an illegal value
 ** On entry to DLASCL parameter number  5 had an illegal value
 ** On entry to DLASCL parameter number  4 had an illegal value
Traceback (most recent call last):
  File "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/A23_ds_reproducibility.py", line 131, in <module>
    main()
  File "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/A23_ds_reproducibility.py", line 102, in main
    slope = float(np.polyfit(Ba[m], Bb[m], 1)[0]) if m.sum() > 10 else np.nan
  File "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/lib/python3.10/site-packages/numpy/lib/polynomial.py", line 669, in polyfit
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
  File "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/lib/python3.10/site-packages/numpy/linalg/linalg.py", line 2326, in lstsq
    x, resids, rank, s = gufunc(a, b, rcond, signature=signature, extobj=extobj)
  File "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/lib/python3.10/site-packages/numpy/linalg/linalg.py", line 124, in _raise_linalgerror_lstsq
    raise LinAlgError("SVD did not converge in Linear Least Squares")
numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
--- K=30 ---
  seed 0: 0 calls at P(|beta|>0.1) > 0.9
  seed 1: 0 calls at P(|beta|>0.1) > 0.9
  seed 2: 0 calls at P(|beta|>0.1) > 0.9
  seed0_vs_seed1: Jaccard 0.000  slope nan (r nan)  top-2000 overlap 100.0%
  seed0_vs_seed2: Jaccard 0.000  slope nan (r nan)  top-2000 overlap 100.0%
  seed1_vs_seed2: Jaccard 0.000  slope nan (r nan)  top-2000 overlap 100.0%

=== summary ===
  calls per fit: [0, 0, 0, 0, 0, 0]
  Jaccard 0.000   effect-size slope nan (1.0 = no attenuation)   top-2000 overlap 100.0%

  Slope, not correlation, is the attenuation test: correlation is scale-invariant and would read ~1.0
  even if one fit shrank every effect systematically.

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-29_atlas9_clean/A23_ds_K30.csv
```
