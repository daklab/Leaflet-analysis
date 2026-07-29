# Morning report — overnight of 2026-07-28

Question: does a deterministic **jitter** initialisation make independently seeded fits agree,
and are the agreeing fits actually *right*? NNDSVD was dropped — its cell-to-component assignment
collapses at K=30 (22 of 30 groups below any usable size), reintroducing the symmetry it was
meant to break. Jitter has no clustering step and is stable from K=5 to K=50.

## Job outcomes
```
             JobID     JobName       State    Elapsed 
------------------ ----------- ----------- ---------- 
          19353984    A16_init   COMPLETED   01:50:30 
          19354042     A17_sym   COMPLETED   01:14:12 
        19354144_0     sim_jit   COMPLETED   00:10:59 
        19354144_1     sim_jit   COMPLETED   00:13:07 
        19354144_2     sim_jit   COMPLETED   00:05:58 
        19354144_3     sim_jit   COMPLETED   00:10:21 
        19354144_4     sim_jit   COMPLETED   00:08:39 
        19354144_5     sim_jit   COMPLETED   00:05:44 
        19354144_6     sim_jit   COMPLETED   00:10:22 
        19354144_7     sim_jit   COMPLETED   00:10:17 
        19354144_8     sim_jit   COMPLETED   00:10:06 
        19354144_9     sim_jit   COMPLETED   00:10:13 
       19354144_10     sim_jit   COMPLETED   00:08:53 
       19354144_11     sim_jit   COMPLETED   00:10:06 
       19354144_12     sim_jit   COMPLETED   00:10:51 
       19354144_13     sim_jit   COMPLETED   00:06:20 
       19354144_14     sim_jit   COMPLETED   00:07:37 
       19354144_15     sim_jit   COMPLETED   00:10:14 
       19354144_16     sim_jit   COMPLETED   00:08:45 
       19354144_17     sim_jit   COMPLETED   00:08:45 
          19354151     A18_jit   COMPLETED   02:46:13 
```

## A18 — seed consistency
```

=== no overdispersion ===
      default: consistency 0.799   effective K 4.0/4 (100%)   accuracy 0.989
       jitter: consistency 0.798   effective K 4.0/4 (100%)   accuracy 0.951
   jitter+phi: consistency 0.794   effective K 4.0/4 (100%)   accuracy 0.918

=== mild (c=20) ===
      default: consistency 0.763   effective K 3.6/4 (90%)   accuracy 0.956
       jitter: consistency 0.764   effective K 3.2/4 (80%)   accuracy 0.924
   jitter+phi: consistency 0.764   effective K 3.3/4 (82%)   accuracy 0.923

=== atlas-like (c=2) ===
      default: consistency 0.518   effective K 1.2/4 (30%)   accuracy 0.699
       jitter: consistency 0.543   effective K 2.1/4 (52%)   accuracy 0.781
   jitter+phi: consistency 0.649   effective K 2.6/4 (65%)   accuracy 0.729

=== summary ===
  no overdispersion    consistency -0.001  accuracy -0.038   -> no material effect
  mild (c=20)          consistency +0.001  accuracy -0.032   -> no material effect
  atlas-like (c=2)     consistency +0.025  accuracy +0.082   -> helps on both

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/A18_jitter.csv
```

## A17 — location vs symmetry
```
  A default (0.5, symmetric)   consistency 0.518  effective K 1.2/5  accuracy 0.699
    [B mean (symmetric)] contrast 0.0000 mean 0.333
  B mean (symmetric)           consistency 0.486  effective K 1.6/5  accuracy 0.689
    [C mean + fixed jitter] contrast 0.0613 mean 0.336
  C mean + fixed jitter        consistency 0.543  effective K 2.1/5  accuracy 0.781
    [D NNDSVD (data-driven)] contrast 0.1106 mean 0.297
  D NNDSVD (data-driven)       consistency 0.704  effective K 3.0/5  accuracy 0.847

=== what actually matters? ===
  location alone (A->B):            -0.010 accuracy
  + symmetry, uninformative (B->C): +0.092
  + data information (C->D):        +0.066

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/A17_offset_vs_symmetry.csv
```

## A16 — psi-init vs minibatch
```

=== no overdispersion ===
      random: consistency 0.799  effective K 4.0/5  accuracy 0.989
         psi: consistency 0.801  effective K 4.0/5  accuracy 0.988
   psi + phi: consistency 0.802  effective K 4.0/5  accuracy 0.989

=== atlas-like (c=2) ===
      random: consistency 0.518  effective K 1.2/5  accuracy 0.699
         psi: consistency 0.704  effective K 3.0/5  accuracy 0.847
   psi + phi: consistency 0.781  effective K 3.7/5  accuracy 0.848

=== decomposition ===

no overdispersion:
  psi init      +0.002
  phi init      +0.001
  residual (minibatch only) 0.802

atlas-like (c=2):
  psi init      +0.187
  phi init      +0.077
  residual (minibatch only) 0.781

wrote /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/A16_init_decomposition.csv
```

## Full simulation benchmark with jitter — every Figure 2 metric
```
108 rows from 18 datasets

--- JITTER init, LeafletFA arms ---
                               psi_r    sil    ari  auroc  spearman_delta
sim_bb_conc effect likelihood                                            
2.0         0.25   bb          0.873  0.194 -0.024  0.817           0.238
                   binom       0.919 -0.002  0.083  0.693           0.039
20.0        0.25   bb          0.966  0.605  0.491  0.929           0.416
                   binom       0.976  0.525  0.896  0.880           0.371
inf         0.10   bb          0.960  0.556  0.118  0.888           0.364
                   binom       0.975  0.585  0.307  0.866           0.336
            0.25   bb          0.993  0.815  0.994  0.967           0.653
                   binom       0.992  0.806  0.976  0.969           0.653
            0.50   bb          0.994  0.853  1.000  0.992           0.844
                   binom       0.995  0.854  1.000  0.991           0.853
            1.00   bb          0.996  0.867  1.000  0.999           0.964
                   binom       0.996  0.868  1.000  0.999           0.965

--- DEFAULT init, same effect sweep (for comparison) ---
                   psi_r    sil    ari  auroc  spearman_delta
effect likelihood                                            
0.10   bb          0.941  0.535  0.274  0.888           0.396
       binom       0.984  0.598  0.401  0.862           0.361
0.25   bb          0.994  0.820  0.995  0.966           0.679
       binom       0.991  0.809  0.926  0.968           0.633
0.50   bb          0.994  0.854  1.000  0.992           0.847
       binom       0.995  0.853  1.000  0.992           0.849
1.00   bb          0.996  0.868  1.000  0.999           0.961
       binom       0.996  0.868  1.000  0.999           0.960
```

## Decision

Jitter improves BOTH consistency and accuracy against known phi at atlas-like
overdispersion. Atlas retrain (K=30, 3 seeds, jitter init) submitted:
Submitted batch job 19354509

When it finishes, run the calibration test:
```
python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/A19_uncertainty_calibration.py /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-28_full_K30_jitter 30 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-28_full_K30_jitter/A19_calibration.csv
```

## Standing conclusions (unchanged)

- Psi reproduces at r = 0.97 and the program x cell-type map at r = 0.90 — the paper's claims.
- Overdispersion (c = 1.96, flat across K=10/20/30) caps per-junction information at ~3 reads
  regardless of depth; it penalises a wrong phi 20x less than no overdispersion. Not fixable.
- Longer training, the junction prior, and program collinearity were each tested and refuted.
- phi uncertainty is now saved by A1 (phi_sd, phi_top_prob) for the calibration test.
