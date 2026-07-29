# Initialisation comparison — simulation and 70% atlas

**Goal:** consistent, meaningful programs across random seeds.

Every arm is scored on BOTH axes. Consistency alone is not success: an initialisation that makes
runs agree on a worse answer would score well on consistency and badly on accuracy, and that has
already been observed once (pinning phi with arbitrary offsets: consistency 0.543 -> 0.649,
accuracy 0.781 -> 0.729).

Reproducible fractions are of **K-1**, not K: phi lies on the simplex, so K programs span only
K-1 directions and the K-th canonical correlation is numerically arbitrary.

## Job outcomes
```
           JobID    JobName       State    Elapsed 
---------------- ---------- ----------- ---------- 
        19358688   A20_init   COMPLETED   01:41:05 
      19358721_0       wp70   COMPLETED   01:07:16 
      19358721_1       wp70   COMPLETED   01:06:03 
      19358721_2       wp70   COMPLETED   00:56:04 
      19358950_0    wp70phi   COMPLETED   01:15:45 
      19358950_1    wp70phi   COMPLETED   01:09:39 
      19358950_2    wp70phi   COMPLETED   01:02:43 
        19359216   A21_pyro      FAILED   00:00:31 
      19359217_0      smp70   COMPLETED   00:02:30 
      19359217_1      smp70   COMPLETED   00:01:54 
      19359217_2      smp70   COMPLETED   00:01:55 
```

## Simulation — all arms, three overdispersion levels
```

--- reproducible fraction of K-1 ---
arm                default  waypoints  waypoints+phi
label                                               
atlas-like (c=2)     0.417      0.417           0.50
mild (c=20)          0.833      0.833           0.75
no overdispersion    1.000      1.000           1.00

--- accuracy vs known phi ---
arm                default  waypoints  waypoints+phi
label                                               
atlas-like (c=2)     0.788      0.718          0.801
mild (c=20)          0.959      0.950          0.859
no overdispersion    0.990      0.976          0.980

--- both, at atlas-like overdispersion (c=2) ---
                frac  phi_recovery  consistency
arm                                            
waypoints+phi  0.500         0.801        0.560
default        0.417         0.788        0.570
waypoints      0.417         0.718        0.543
```

## 70% atlas, K=30, 3 seeds
```
arm                             effective K  consistency   psi across seeds
default (baseline)            11.3/29 (39%)        0.650              0.966
waypoints (psi only)          10.7/29 (37%)        0.644              0.961
waypoints + phi                9.7/29 (33%)        0.641              0.953
Traceback (most recent call last):
  File "<stdin>", line 28, in <module>
  File "<stdin>", line 28, in <listcomp>
  File "<stdin>", line 7, in cca
  File "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/lib/python3.10/site-packages/numpy/linalg/linalg.py", line 1693, in svd
    s = gufunc(a, signature=signature, extobj=extobj)
  File "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/lib/python3.10/site-packages/numpy/linalg/linalg.py", line 121, in _raise_linalgerror_svd_nonconvergence
    raise LinAlgError("SVD did not converge")
numpy.linalg.LinAlgError: SVD did not converge
```

## How to read this

- **effective K** — how many of the K-1 phi directions two seeds agree on (canonical corr > 0.9).
- **psi across seeds** — Hungarian-matched program-profile correlation. This is the number the
  paper's claim rests on, and it was already ~0.97 with the default init.
- Pick on BOTH simulation accuracy and atlas reproducibility. Simulation has mispredicted the
  atlas once already: jitter helped at c=2 in simulation and made the atlas worse (36% -> 29%).

## Next

Once an arm wins on both, run the calibration test on its models:
```
python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/A19_uncertainty_calibration.py <model_dir> 30 <out.csv>
```
It asks whether per-cell posterior width predicts WHICH cells the seeds disagree on. If it does,
the residual disagreement is reportable as calibrated uncertainty rather than instability.

---

# VERDICT

**No initialisation beats the default. Stop trying to fix initialisation.**

| intervention | atlas effective K (of 29) | Ψ across seeds | verdict |
|---|---|---|---|
| **default (`init_to_feasible`)** | **11.3 (39%)** | **0.966** | **best on every metric** |
| waypoints (ψ only) | 10.7 (37%) | 0.961 | worse |
| waypoints + φ | 9.7 (33%) | 0.953 | worst of the three |
| jitter | 8.3 (29%) | — | worse |
| NNDSVD | — | — | assignment degenerates at K=30 (22/30 groups collapse) |
| `init_to_sample` | — | — | **NaN at step 1–2; does not train** |
| 2× training budget | no change | — | no effect (tested twice) |
| drop junction-specific prior | no change | — | no effect |

## Simulation mispredicted the atlas twice

| | simulation (c=2) | atlas (K=30) |
|---|---|---|
| jitter | helped (30% → 52%) | **hurt** (36% → 29%) |
| waypoints + φ | helped (42% → 50%) | **hurt** (39% → 33%) |

K=5 with 6,986 cells does not predict K=30 with 92,894. **Any future initialisation claim must be
validated on the atlas**; simulation is advisory only.

Also note the simulation's own measurement noise: the *same* default arm at c=2 scored accuracy 0.699 /
30% in A15 (5 seeds) and 0.788 / 42% in A20 (3 seeds). That swing is larger than most between-arm
differences, so simulation gaps below ~0.1 should not be read as real.

## What this means

The reproducibility ceiling is **overdispersion**, and it is not reachable by a better starting point.
Every attempt to reach it has left the model where it was or made it worse. Combined with the earlier
mechanism work — `bb_conc` ≈ 1.96 flat across K, information saturating at ~3 reads per junction
regardless of depth, the likelihood penalising a wrong φ 20× less at c = 2 — the conclusion is that
per-cell φ is underdetermined by the data, full stop.

**The paper's claim was never about per-cell φ.** Ψ reproduces at **0.966** across seeds with the
default init, π at 0.93, and the program × cell-type map at 0.90. Those are the objects the results
rest on, and they are already excellent — and marginally *better* under the default than under any
alternative tried.

## Recommendation

1. **Keep the default initialisation.** It is the best-performing option measured, and it is what every
   existing model already uses — no retraining needed.
2. **Release one model**, chosen as the medoid across three seeds (not best ELBO — those disagreed at
   K = 20 and 30).
3. **Report three seeds as validation**, leading with Ψ = 0.966 and the cell-type map at 0.90.
4. **Report effective K alongside nominal K**, framed as a property of the assay's overdispersion —
   which is now supported by a dose–response in simulation and by every failed intervention here.
5. **Ship φ with its posterior width** (`phi_sd`, `phi_top_prob`, now saved by A1) so users can tell
   which cells are determined and which are not.

Do not: retrain with a different init, ensemble, train longer, or add penalties. All measured, none help.

## Why initialising φ cannot help — the mechanism (A22)

At atlas-like overdispersion, after fitting from a waypoint φ start:

| | value |
|---|---|
| persistence, corr(φ_final, φ_init) | **−0.289** |
| compositional null for K = 5, −1/(K−1) | **−0.25** |
| improvement, accuracy(final) − accuracy(init) | **+0.604** |

Persistence sits *at* the null, so the final φ bears no relationship to where it started — the
optimiser overwrites φ completely, and ends up far better than its own initial point.

This retires a concern raised earlier in this work: that φ_init would persist in the degenerate regime
(where the data cannot overrule it) and buy seed agreement through bias rather than through fit. It
does not persist. And that is precisely why initialising φ cannot improve reproducibility — nothing of
it survives to the end. The earlier jitter+φ observation (consistency up, accuracy down) is better
explained as the arbitrary ψ start damaging the fit than as φ_init anchoring it.

Consistent with this: φ is a LOCAL variable, and given Ψ the per-cell problem is nearly convex, so the
optimiser reliably finds its own φ from anywhere. Initialisation leverage lives in Ψ, and Ψ already
reproduces at 0.966.
