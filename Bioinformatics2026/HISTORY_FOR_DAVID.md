# LeafletFA, June to November 2026: dated timeline with sources

**1. Plan, June to August**
- **Target journal: Genome Biology, not Genome Research.**
  - Source: `Leaflet-analysis/GenomeBiology_Submission2026/THESIS.md`, added 2026-07-20 in ccc32d4.
  - "Genome Research" appears in none of the sources read: UNSOURCED.
- **Single-cell aim** (THESIS.md): recover splicing variation across and within cell types from sparse reads, and impute per cell reliably enough to serve as a resource.
- **Claims and data** (`LeafletFA_overleaf_old/MANUSCRIPT.tex`, only commit 946e49a, 2026-07-08):
  - More than 200,000 mouse and human cells and nuclei across ages.
  - Mouse model fit on 142,315 cells (159 tissue–cell types), K = 20.
  - Aging claims, e.g. older cells more similar (Spearman 0.712 vs 0.647).
  - Splicing adds to expression in age prediction for 46 of 76 types.

**2. Switch to the Dirichlet-multinomial (DM), 2026-08-03**
- **Code:** e563a26 (LeafletFA repo).
- **Why:** it is the same likelihood as LeafCutter's (`METHODS_dirichlet_multinomial.md`).
- **Fixed:**
  - Program ψ now sums to 1 within every ATSE: 100 % of 24,475 events, vs 86.9 % under the Beta-binomial (sums 0 to 4.4).
  - About 3× cheaper to fit (f0f93ba).
- **Not fixed:** "The DM is not the calibration fix" (df9c3d4, 2026-08-03).

**3. What still failed** (`LeafletFA/scripts/validation/VALIDATION_REPORT.md`)
- **Programs across seeds (check 3, after amendment 2, 2026-10-25):**
  - Centered-ψ replication: TMS 3 programs (needed 4), Allen 1 (needed 3).
  - Consensus over 15 seeds (check 8, 487902c, 2026-10-30): 3 recurrent programs vs 8 needed.
- **Differential splicing across seeds (check 9, 2654c02 and f0c3ce7, 2026-11-01):**
  - Shipped test: median Jaccard 0.0. Genome Biology Fig. 3c setting: 0.548. Pass needed 0.7.
  - Per-cell ψ does reproduce (r 0.998).
- **Check 10.2, cell-type-specific calls (bea2b12, 2026-11-02):**
  - Confirmed vs null: TMS 0.813 vs 0.422; Allen 0.837 vs 0.771. Rule: 5× the null.
  - Check 10.7, pairwise calls: 0.780 vs 0.473. FAIL.
- **Check 10.3, interval coverage (bea2b12):** 0.136 to 0.346 against nominal 0.95.
- **Allen null (fb81dee 2026-11-04; a7945e3 2026-11-06):**
  - Donors were sampled by Cre line, so cell class is confounded with donor.
  - The within-donor null is inflated: about 0.31 when donors are ignored.
  - LeafCutter shows the same (0.604 vs 0.229).
- **Smoothing (688780a, 2026-11-06):** at the most different events, cell-type Δψ is compressed by up to about 0.4 in TMS, because cells without reads pull the estimate toward the shared programs.
- **Aging:** check 0 flagged the aging replication as a serious text problem (VALIDATION_REPORT.md, Verdict); aging is now dropped from the paper. The date of the decision to drop it: UNSOURCED.

**4. Application Note**
- **Decision (bea2b12, 2026-11-02):** the pre-registered reading was "10.1 passes, 10.2 fails → Application Note", accepted by Karin.
  - A full Bioinformatics paper had been planned on 2026-11-01 (4b8337c).
- **Scope:**
  - Atlas: catalog and model.
  - Browser.
  - Check 10.1 accuracy: LeafletFA beats raw pseudobulk on held-out cells. Error difference: TMS −0.0072 (CI −0.0094 to −0.0050); Allen −0.0021 (CI −0.0039 to −0.0005).
  - Gria2, Fgfr2, Nrxn1 and Ptprc as examples.
  - Calls and intervals reported as limits.
