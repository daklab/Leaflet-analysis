# LeafletFA, June to November 2026

**1. Plan, June to August**
- **Journal:**
  - Genome Biology, per `Leaflet-analysis/GenomeBiology_Submission2026/THESIS.md` (ccc32d4, 2026-07-20).
  - "Genome Research" appears in none of the sources read: UNSOURCED.
- **Aim:** recover splicing variation across and within cell types from sparse single-cell reads, and impute per cell (THESIS.md).
- **Claims and data** (`LeafletFA_overleaf_old/MANUSCRIPT.tex`, 946e49a, 2026-07-08):
  - More than 200,000 mouse and human cells across ages.
  - Mouse fit on 142,315 cells (159 types), K = 20.
  - Aging claims (older cells more similar, Spearman 0.712 vs 0.647).
  - Splicing improves age prediction in 46 of 76 types.

**2. Dirichlet-multinomial (DM), 2026-08-03 (e563a26)**
- **Why:** the same likelihood as LeafCutter (`LeafletFA/METHODS_dirichlet_multinomial.md`).
- **Fixed:**
  - Program ψ sums to 1 in 100 % of 24,475 ATSEs, vs 86.9 % under the Beta-binomial.
  - About 3× cheaper (f0f93ba).
- **Not fixed:** "The DM is not the calibration fix" (df9c3d4).

**3. Still failing** (`LeafletFA/scripts/validation/VALIDATION_REPORT.md`)
- **Programs across seeds:**
  - Centered-ψ replication: TMS 3 (needed 4), Allen 1 (needed 3). Check 3, run after amendment 2, 2026-10-25.
  - Consensus: 3 of 8 needed (487902c, 2026-10-30).
- **Differential splicing across seeds** (2654c02, f0c3ce7, 2026-11-01):
  - Median Jaccard 0.0 for the shipped test, 0.548 in the GB Fig. 3c setting. Pass needed 0.7.
  - Per-cell ψ reproduces (r 0.998).
- **Check 10.2, cell-type-specific calls** (bea2b12, 2026-11-02):
  - Confirmed vs null: TMS 0.813 vs 0.422; Allen 0.837 vs 0.771. Rule: 5× the null.
  - Check 10.7, pairwise: 0.780 vs 0.473.
- **Interval coverage (10.3):** 0.136 to 0.346, nominal 0.95.
- **Allen null** (fb81dee; a7945e3, 2026-11-06):
  - Donors were sampled by Cre line, so the within-donor null is inflated: about 0.31 when donors are ignored.
  - LeafCutter shows the same (0.604 vs 0.229).
- **Smoothing** (688780a, 2026-11-06): the largest cell-type differences are compressed by up to about 0.4 (TMS), from cells without reads.
- **Aging:** check 0 flagged the aging replication as serious. Date of the decision to drop aging: UNSOURCED.

**4. Application Note**
- **Decision** (bea2b12, 2026-11-02): the pre-registered reading "10.1 passes, 10.2 fails → Application Note", accepted by Karin. A full paper had been planned on 2026-11-01 (4b8337c).
- **Scope:**
  - Catalog and model, and the browser.
  - Check 10.1: LeafletFA beats raw pseudobulk on held-out cells. TMS −0.0072 (CI −0.0094 to −0.0050); Allen −0.0021 (−0.0039 to −0.0005).
  - Gria2, Fgfr2, Nrxn1 and Ptprc as examples.
  - Calls and intervals reported as limits.
