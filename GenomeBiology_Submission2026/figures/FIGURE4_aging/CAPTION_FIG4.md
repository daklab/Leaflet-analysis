# Supplementary Figure SX caption (formerly Figure 4) — draft 2026-08-19, revised 2026-10-25
# Decision (Karin, 2026-10-25): option (b) of manuscript/AGING_CLAIM_OPTIONS.md -- soften to partial
# replication (centred-psi cosine 0.65) and move from the main text to the Supplement.

> **PENDING (Karin, 2026-10-25): age is confounded with sequencing batch; this aging text is to be replaced by draft (a) or (b) in manuscript/AGING_SUPPLEMENT_DRAFTS.md. Do not cite it as an aging result.**

**Supplementary Figure SX. An aging-associated splicing program, partially reproducible across seeds.**
All aging analyses use Tabula Muris Senis cells only (the Allen Brain dataset is entirely 2-month
animals from one laboratory); ages are 3, 18 and 24 months, and because 24-month animals are all
male, formal tests are per-mouse and male-only.
**(a)** What each program tracks: each SP's standardised activity decomposed by Type-II ANOVA
(SP ~ tissue-cell-type + sex + age + tissue-cell-type:age + sex:age); cells show each term's share
of the SP's explained variance (printed when ≥ 0.05); row labels carry the model R². Cell identity
dominates every program; the aging-associated program carries the largest age share.
**(b)** The aging program's activity per cell by age (violins: median and IQR), with each mouse's
mean activity overlaid (circles = male, triangles = female). Mouse-level Mann-Whitney, males,
3m vs 24m (5 vs 4 mice): p = 0.016, the smallest attainable at these group sizes; activity rises
with age in 94–99% of tissue-cell-types in all three seeded fits. The program's specific
composition agrees only partly across seeds (centred-ψ cosine 0.65 in both other seeds; the threshold
for program-level claims is 0.8, Methods), so panels b–d describe the seed-0 program.
**(c)** Aging shift per tissue-cell-type (≥ 20 cells per age, males): tile colour = change in mean
SP activity from 3m to 24m; circle size = the cell type's median activity of that SP, so shifts in
programs a cell type actually uses stand out. Cell types sorted by the aging program's shift.
**(d)** What the aging program splices: junctions where its composition departs most from the
π-weighted mean of the other programs (top 14; rose = used more by the aging program).
**(e, f)** Worked example, chosen by screening the aging program's top-200 junctions across all
tissue-cell-types with ≥ 25 covered cells at each age: *Kcnip3* in marrow hematopoietic progenitors.
(e) gene model with the ATSE's junctions; the aging junction in teal. (f) that junction's usage per
covered cell by age (violin; diamond = pooled PSI: 0.35 → 0.12 → 0.03) while *Kcnip3* expression is
unchanged (inset; Spearman vs age 0.03).
