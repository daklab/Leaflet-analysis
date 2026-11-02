# Definitions as implemented (ATSEs, ψ, the check 10 specificity rule)

2026-11-01. Read from the code, not from documentation. Line numbers refer to:
- ATSEmapper `main` (`~/ATSEmapper`, commit d15de61);
- Leaflet-analysis (`GenomeBiology_Submission2026/pipelines/`, `config/`);
- LeafletFA `leafletfa-validation` (`~/LeafletFA`).

**Where the production mouse catalog came from:**
`pipelines/02_atsemapper/04_run_atsemapper.py` (ATSEmapper modules, settings from `config/mouse_smartseq2.yaml`), then
the re-grouping in `pipelines/04_align/09b_filter_aligned.py` after the cell and junction filters (`CATALOG_NUMBERS.md`).

## 1. ATSEmapper: from junction reads to ATSEs

### 1.1 Junction reads (per cell file → pooled)
- **Input:** regtools junction BED, one file per cell (`junction_parser.py:44-47`).
- **Coordinates:** the intron is chromStart + first block size to chromEnd − second block size (`:50-54`).
- **Intron length filter:** 50 ≤ length ≤ 500,000 (`:57-60`; production `config/mouse_smartseq2.yaml:35-36`).
- **Chromosomes:** only `^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|MT)$` (`:61-62`), so `chrM` and unplaced contigs are dropped.
- **Junction id:** `chrom_start_end_strand` (`:64-69`).
- **Pooling:** per junction, `cells` = number of cell files containing it and `total_score` = summed reads (`:71-85`,
  `:103-139`).
- **Junction QC:** keep if `cells ≥ min_cells` and `total_score ≥ min_reads` (`:166`). CLI defaults are 2 cells and 100
  reads (`main.py:35-37`); **production used ≥ 10 cells and ≥ 100 reads, pooled over all cells**
  (`config/mouse_smartseq2.yaml:37-38`, applied in `pipelines/01_junctions/03_merge.sh:32-58`).

### 1.2 Canonical motifs
- The 2 bp at each end are read from the genome FASTA (`genome_utils.py:71-82`).
- **Canonical:** GT-AG, GC-AG, AT-AC (`:60-64`); anything else is "non-canonical" (`:88-92`).
- **Production removes non-canonical junctions** (`04_run_atsemapper.py:69`, `only_canonical=True`; CLI flag
  `--only-canonical`, `main.py:51`).

### 1.3 Annotation and gene assignment
- **Matching against transcripts:** for each junction, same-strand transcripts in the region (± 1 kb) are checked for an
  exon boundary at each end within **tolerance 1 bp** (`genome_utils.py:136-281`; tolerance `:55-59`, production
  `04_run_atsemapper.py:65`).
- **Perfect match:** an end matches exactly, offset 0 or 1 (`:225-232` and the three analogous blocks).
- **Genes:** a junction is assigned the genes of every transcript with a matching end (`genes_found`, `:285-298`,
  stored as `gene_ids` `:324`).
- **Annotation status** (`filter_annotated`, `:374-389`):

  | Status | Meaning |
  |---|---|
  | `both` | both ends match annotated exon boundaries |
  | `five_prime` | only the 5′ end matches |
  | `three_prime` | only the 3′ end matches |
  | `unannotated` | neither end matches |

- **Which statuses are kept:** `both` → `both` only; `either` → `both`/`five_prime`/`three_prime`; `unanno_also` → all
  (`:393-399`). **Production: `unanno_also`** (`config/mouse_smartseq2.yaml:41`).
- **Consequence:** an unannotated junction has no gene (no transcript matched either end), and junctions without a gene
  are excluded when the graph is built (`event_detection.py:30-32`). So the production catalog effectively contains only
  junctions with at least one annotated end.
- **Multi-gene junctions** (> 1 gene id) are **removed** and written to `multi_gene_junctions_<time>.csv`
  (`genome_utils.py:366-370`, `:408-411`). Graph building takes `gene_ids[0]` (`event_detection.py:32`), which is then
  the only gene.

### 1.4 Linking junctions into an ATSE
- **The graph:** one graph per gene (`event_detection.py:15-56`). Nodes are splice sites `(chrom, position, 'donor' |
  'acceptor')`; each junction is an edge from its donor to its acceptor. Donor and acceptor are strand-aware: on +
  donor = start; on − donor = end (`:43-48`).
- **Two junctions are linked if they share a donor or share an acceptor** (the same node).
  - A donor and an acceptor at the same coordinate are different nodes, so "exon end = next intron start" does not link.
- **ATSE = a connected component** of this graph, with links followed transitively
  (`find_connected_junctions`, `nx.node_connected_component`, `:88-113`), containing ≥ 2 junctions (`:113`).
- **Splice-site usage filter** (`find_atse_groups`, `:198-260`):
  - For each junction, its reads divided by the total reads of all component junctions at its donor, and at its acceptor.
  - The junction is removed if **either share is < `min_splice_site_usage`** (default 0.01, `main.py:57`; production
    0.01, `04_run_atsemapper.py:86`; `:242-256`).
- **Re-split after filtering:** the remaining junctions are re-split into connected components (`:262-282`); each
  component with ≥ 2 junctions is an ATSE (`:289-310`).
- **Naming:** ATSEs are numbered along the gene as `<gene_id>_atse_<n>` (`_finalize_atse_groups`, `:407-461`).
- **Singletons:** a junction that shares no splice site with another junction, or is left alone after the usage filter.
  - Singletons are **dropped** and written to `singleton_junctions.tsv` (`:311-332`, `:344-351`).
  - This removes real constitutive junctions (README, "Known limitation").
  - **Worked example, *COL1A1*** (human bulk RNA atlas, `ENSG00000108821.14`, found 2026-07-11): two adjacent exons are
    skipped together.
    - `atse_5` keeps three junctions: the two inclusion junctions into and out of the exon pair (2.94 M and 2.75 M
      reads) and the skipping junction (67 k reads). All three share splice site 50189010 or 50189678.
    - The **internal** junction between the two skipped exons (50189275–50189376; 2.88 M reads, in 2,095 of 2,228
      samples) shares a splice site with none of them, so it is a singleton and is dropped. The same happens in
      `atse_4` (50189912–50190000, 1.71 M reads).
    - **Why this is acceptable:** the internal junction is used whenever the exon pair is included and never when it is
      skipped, so it carries no information on the inclusion-versus-skipping choice, and the event's denominator is
      unaffected. ATSEmapper describes alternative usage, not total junction coverage.
- **Observation (not tested):** the donor/acceptor labels used for the usage columns come from the undirected graph's edge
  iteration order (`:197-205`), which can swap them for some junctions. The filter is symmetric (either share < 0.01), so
  only the reported `five_prime_usage` / `three_prime_usage` columns could be mislabeled, not the ATSEs.
  **Checked 2026-11-01:** no reported number uses these columns (no code in Leaflet-analysis, LeafletFA or
  SpliceVI-utils reads them; they appear only in a SpliceVI plotting notebook's saved outputs). Fix after submission:
  take donor and acceptor from the junction's strand-aware coordinates instead of the edge order.

### 1.5 Production re-grouping after filtering (Leaflet-analysis `09b_filter_aligned.py`)
- **Inputs:** after cell QC and the junction filter (junction in ≥ 75 cells; cells with ≥ 500 detected junctions),
  ATSEs are re-derived on the surviving junctions with ATSEmapper's `ATSEAnalyzer` (`09b_filter_aligned.py:191-290`).
- **Settings:** `build_splice_graph`, then `find_atse_groups(min_splice_site_usage=0.0)`, so no usage filter the second
  time (`:239-249`); junction scores are the summed reads, at least 1 (`:221-237`).
- **Singletons are dropped:** 9,065 junctions (`CATALOG_NUMBERS.md`).
- **ATSE totals:** recomputed from the surviving junctions only (`recompute_cluster_matrix`, `:123-185`).
- **Result:** 79,768 junctions in 24,475 ATSEs.

### 1.6 Event types of the 24,475 ATSEs (computed from junction coordinates, 2026-11-01)

| Type (rule) | ATSEs | Share |
|---|---|---|
| Alternative 3′ site (2 junctions, shared donor) | 5,810 | 23.7 % |
| Alternative 5′ site (2 junctions, shared acceptor) | 4,867 | 19.9 % |
| Skipped exon (3 junctions: one spans the other two, sharing its donor and acceptor; the two do not overlap) | 5,708 | 23.3 % |
| Complex (all other 3-junction ATSEs and every ATSE with ≥ 4 junctions) | 8,090 | 33.1 % |

- **Junctions per ATSE:** 2: 10,677; 3: 7,600; 4: 2,381; 5: 1,561; 6: 807; ≥ 7: the rest. Mean 3.26.
- **Mutually exclusive exons (corrected 2026-11-01, from the SpliceVI session).** A mutually exclusive exon pair forms one
  ATSE only if another junction links its two halves; otherwise it appears as two ATSEs, an alternative-3′ and an
  alternative-5′ pair. (The two upstream junctions share a donor and the two downstream junctions share an acceptor; the
  pairs share no splice site unless a third junction, e.g. one that skips both exons or joins them, connects them.)
  - **Linked, one ATSE:** *Gria2* flip/flop, `ENSMUSG00000033981.14_atse_3`, 7 junctions.
  - **Unlinked, two ATSEs:** *Fgfr2* IIIb/IIIc. `ENSMUSG00000030849.18_atse_3` is the upstream pair (shared exon-7 site
    130,200,982 → IIIb or IIIc; classified "alternative 3′ site") and `ENSMUSG00000030849.18_atse_4` the downstream pair
    (IIIb or IIIc → shared site 130,196,374); no junction links them (checked 2026-11-01 from the catalog var).
  - The geometric rule in the table never labels an ATSE "mutually exclusive": linked MXE ATSEs fall under "complex".
- These labels are geometric. A 2-junction "alternative 3′ site" ATSE can also be the upstream half of an MXE or an
  alternative last exon.

## 2. ψ

**Per cell, observed** (model input):
- y_cj reads on junction j in cell c, N_ce = Σ_{j∈e} y_cj (`METHODS_LEAFLETFA_CURRENT.md` §1);
- raw ψ_cj = y_cj / N_ce, undefined where N_ce = 0.

**Per cell, model:**
- ρ_cj = Σ_k φ_ck ψ_kj (`leafletfa/minibatch.py:364`, dense; `dirichlet_multinomial.py:402`, sparse).
- ψ_k is a softmax within each ATSE (`dirichlet_multinomial.py:211`).
- This is the prior mean of the cell's composition.

**Per cell, posterior mean ("imputed"):**
- ρ̃_cj = (γ_e ρ_cj + y_cj) / (γ_e + N_ce), γ_e the per-ATSE concentration.
- Equal to ρ_cj where the cell has no reads at e.
- Used in the GB *Fgfr2* figure (`figures/FIGURE2_simulation_imputation/code/build_fgfr2_example.py:96`) and in check 10
  (`LeafletFA scripts/validation/c10_catalog.py:7-12`, `:106-113`).

**Per cell type (check 10):**
- LeafletFA cell-type ψ_tj = mean of ρ̃_cj over the type's TRAIN cells.
  - Computed as Σ_k ψ_kj [Σ_c φ_ck − Σ_c v_ce φ_ck] + Σ_c y_cj/(γ_e + N_ce), with v_ce = N_ce/(γ_e + N_ce)
    (`c10_catalog.py:94-113`).
  - Point estimate from the fit's φ and ψ; 95 % interval over 200 draws (`estimates`, `:127-155`).
  - **What a draw is** (amendment 6 note 2): one joint draw of φ for all TRAIN cells from q(φ) and of every program's
    ψ_k from q(ψ); y, N and γ_e fixed; given the draw, each cell's ρ̃ is computed deterministically and averaged over the
    type. No per-cell Dirichlet-multinomial sampling, no bootstrap over cells. The interval is the 2.5–97.5 % quantile of
    the type mean over draws, so it reflects model-parameter uncertainty only. Variational posteriors are often too narrow;
    interval coverage (10.3) and the mouse-half replication are the safeguards.
- Raw pseudobulk ψ_tj = Σ_c y_cj / Σ_c N_ce over the type's TRAIN cells. Where the type has no reads at e, the TRAIN
  population ψ is used (`:214-219`).
- **Leave-one-type-out ψ_−t:** the same estimator over all other TRAIN cells, from the same draws (`:131-132`, `:146-148`).

## 3. Check 10 specificity rule (amendment 6, 10.2; `c10_catalog.py`)

As amended by amendment 6 note 2 (recorded before any check 10 result was read), (event e, cell type t) is called
specific if all of the following hold (`call`, `:172-175`):
1. **Effect:** TV = ½ Σ_{j∈e} |ψ_tj − ψ_−t,j| > δ, on point estimates (`tv`, `:158-163`).
2. **Posterior probability:** P(TV > δ) ≥ 0.95, where TV is computed per draw between the type and the rest from the
   same draws (`:149-150`).
3. **Data in t:** ≥ 20 TRAIN reads at e and ≥ 3 TRAIN cells with reads at e (`:175`; constants `:23`). No calls from
   the prior alone.

**δ** = the 99th percentile of TV between two within-mouse halves of each type's TRAIN cells (point estimates), over
callable units, rounded up to the next 0.05. It is computed before the draws (`:219-229`).

**Pairwise version (10.7):** the same conditions between two types t and u, with P(TV > δ) from per-draw TV between the
two types (`:151-153`) and the read and cell condition required in both (`pairwise`, `:345-353`).

(The pre-amendment rule used non-overlapping 95 % intervals at ≥ 1 junction instead of condition 2.)
