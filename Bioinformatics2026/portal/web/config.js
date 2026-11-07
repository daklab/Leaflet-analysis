// Every wording and rule the browser must state in one place. Edit here; nothing else repeats these values.
const CONFIG = {
  INTERVAL_LABEL: "95 % predictive interval for the pseudobulk ψ of this type's held-out cells at their read depth (check 12; shown where held-out cells have ≥ 100 reads and ≥ 10 cells with reads at the ATSE)",
  IV_MIN_TEST_READS: 100,   // check 12 units; must equal c12_export.MIN_READS / MIN_CELLS
  IV_MIN_TEST_CELLS: 10,
  REST_INTERVAL_NOTE: "rest ψ has no validated interval",
  COMPARE_NOTE: "Ranking for exploration; LeafletFA's cell-type specificity calls did not pass validation (see paper).",
  // Which cell types show estimates in grey.
  //   "n_mice_lt2": fewer than 2 mice (Karin's spec, 2026-11-06; default). No type has fewer than 2 mice at present, so nothing is grey.
  //   "eligible": alternative, the saved check 10 flag is false (fewer than 2 mice, or no TRAIN cells in one of the two frozen mouse halves)
  GREY_RULE: "n_mice_lt2",
  DEFAULT_MIN_READS: 20,
  DEFAULT_MIN_CELLS: 3,
  PAGE_SIZE: 100,
  SPLICEVI_HEADER: "SpliceVI DS call",
  SPLICEVI_EMPTY_NOTE: "from SpliceVI cell-type DS runs; not yet filled",
  DATASET_COLOR: { TMS: "#5E5A7A", Allen: "#A8B39A" },
  // Fixed order; junctions past the eighth use the neutral color and rely on their J label.
  // Light and dark are the same eight hues, each stepped for its surface (dataviz reference palette); both validated (README).
  JUNCTION_COLORS: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  JUNCTION_COLORS_DARK: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  JUNCTION_OVERFLOW_COLOR: "#6E7478",
  CLASS_LABEL: { alt3: "alternative 3′ site", alt5: "alternative 5′ site", skipped_exon: "skipped exon", mxe_linked: "MXE linked", complex: "complex" },
  ANNOTATION_LABEL: ["both ends annotated", "5′ end only", "3′ end only", "unannotated"],
};
