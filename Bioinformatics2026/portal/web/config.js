// Every wording and rule the browser must state in one place. Edit here; nothing else repeats these values.
const CONFIG = {
  INTERVAL_LABEL: "posterior interval (not calibrated; see Supplementary)",
  COMPARE_NOTE: "Ranking for exploration; LeafletFA's cell-type specificity calls did not pass validation (see paper).",
  // Which cell types show estimates in grey.
  //   "eligible": saved flag of check 10 is false (fewer than 2 mice, or no cells in one of the two frozen mouse halves)
  //   "n_mice_lt2": fewer than 2 mice only
  GREY_RULE: "eligible",
  DEFAULT_MIN_READS: 20,
  DEFAULT_MIN_CELLS: 3,
  PAGE_SIZE: 100,
  SPLICEVI_HEADER: "SpliceVI DS call",
  SPLICEVI_EMPTY_NOTE: "from SpliceVI cell-type DS runs; not yet filled",
  DATASET_COLOR: { TMS: "#5E5A7A", Allen: "#A8B39A" },
  // Fixed order; junctions past the eighth use the neutral color and rely on their J label.
  JUNCTION_COLORS: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  JUNCTION_OVERFLOW_COLOR: "#6E7478",
  CLASS_LABEL: { alt3: "alternative 3′ site", alt5: "alternative 5′ site", skipped_exon: "skipped exon", mxe_linked: "MXE linked", complex: "complex" },
  ANNOTATION_LABEL: ["both ends annotated", "5′ end only", "3′ end only", "unannotated"],
};
