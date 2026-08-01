#!/usr/bin/env python
"""Render the HELD-OUT IMPUTATION half of Figure 2 (panels 2E-F), its supplements, and the
assembled main figure.

    python code/build_imputation.py

Reads the 70/30 held-out models at `panels.IMPUTE_DIR` (2026-07-30_atlas9_default -- the fresh
default-init fits; 2026-07-26_atlas9 is superseded). "atlas" is deliberately absent from the names
here: in this paper atlas means the full dataset and the full-data models, not this analysis.

FIG2_validation is the assembled main figure and spans both halves, so it lives here rather than in
build_sim.py -- it is the one that needs the model runs present.
"""
import panels as P

FIGURES = (
    ("FIG2_validation",                 P.figure_validation),
    ("FIG2_imputation",                 P.figure_imputation),
    ("FIG2_supp_heldout_structure",     P.figure_heldout_structure_supp),
    ("FIG2_supp_program_celltype",      P.figure_program_celltype),
    ("FIG2_supp_program_celltype_K10",  lambda: P.figure_program_celltype_all(10)),
    ("FIG2_supp_program_celltype_K20",  lambda: P.figure_program_celltype_all(20)),
    ("FIG2_supp_program_celltype_K30",  lambda: P.figure_program_celltype_all(30)),
    # K20/K30 were being rendered to disk without ever being in the build list -- all three now are.
    ("FIG2_supp_transfer_controls_K10", lambda: P.figure_transfer_controls(10)),
    ("FIG2_supp_transfer_controls_K20", lambda: P.figure_transfer_controls(20)),
    ("FIG2_supp_transfer_controls_K30", lambda: P.figure_transfer_controls(30)),
    ("FIG2_supp_transfer_all_K",        P.figure_transfer_all_K),
    ("FIG2_supp_cellcell_consistency",  P.figure_cellcell_consistency),
)


def main() -> int:
    print(f"models: {P.IMPUTE_DIR}\n")
    failed = []
    for name, fn in FIGURES:
        try:
            fn()
            print(f"wrote plots/{name}.{{pdf,png}}")
        except FileNotFoundError as e:
            print(f"FAILED {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)}/{len(FIGURES)} figures failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(FIGURES)} imputation figures rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
