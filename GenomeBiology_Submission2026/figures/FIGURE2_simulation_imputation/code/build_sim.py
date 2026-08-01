#!/usr/bin/env python
"""Render the SIMULATION half of Figure 2 (panels 2B-D) and its supplements.

    python code/build_sim.py

Panels come from `panels.py`, which is the shared library for both halves -- the two builders are
separate entry points, not separate copies of the code. `figure_validation` is the assembled main
Figure 2 and deliberately spans both halves (Figure 2 *is* simulation + imputation), so it is
rendered by build_imputation.py, which is the one that depends on the model runs.

Everything here reads only ../data/ and writes only ../plots/ -- no model directory, no GPU.
"""
import panels as P

FIGURES = (
    ("FIG2_simulation",              P.figure_simulation),
    ("FIG2_supp_draft",              P.figure_sim_supp),
    ("FIG2_supp_seed_consistency_sim", P.figure_seed_consistency_sim),
)


def main() -> int:
    failed = []
    for name, fn in FIGURES:
        try:
            fn()
            print(f"wrote plots/{name}.{{pdf,png}}")
        except FileNotFoundError as e:
            # A missing input is a real problem, not something to shrug at -- report it and make the
            # exit code reflect it, rather than printing "skipped" and exiting 0 as before.
            print(f"FAILED {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)}/{len(FIGURES)} figures failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(FIGURES)} simulation figures rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
