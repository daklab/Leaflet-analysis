#!/usr/bin/env python
"""Render the SIMULATION half of Figure 2 (panels 2B-D) and its supplements.

    python code/build_sim.py

Panels come from `panels.py`, which is the shared library for both halves -- the two builders are
separate entry points, not separate copies of the code. `figure_validation` is the assembled main
Figure 2 and deliberately spans both halves (Figure 2 *is* simulation + imputation), so it is
rendered by build_imputation.py, which is the one that depends on the model runs.

Everything here reads only ../data/ and writes only ../plots/ -- no model directory, no GPU.
"""
import subprocess, sys, os
import panels as P


def _verify_inputs() -> None:
    """Warn if data/ no longer matches data/INPUTS.json.

    Not fatal -- regenerating an input legitimately changes its hash, and the manifest is then
    refreshed by re-running collect_inputs.py. But a silent mismatch means the figure was built from
    something other than what is recorded, which is exactly what made this half unreproducible.
    """
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "collect_inputs.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("WARNING: inputs do not match data/INPUTS.json --")
        print("".join(f"    {l}\n" for l in r.stdout.strip().splitlines()))
        print("    re-run code/collect_inputs.py if this is expected.\n")

FIGURES = (
    ("FIG2_simulation",              P.figure_simulation),
    ("FIG2_supp_draft",              P.figure_sim_supp),
    ("FIG2_supp_seed_consistency_sim", P.figure_seed_consistency_sim),
)


def main() -> int:
    _verify_inputs()
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
