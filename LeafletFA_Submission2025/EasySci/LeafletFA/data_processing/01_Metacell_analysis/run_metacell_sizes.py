#!/usr/bin/env python3
"""
Run metacell analysis for each TARGET_METACELL_SIZE (100, 500, 1000, 2000, 4000)
by calling 00_MetaCellAnalysis.py with --size and --suffix. All print output
appears in the terminal.

Usage:
  cd .../01_Metacell_analysis
  python run_metacell_sizes.py

  # Or custom sizes (space-separated):
  python run_metacell_sizes.py 100 500 2000
"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PY_SCRIPT = SCRIPT_DIR / "00_MetaCellAnalysis.py"

# Default: run on 100, 500, 1000, 2000, 4000
SIZES = [100, 500, 1000, 2000, 4000] if len(sys.argv) <= 1 else [int(x) for x in sys.argv[1:]]


def main():
    if not PY_SCRIPT.exists():
        print(f"Script not found: {PY_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    print(f"Sizes to run: {SIZES}", flush=True)
    for i, size in enumerate(SIZES):
        suffix = f"_{size}"
        print(f"\n{'='*60}\n[{i+1}/{len(SIZES)}] TARGET_METACELL_SIZE={size} METACELL_RUN_SUFFIX={suffix!r}\n{'='*60}", flush=True)
        r = subprocess.run(
            [sys.executable, str(PY_SCRIPT), "--size", str(size), "--suffix", suffix],
            cwd=str(SCRIPT_DIR),
        )
        if r.returncode != 0:
            print(f"Failed size {size} with return code {r.returncode}", file=sys.stderr)
            sys.exit(r.returncode)
    print("\nAll runs finished.", flush=True)


if __name__ == "__main__":
    main()
