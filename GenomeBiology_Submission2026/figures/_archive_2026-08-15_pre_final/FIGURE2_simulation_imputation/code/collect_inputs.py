#!/usr/bin/env python
"""Consolidate every Figure 2 input into `data/`, so the figures build from the repo alone.

    python code/collect_inputs.py [--check]

Three inputs used to be read from outside the repo, by absolute path, and none of them was tracked:

    $WD/SIMULATION/A15_seed_consistency.csv                       seed-consistency panel
    $WD/SIMULATION/A26_cellcell.npz                    (94 MB)    cell-cell consistency panel
    ~/LeafletFA/experiments/minibatch/sim_recover10_Kmodel{10,20,30}.csv   minibatch-vs-full panel

The figures built only because those paths happened to exist on this filesystem. A fresh clone, or
this filesystem after a cleanup, could not reproduce the simulation half at all.

This script copies them in and writes `data/INPUTS.json` recording, for each, the source path, size,
mtime and sha256 — so a later mismatch is detectable rather than silent.

A26 is reduced rather than copied. The panel renders `n_show=400` cells and only ever displays
seed0 / seed1 / true, so 3 of the 12 stored 1500x1500 matrices are never read and the rest are
displayed at 7% of their stored resolution. Keeping 600x600 (headroom above the 400 shown) for the
9 matrices actually used takes it from 94 MB to 11 MB with no visible change to the figure. The
full-resolution file stays in $WD as the regenerable source of truth.

--check verifies what is in data/ still matches the manifest, and exits non-zero if not. Use it
before rebuilding figures for submission.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
import numpy as np

_CODE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(os.path.dirname(_CODE), "data")

WD_SIM   = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
RECOVER  = "/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"

# Cells kept per A26 similarity matrix. The panel shows 400; the extra headroom means n_show can be
# raised to 600 without regenerating. Above that, re-run A26_cellcell_consistency.py.
A26_KEEP = 600

COPY = [
    (f"{WD_SIM}/A15_seed_consistency.csv", "A15_seed_consistency.csv",
     "pipelines/09_simulation — seed-consistency across simulated fits"),
    (f"{RECOVER}/sim_recover10_Kmodel10.csv", "sim_recover10_Kmodel10.csv",
     "K_true=10 simulation, model K=10 (minibatch vs full-batch)"),
    (f"{RECOVER}/sim_recover10_Kmodel20.csv", "sim_recover10_Kmodel20.csv",
     "K_true=10 simulation, model K=20"),
    (f"{RECOVER}/sim_recover10_Kmodel30.csv", "sim_recover10_Kmodel30.csv",
     "K_true=10 simulation, model K=30"),
]


def sha256(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path: str, source: str, note: str) -> dict:
    st = os.stat(path)
    return dict(source=source, note=note, bytes=st.st_size,
                mtime=int(st.st_mtime), sha256=sha256(path))


def slim_a26(src: str, dst: str) -> None:
    """Keep only the matrices the panel reads, at the resolution it displays them.

    `order_{tag}` is NOT a permutation -- it is the true program label per cell (0..K-1), so it is
    heavily tied and `np.argsort` on it is what groups cells into blocks. Subsetting it and
    re-ranking would turn tied labels into distinct ranks and silently reorder the heatmap.

    So the matrices are stored ALREADY IN DISPLAY ORDER and `order_{tag}` becomes `arange(n)`. The
    panel's `np.argsort(order)[:n_show]` then evaluates to `0..n_show-1`, selecting the same cells
    in the same order as it would have from the full file -- and, because the tie-breaking is baked
    in here rather than recomputed, it stays identical regardless of numpy's sort behaviour.
    """
    z = np.load(src, allow_pickle=True)
    out = {}
    for tag in ("binom", "bb20", "bb2"):
        if f"S_{tag}_seed0" not in z.files:
            continue
        order = np.asarray(z[f"order_{tag}"])
        disp = np.argsort(order)[:A26_KEEP]          # exactly the rows the panel would draw
        out[f"order_{tag}"] = np.arange(len(disp))   # -> the panel's argsort is now the identity
        for k in ("seed0", "seed1", "true"):         # seed2 is stored upstream but never displayed
            out[f"S_{tag}_{k}"] = np.asarray(z[f"S_{tag}_{k}"])[np.ix_(disp, disp)].astype(np.float32)
    if "summary" in z.files:
        out["summary"] = z["summary"]                # the reported statistics, computed over ALL seeds
    np.savez_compressed(dst, **out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify data/ matches INPUTS.json; do not copy anything")
    args = ap.parse_args()
    manifest_path = os.path.join(DATA, "INPUTS.json")

    if args.check:
        if not os.path.exists(manifest_path):
            print(f"no manifest at {manifest_path} — run collect_inputs.py first")
            return 1
        man = json.load(open(manifest_path))
        bad = []
        for name, rec in man["files"].items():
            p = os.path.join(DATA, name)
            if not os.path.exists(p):
                bad.append(f"{name}: MISSING")
            elif sha256(p) != rec["sha256"]:
                bad.append(f"{name}: sha256 differs from manifest")
        for b in bad:
            print(f"  {b}")
        print(f"{len(man['files']) - len(bad)}/{len(man['files'])} inputs verified")
        return 1 if bad else 0

    os.makedirs(DATA, exist_ok=True)
    files = {}
    for src, name, note in COPY:
        if not os.path.exists(src):
            print(f"  MISSING SOURCE, skipping: {src}")
            continue
        dst = os.path.join(DATA, name)
        shutil.copy2(src, dst)
        files[name] = stamp(dst, src, note)
        print(f"  copied  {name}  ({files[name]['bytes']:,} B)")

    src = f"{WD_SIM}/A26_cellcell.npz"
    if os.path.exists(src):
        dst = os.path.join(DATA, "A26_cellcell_fig.npz")
        slim_a26(src, dst)
        files["A26_cellcell_fig.npz"] = stamp(
            dst, src,
            f"reduced for the figure: {A26_KEEP}x{A26_KEEP} cells, seed0/seed1/true only "
            f"(panel shows 400 and never displays seed2); full file stays in $WD")
        print(f"  reduced A26_cellcell_fig.npz  "
              f"({os.path.getsize(src)/1e6:.0f} MB -> {files['A26_cellcell_fig.npz']['bytes']/1e6:.0f} MB)")
    else:
        print(f"  MISSING SOURCE, skipping: {src}")

    # already-local inputs, recorded so the manifest describes every file the build reads
    for name, note in (("benchmark_results.csv", "frac sweep, from step3_assemble.sh"),
                       ("benchmark_effect_sweep.csv", "effect sweep, from eff3_assemble.sh"),
                       ("panel_data.npz", "from prep_panels.py"),
                       ("representative_fit.npz", "from prep_panels.py")):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            files[name] = stamp(p, "(generated in place)", note)

    json.dump(dict(generated_by="code/collect_inputs.py", a26_cells_kept=A26_KEEP, files=files),
              open(manifest_path, "w"), indent=2, sort_keys=True)
    print(f"\nwrote {manifest_path}  ({len(files)} inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
