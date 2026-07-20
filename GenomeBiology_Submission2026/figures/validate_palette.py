#!/usr/bin/env python
"""
Check that the LeafletFA palette survives colorblindness and greyscale printing.

A palette is only usable for a manuscript if every pair of colors that can appear in the
SAME panel stays distinguishable for readers with deuteranopia (~6% of men) and protanopia
(~2%), and if the figure still reads when printed in greyscale.

Run:  python validate_palette.py
Exits non-zero if any pair fails, so it can gate a figure build.

Method: Viénot, Brettel & Mollon (1999) dichromat simulation in LMS space, then CIE76 ΔE
in CIELAB. Thresholds are deliberately conservative for small marks at 6 pt.
"""

from __future__ import annotations
import itertools
import sys

import numpy as np

# Minimum CIE76 ΔE between any two colors used in one panel.
# ~10 is "clearly different at a glance"; small/thin marks need more headroom than swatches.
MIN_DE_NORMAL = 20.0
MIN_DE_CVD = 15.0
# Minimum CIELAB L* gap so the panel survives greyscale printing.
MIN_DL_GREY = 8.0


# ── color conversions ───────────────────────────────────────────────────────────────────
def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return m @ rgb


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def hex_to_lab(h: str) -> np.ndarray:
    return xyz_to_lab(linear_to_xyz(srgb_to_linear(hex_to_rgb(h))))


# ── dichromat simulation (Viénot et al. 1999) ───────────────────────────────────────────
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.02996, 0.18431, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)

_SIM = {
    "deuteranopia": np.array([[1.0, 0.0, 0.0],
                              [0.494207, 0.0, 1.24827],
                              [0.0, 0.0, 1.0]]),
    "protanopia":   np.array([[0.0, 2.02344, -2.52581],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]]),
    "tritanopia":   np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [-0.395913, 0.801109, 0.0]]),
}


def simulate_cvd(h: str, kind: str) -> np.ndarray:
    """Return CIELAB of `h` as seen with `kind`."""
    lin = srgb_to_linear(hex_to_rgb(h))
    sim_lin = _LMS2RGB @ (_SIM[kind] @ (_RGB2LMS @ lin))
    return xyz_to_lab(linear_to_xyz(np.clip(sim_lin, 0, 1)))


def de76(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── the palette under test ──────────────────────────────────────────────────────────────
def check(palette: dict[str, str], verbose: bool = True) -> bool:
    keys = list(palette)
    rows, ok = [], True

    for a, b in itertools.combinations(keys, 2):
        ha, hb = palette[a], palette[b]
        d_norm = de76(hex_to_lab(ha), hex_to_lab(hb))
        d_deut = de76(simulate_cvd(ha, "deuteranopia"), simulate_cvd(hb, "deuteranopia"))
        d_prot = de76(simulate_cvd(ha, "protanopia"), simulate_cvd(hb, "protanopia"))
        d_grey = abs(hex_to_lab(ha)[0] - hex_to_lab(hb)[0])

        fails = []
        if d_norm < MIN_DE_NORMAL:
            fails.append("normal")
        if d_deut < MIN_DE_CVD:
            fails.append("deut")
        if d_prot < MIN_DE_CVD:
            fails.append("prot")
        if d_grey < MIN_DL_GREY:
            fails.append("grey")
        if fails:
            ok = False
        rows.append((f"{a}/{b}", d_norm, d_deut, d_prot, d_grey, ",".join(fails) or "ok"))

    if verbose:
        print(f"{'pair':<28}{'ΔE':>7}{'deut':>7}{'prot':>7}{'ΔL*':>7}  status")
        print("-" * 70)
        for name, dn, dd, dp, dg, status in sorted(rows, key=lambda r: r[1]):
            flag = "" if status == "ok" else "  <-- FAIL"
            print(f"{name:<28}{dn:7.1f}{dd:7.1f}{dp:7.1f}{dg:7.1f}  {status}{flag}")
        print("-" * 70)
        print(f"thresholds: ΔE≥{MIN_DE_NORMAL} normal, ≥{MIN_DE_CVD} CVD, ΔL*≥{MIN_DL_GREY} greyscale")
        print("PASS" if ok else "FAIL")
    return ok


def proof_sheet(palette: dict[str, str], out: str = "palette_reference"):
    """Render the palette as normal / deuteranopia / protanopia / greyscale swatch rows,
    so the numbers above can be eyeballed. Numbers catch what eyes miss and vice versa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from leaflet_theme import apply_theme, save, PT

    apply_theme()
    rows = [("normal vision", lambda h: hex_to_rgb(h)),
            ("deuteranopia", lambda h: lab_to_rgb(simulate_cvd(h, "deuteranopia"))),
            ("protanopia", lambda h: lab_to_rgb(simulate_cvd(h, "protanopia"))),
            ("greyscale", lambda h: np.repeat(hex_to_lab(h)[0] / 100.0, 3))]

    keys = list(palette)
    fig, axes = plt.subplots(len(rows), 1, figsize=(len(keys) * 0.95, len(rows) * 0.75))
    for ax, (title, fn) in zip(axes, rows):
        for i, k in enumerate(keys):
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=np.clip(fn(palette[k]), 0, 1)))
        ax.set_xlim(0, len(keys)); ax.set_ylim(0, 1)
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=PT["tick"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for i, k in enumerate(keys):
        axes[-1].text(i + 0.46, -0.35, f"{k}\n{palette[k]}", ha="center", va="top",
                      fontsize=PT["annot"] - 1)
    return save(fig, out, close=True)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB -> sRGB, for rendering the simulated swatches."""
    L, a, b = lab
    fy = (L + 16) / 116
    f = lambda t: t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)
    xyz = np.array([0.95047 * f(fy + a / 500), f(fy), 1.08883 * f(fy - b / 200)])
    m = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]]))
    lin = np.clip(m @ xyz, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


if __name__ == "__main__":
    sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
    from leaflet_theme import PALETTE, MAX_SERIES

    # `grey` is excluded by design: it marks non-focal context and never encodes a series
    # that a reader must tell apart from another series. Every OTHER color must pass.
    series = {k: v for k, v in PALETTE.items() if k != "grey"}
    assert len(series) == MAX_SERIES, (
        f"PALETTE has {len(series)} series colors but MAX_SERIES={MAX_SERIES}; "
        "adding a sixth breaks greyscale/dichromat separation — facet instead."
    )
    print(f"LeafletFA palette — {len(series)} series colors (grey excluded: non-series)\n")
    ok = check(series)
    if "--sheet" in sys.argv:
        print("\nwrote:", *[str(p) for p in proof_sheet(PALETTE)])
    sys.exit(0 if ok else 1)
