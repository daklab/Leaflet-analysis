#!/usr/bin/env python3
"""Build the validation candidate figures (panels/fig_appnote_validation.py) without changing make_figures.py or
paper_config.yaml: the figure entries are added to the loaded config at run time and built with make_figures.build_panel.
    python build_validation_figures.py
Outputs under <output_dir>: figAN_b17/, figAN_b18/ (figure AN panel b candidates, also copied next to figAN as
figAN_b_candidate_check17/18.{pdf,png}), figS_ds_validation/, figS_simulation/, figS_scaling/ (panels/<id>.pdf, _review.png,
source_data/)."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "panels"))
import make_figures as M  # noqa: E402
from style import apply_style, load_config  # noqa: E402

FIGS = {"_b17": ("ANb17_b", 85, 50), "_b18": ("ANb18_b", 85, 50), "S_ds_validation": ("Sds_a", 183, 120), "S_simulation": ("Ssim_a", 183, 112),
        "S_scaling": ("Sscale_a", 183, 58)}
DIRS = {"_b17": "figAN_b17", "_b18": "figAN_b18", "S_ds_validation": "figS_ds_validation", "S_simulation": "figS_simulation", "S_scaling": "figS_scaling"}


def main():
    cfg = load_config(); apply_style(cfg)
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    print("[FIG] font resolves to", fm.findfont(fm.FontProperties(family=plt.rcParams["font.family"])))
    for fid, (pid, w, h) in FIGS.items():
        key = "AN" + fid if fid.startswith("_b") else fid          # fig<key> is the output folder name in make_figures
        M.FIGURE_MODULES[key] = "fig_appnote_validation"
        cfg["figures"][key] = dict(title="validation candidate", width="appnote", formats=["pdf", "png"], panels={pid: dict(w=w, h=h)})
        p = M.build_panel(cfg, key, pid); print("[FIG]", p)
        out = os.path.join(cfg["output_dir"], f"fig{key}")
        if out != os.path.join(cfg["output_dir"], DIRS[fid]):
            raise SystemExit(f"unexpected output folder {out}")
    for k, pid, chk in (("b17", "ANb17_b", "check17"), ("b18", "ANb18_b", "check18")):
        src = os.path.join(cfg["output_dir"], f"figAN_{k}", "panels", pid)
        dst = os.path.join(cfg["output_dir"], "figAN", f"figAN_b_candidate_{chk}")
        shutil.copy(src + ".pdf", dst + ".pdf"); shutil.copy(src + "_review.png", dst + "_review.png")
        print("[FIG] copied", dst + ".pdf")


if __name__ == "__main__":
    main()
