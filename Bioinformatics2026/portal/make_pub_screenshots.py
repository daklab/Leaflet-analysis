"""Cropped, high-resolution screenshots of the Gria2 event page (Allen) as candidates for a figure or supplementary panel.

    python make_pub_screenshots.py [--site SITE] [--out DIR]     # sbatch, as make_screenshots.py (needs libgbm on LD_LIBRARY_PATH)

Writes two PNGs, light theme, at 4 device pixels per CSS pixel (>= 600 dpi at 183 mm; the report gives the exact dpi):
  gria2_allen_structure_panel.png          gene structure + per-cell-group panel, all cell groups, grouped by class
  gria2_allen_structure_panel_tooltip.png  gene structure + the first rows of the panel + one tooltip on a row that has an interval
Only display is changed for the crop: the junction table and the controls are hidden, and the panel's scroll box is expanded.
Prints the crop width and the size of the smallest text (9 px tick labels) at 183 mm, to compare with the 5 pt floor.
"""
import argparse, functools, http.server, json, os, socketserver, threading
from playwright.sync_api import sync_playwright

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser"
GRIA2 = "ENSMUSG00000033981.14_atse_3"
WIDTH_MM, DPI, MIN_PX = 183, 600, 9           # smallest font in the event page: axis tick labels, 9 px
ap = argparse.ArgumentParser()
ap.add_argument("--site", default=f"{V}/site"); ap.add_argument("--out", default=f"{V}/screenshots/2026-11-09_pub")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)


class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x):
        pass


srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Q, directory=a.site))
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/"
VIEW_W = 1440
SCALE = 4                                     # device pixels per CSS pixel; the report gives the resulting dpi at 183 mm
HIDE = """() => {
  document.querySelectorAll('header, nav, footer, .controls, #footer').forEach(e => e.style.display = 'none');
  const t = document.querySelector('main table:not(.dots)'); if (t) t.closest('.wrap') ? t.closest('.wrap').style.display = 'none' : t.style.display = 'none';
  document.querySelectorAll('.dotwrap').forEach(e => { e.style.maxHeight = 'none'; e.style.overflow = 'visible'; });
}"""
report = {}


def union(pg, sels):
    boxes = [pg.locator(s).first.bounding_box() for s in sels]
    boxes = [b for b in boxes if b]
    x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["width"] for b in boxes); y1 = max(b["y"] + b["height"] for b in boxes)
    return dict(x=x0 - 6, y=y0 - 6, width=x1 - x0 + 12, height=y1 - y0 + 12)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport=dict(width=VIEW_W, height=1000), device_scale_factor=SCALE, color_scheme="light")
    pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + f"#/event/Allen/{GRIA2}"); pg.wait_for_selector("table.dots"); pg.wait_for_timeout(500)
    pg.evaluate(HIDE); pg.wait_for_timeout(200)
    # 1: structure + full panel
    clip = union(pg, ["svg.structure", "table.dots"])
    pg.set_viewport_size(dict(width=VIEW_W, height=int(clip["y"] + clip["height"] + 40)))
    clip = union(pg, ["svg.structure", "table.dots"])
    f1 = os.path.join(a.out, "gria2_allen_structure_panel.png")
    pg.screenshot(path=f1, clip=clip, full_page=True)
    report["structure_panel"] = dict(file=f1, crop_css_px=[round(clip["width"]), round(clip["height"])])
    # 2: tooltip on the first cell (excitatory group) whose type has an interval at this ATSE
    tds = pg.locator("table.dots td[data-t]"); n = tds.count(); hit = None
    for i in range(n):
        td = tds.nth(i); td.hover(); pg.wait_for_timeout(40)
        tip = pg.inner_text("#tip")
        if "no interval" not in tip and "[" in tip and int(td.get_attribute("data-k")) == 2:   # J3, as in the review screenshot
            hit = (td, tip); break
    assert hit, "no cell with an interval found"
    td, tip = hit
    rows = pg.locator("table.dots tbody tr")
    last = min(rows.count() - 1, 22)
    tb = pg.locator("#tip").bounding_box(); rb = rows.nth(last).bounding_box(); sb = pg.locator("svg.structure").bounding_box()
    y1 = max(tb["y"] + tb["height"], rb["y"] + rb["height"])
    x1 = max(pg.locator("table.dots").bounding_box()["x"] + pg.locator("table.dots").bounding_box()["width"], tb["x"] + tb["width"], sb["x"] + sb["width"])
    clip2 = dict(x=sb["x"] - 6, y=sb["y"] - 6, width=x1 - sb["x"] + 12, height=y1 - sb["y"] + 12)
    f2 = os.path.join(a.out, "gria2_allen_structure_panel_tooltip.png")
    pg.screenshot(path=f2, clip=clip2, full_page=True)
    report["structure_panel_tooltip"] = dict(file=f2, crop_css_px=[round(clip2["width"]), round(clip2["height"])], tooltip=tip)
    b.close()

for k in ("structure_panel", "structure_panel_tooltip"):
    w = report[k]["crop_css_px"][0]
    mm_per_px = WIDTH_MM / w
    report[k]["at_183mm"] = dict(dpi=round(w * SCALE / (WIDTH_MM / 25.4)), height_mm=round(report[k]["crop_css_px"][1] * mm_per_px, 1),
                                 smallest_text_pt=round(MIN_PX * mm_per_px * 72 / 25.4, 2), body_12px_pt=round(12 * mm_per_px * 72 / 25.4, 2))
report["js_errors"] = errs
json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
