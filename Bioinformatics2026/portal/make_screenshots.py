"""Drive the built site in headless Chromium: screenshots of each view plus functional checks. Prints every check; exits non-zero on failure.

    python make_screenshots.py [--site SITE] [--out DIR]
(Chromium may need libgbm.so.1 on LD_LIBRARY_PATH on cluster nodes.)
"""
import argparse, http.server, json, os, functools, socketserver, sys, threading, urllib.parse
from playwright.sync_api import sync_playwright

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser"
ap = argparse.ArgumentParser()
ap.add_argument("--site", default=f"{V}/site"); ap.add_argument("--out", default=f"{V}/screenshots")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x):
        pass


srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Q, directory=a.site))
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/"
enc = urllib.parse.quote
NOTE = "Ranking for exploration; LeafletFA's cell-type specificity calls did not pass validation (see paper)."
IV = "95 % predictive interval for the pseudobulk ψ of this type's held-out cells"   # check 12 (a)+(b); full label in web/config.js
GRIA2 = "ENSMUSG00000033981.14_atse_3"
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport=dict(width=1440, height=950), accept_downloads=True, device_scale_factor=1)
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    bad = []
    pg.on("response", lambda r: bad.append(r.url) if r.status >= 400 else None)

    def go(h, sel):
        pg.goto("about:blank"); pg.goto(BASE + "index.html" + h); pg.wait_for_selector(sel, timeout=60000); pg.wait_for_timeout(150)   # full reload: no stale view

    def shot(name, full=True):
        pg.screenshot(path=os.path.join(a.out, name + ".png"), full_page=full)

    def text():
        return pg.inner_text("body")

    # landing
    go("#/", ".card"); shot("01_landing")
    I = json.loads(pg.evaluate("JSON.stringify(INDEX.featured)"))
    for f in I:
        for e in f["events"]:
            go(f"#/event/{f['dataset']}/{enc(e, safe='')}", "table.dots")
            check(f"featured link {f['gene']} {f['dataset']} {e}", f["gene"] in text() and e in text())
    # search
    go("#/", ".card"); pg.fill("#q", "gria2"); pg.wait_for_selector("#q-results a[href]"); shot("02_search_gria2", full=False)
    check("search finds Gria2", "Gria2" in pg.inner_text("#q-results"))
    pg.press("#q", "Enter"); pg.wait_for_selector("h1 .gene"); shot("03_gene_gria2")
    check("gene page lists atse_3", GRIA2 in text())
    pg.fill("#q", GRIA2); pg.wait_for_selector("#q-results a[href]"); check("search by ATSE id", GRIA2 in pg.inner_text("#q-results"))
    # event pages
    for ds in ("Allen", "TMS"):
        go(f"#/event/{ds}/{GRIA2}", "table.dots"); shot(f"04_event_gria2_{ds}")
        check(f"event {ds}: interval label", IV in text())
        check(f"event {ds}: 7 panels", pg.locator("th.panel").count() == 7, str(pg.locator("th.panel").count()))
        check(f"event {ds}: structure", pg.locator("svg.structure path").count() == 7 and pg.locator("svg.structure rect").count() >= 4)
    pg.select_option("#ev-group", "group2"); pg.wait_for_timeout(200); check("event group switch", pg.locator("tr.group").count() > 3)
    with pg.expect_download() as d:
        pg.click("#ev-dl")
    t = open(d.value.path()).read().splitlines()
    print("\n".join(t[:10])); check("event TSV header", t[0].startswith("# LeafletFA") and any("fit.npz md5" in x for x in t[:9]) and any(IV in x for x in t[:9]) and any("intervals md5" in x for x in t[:9]))
    hd = [x for x in t if not x.startswith("#")]; cols = hd[0].split("\t"); body = [x.split("\t") for x in hd[1:]]
    check("event TSV has interval_status, test_reads, test_cells_with_reads", all(c in cols for c in ("interval_status", "test_reads", "test_cells_with_reads")))
    si, li, tri, tci = cols.index("interval_status"), cols.index("lo"), cols.index("test_reads"), cols.index("test_cells_with_reads")
    ok_rows = [r for r in body if r[si] == "check 12 (a)+(b)"]; na_rows = [r for r in body if r[si] != "check 12 (a)+(b)"]
    check("event TSV: interval present exactly where TEST >= 100 reads and >= 10 cells",
          all(r[li] != "NA" and int(r[tri]) >= 100 and int(r[tci]) >= 10 for r in ok_rows) and all(r[li] == "NA" and (int(r[tri]) < 100 or int(r[tci]) < 10) for r in na_rows),
          f"{len(ok_rows)} with, {len(na_rows)} without")
    # hover
    go(f"#/event/Allen/{GRIA2}", "table.dots"); pg.hover("td[data-t='5'][data-k='2']"); pg.wait_for_timeout(100)
    check("dot tooltip", "LeafletFA" in pg.inner_text("#tip") and "held-out (TEST)" in pg.inner_text("#tip")); pg.screenshot(path=os.path.join(a.out, "04b_event_tooltip.png"))
    pg.set_viewport_size(dict(width=1440, height=950))
    # grey type
    go("#/event/TMS/ENSMUSG00000030849.18_atse_3", "table.dots")
    rule = pg.evaluate("CONFIG.GREY_RULE"); ng = pg.locator("tr.grey").count()
    check(f"grey rows on event page follow GREY_RULE={rule}", ng == 0 if rule == "n_mice_lt2" else ng >= 1, str(ng))
    # compare Allen excitatory vs inhibitory
    A, B = "L2/3 IT CTX", "Pvalb"
    go(f"#/compare/Allen/{enc(A, safe='')}/{enc(B, safe='')}", "#cmp-table table"); shot("05_compare_allen_excitatory_vs_inhibitory", full=False)
    check("compare note visible", NOTE in text())
    check("compare SpliceVI column", "SpliceVI DS call" in text() and "from SpliceVI cell-type DS runs; not yet filled" in text())
    n0 = pg.inner_text(".pager b")
    pg.fill("#f-reads", "200"); pg.wait_for_timeout(600); n1 = pg.inner_text(".pager b")
    check("read filter narrows the table", int(n1.replace(",", "")) < int(n0.replace(",", "")), f"{n0} -> {n1}")
    pg.fill("#f-reads", "20"); pg.wait_for_timeout(500)
    with pg.expect_download() as d:
        pg.click("#dl")
    t = open(d.value.path()).read().splitlines()
    print("\n".join(t[:12])); check("compare TSV rows", len([x for x in t if not x.startswith("#")]) == int(n0.replace(",", "")) + 1)
    pg.fill("#f-gene", "gria2"); pg.wait_for_timeout(500); shot("06_compare_allen_filtered_gria2", full=False)
    check("compare gene filter", "Gria2" in pg.inner_text("#cmp-table"))
    pg.click("#cmp-table td a.mono"); pg.wait_for_selector("table.dots"); check("compare row links to event, type highlighted", pg.locator("tr.hl").count() >= 1, str(pg.locator('tr.hl').count()))
    go("#/compare/TMS", "#sel-a"); check("compare note on empty TMS", NOTE in text())
    pg.select_option("#sel-a", "Brain_Non-Myeloid_NEURON"); pg.wait_for_timeout(200)
    pg.select_option("#sel-b", "Mammary_Gland_BASAL CELL"); pg.wait_for_selector("#cmp-table table", timeout=60000); shot("07_compare_tms_grey_type", full=False)
    check(f"grey banner follows GREY_RULE={rule}", ("Grey:" in text()) == (rule != "n_mice_lt2"))
    check("no '(grey)' in the type dropdowns" if rule == "n_mice_lt2" else "dropdown marks grey types", ("(grey)" in pg.inner_html("#sel-b")) == (rule != "n_mice_lt2"))
    # one vs all
    go(f"#/one/Allen/{enc(B, safe='')}", "#one-table table"); shot("08_one_vs_all_allen_pvalb", full=False)
    check("one-vs-all note", NOTE in text()); check("one-vs-all SpliceVI column", "SpliceVI DS call" in text())
    check("one-vs-all: no Overlap column (rest has no validated interval)", pg.locator("#one-table th", has_text="Overlap").count() == 0)
    pg.fill("#f-gene", "gria2"); pg.wait_for_timeout(500); shot("09_one_vs_all_filtered_gria2", full=False)
    check("one-vs-all gene filter", "Gria2" in pg.inner_text("#one-table"))
    with pg.expect_download() as d:
        pg.click("#dl")
    print("\n".join(open(d.value.path()).read().splitlines()[:11]))
    # phone width
    pg.set_viewport_size(dict(width=390, height=800)); go(f"#/event/Allen/{GRIA2}", "table.dots"); shot("10_phone_event", full=False)
    check("no horizontal page scroll at phone width", pg.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), str(pg.evaluate("document.documentElement.scrollWidth")))
    # forbidden words in the rendered text across views
    pg.set_viewport_size(dict(width=1440, height=950))
    for h, sel in (("#/", ".card"), (f"#/event/Allen/{GRIA2}", "table.dots"), (f"#/compare/Allen/{enc(A, safe='')}/{enc(B, safe='')}", "#cmp-table table"), (f"#/one/Allen/{enc(B, safe='')}", "#one-table table")):
        go(h, sel); low = text().lower().replace("cell-type specificity calls", "")
        check(f"no forbidden wording {h}", not any(w in low for w in ("specific", "significant", "differentially spliced", "★")))
    check("no JS errors", not errs, "; ".join(errs[:5])); check("no failed requests", not bad, "; ".join(bad[:5]))
    # dark mode: junction colors switch to the dark set
    pg.emulate_media(color_scheme="dark")
    go(f"#/event/Allen/{GRIA2}", "table.dots"); shot("11_event_gria2_Allen_dark")
    used = pg.evaluate("Array.from(document.querySelectorAll('svg.structure path, svg.structure line')).map(e => e.getAttribute('stroke')).filter(Boolean)")
    dark = pg.evaluate("CONFIG.JUNCTION_COLORS_DARK"); light = pg.evaluate("CONFIG.JUNCTION_COLORS")
    check("dark mode uses the dark junction colors", any(c in dark for c in used) and not any(c in light and c not in dark for c in used), ",".join(sorted(set(used))[:8]))
    go(f"#/compare/Allen/{enc(A, safe='')}/{enc(B, safe='')}", "#cmp-table table"); shot("12_compare_allen_dark", full=False)
    pg.emulate_media(color_scheme="light")
    b.close()
srv.shutdown()
print("FAILURES:", fails); sys.exit(1 if fails else 0)
