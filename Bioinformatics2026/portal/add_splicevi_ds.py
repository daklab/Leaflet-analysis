"""Fill the "SpliceVI DS call" column of the compare views from a data file. No rebuild of the shards.

    python add_splicevi_ds.py CALLS.tsv --source "SpliceVI cell-type DS runs, <run id>" [--site SITE]

CALLS.tsv: tab separated with header; columns dataset, cell_type_a, cell_type_b, event_id, call.
- dataset: TMS or Allen. cell_type_a, cell_type_b: names exactly as in index.json `name`, in either order.
- cell_type_b = "rest" fills the one-vs-all view for cell_type_a.
- call: free text shown verbatim in the column and in the TSV downloads.
Writes SITE/data/<dataset>/splicevi_ds/<i>__<j>.json (i < j; or <i>__rest.json), {event_id: call}, and sets
datasets.<dataset>.splicevi_ds = {source} in index.json. Re-running replaces the earlier fill for that dataset.
"""
import argparse, json, os, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("calls")
ap.add_argument("--source", required=True)
ap.add_argument("--site", default="/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser/site")
a = ap.parse_args()
X = pd.read_csv(a.calls, sep="\t", dtype=str)
assert {"dataset", "cell_type_a", "cell_type_b", "event_id", "call"} <= set(X.columns)
ip = os.path.join(a.site, "data", "index.json")
I = json.load(open(ip)); ids = set(json.load(open(os.path.join(a.site, "data", "catalog", "events.json")))["ids"])
bad = set(X.event_id) - ids
assert not bad, f"event ids not in the catalog: {sorted(bad)[:5]}"
for ds, D in X.groupby("dataset"):
    code = {t["name"]: t["i"] for t in I["datasets"][ds]["types"]}
    out = os.path.join(a.site, "data", ds, "splicevi_ds")
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    unknown = (set(D.cell_type_a) | set(D.cell_type_b)) - set(code) - {"rest"}
    assert not unknown, f"{ds}: cell types not in the site: {sorted(unknown)[:5]}"
    for (x, y), G in D.groupby(["cell_type_a", "cell_type_b"]):
        key = f"{code[x]}__rest" if y == "rest" else "__".join(str(k) for k in sorted((code[x], code[y])))
        p = os.path.join(out, key + ".json"); m = json.load(open(p)) if os.path.exists(p) else {}
        m.update(dict(zip(G.event_id, G["call"])))
        json.dump(m, open(p, "w"), separators=(",", ":"))
    I["datasets"][ds]["splicevi_ds"] = dict(source=a.source)
    print(ds, len(D), "rows")
json.dump(I, open(ip, "w"), separators=(",", ":"))
