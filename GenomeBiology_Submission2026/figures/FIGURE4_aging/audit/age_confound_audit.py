#!/usr/bin/env python
"""Fig 4 confound audit: for the cells each panel uses, is age crossed with plate / lane / campaign / sex?
Plate = the B###### token in the TMS cell ID; lane-like token = D###### (old-format IDs only);
campaign = ID format (3m cohort 'A1-B000126-3_39_F-1-1' vs 18/24m cohort 'P9_D045342_B009019_S369');
batch = obs['batch']. Writes crosstabs + a per-panel summary."""
import re, numpy as np, pandas as pd, h5py
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
    df = pd.DataFrame({k: cat(f["obs"][k]) for k in ["cell_id", "dataset", "age", "sex", "mouse.id", "tissue_cell_type", "batch"]})
df = df[df.dataset == "tabula_muris_senis"].copy()
# three ID formats, one per sequencing campaign:
#   3m : 'E3-D042105-3_11_M-1-1'       well-PLATE-mouse...
#   18m: 'K21_B002333_B009884_S261'    well_PLATE_index_S#
#   24m: 'K20_B001732_S20_L004'        well_PLATE_S#_LANE
tok = df.cell_id.str.split(r"[-_]", regex=True)
df["plate"] = tok.str[1]
df["lane"] = df.cell_id.str.extract(r"_(L\d{3})$")[0].fillna("no-lane-in-ID")
df["campaign"] = np.select([df.cell_id.str.match(r"^[A-P]\d+-[A-Z]+\d+-"),
                            df.cell_id.str.match(r"^[A-P]\d+_[A-Z]+\d+_[A-Z]+\d+_S\d+$"),
                            df.cell_id.str.match(r"^[A-P]\d+_[A-Z]+\d+_S\d+_L\d{3}$")],
                           ["C1 (dash IDs)", "C2 (_index_S IDs)", "C3 (_S_L IDs)"], "unparsed")
print("unparsed plate:", df.plate.isna().sum(), "| campaign counts:", df.campaign.value_counts().to_dict())
def audit(name, d, ages):
    d = d[d.age.isin(ages)]
    pa = d.groupby("plate").age.nunique(); la = d.groupby("lane").age.nunique()
    r = dict(panel=name, ages="/".join(ages), cells=len(d), mice=d["mouse.id"].nunique(), plates=d.plate.nunique(),
             plates_with_2plus_ages=int((pa > 1).sum()), lanes=d.lane.nunique(), lanes_with_2plus_ages=int((la > 1).sum()),
             age_x_campaign=pd.crosstab(d.age, d.campaign).to_dict(), age_x_sex=pd.crosstab(d.age, d.sex).to_dict(),
             batches_with_2plus_ages=int((d.groupby("batch").age.nunique() > 1).sum()), batches=d.batch.nunique())
    return r
rows = []
allT = df
rows.append(audit("a variance partition (all TMS, continuous age)", allT, ["3m", "18m", "24m"]))
rows.append(audit("b violins (all TMS cells, 3 ages)", allT, ["3m", "18m", "24m"]))
rows.append(audit("b test / c delta / A69 selection (MALES, 3m vs 24m)", allT[allT.sex == "male"], ["3m", "24m"]))
rows.append(audit("A69 breadth + 18 vs 24 (all, 18m vs 24m)", allT, ["18m", "24m"]))
rows.append(audit("e/f Kcnip3 (Marrow HSC, 3 ages)", allT[allT.tissue_cell_type == "Marrow_HEMATOPOIETIC PROGENITOR"], ["3m", "18m", "24m"]))
out = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(out.to_string(index=False)); out.to_csv("age_confound_audit.tsv", sep="\t", index=False)
pd.crosstab([df.age, df.sex], df.campaign).to_csv("crosstab_age_sex_campaign.tsv", sep="\t")
pd.crosstab(df.plate, df.age).to_csv("crosstab_plate_age.tsv", sep="\t")
pd.crosstab(df.lane, df.age).to_csv("crosstab_lane_age.tsv", sep="\t")
