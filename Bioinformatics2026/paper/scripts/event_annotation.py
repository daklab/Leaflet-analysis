"""Junctions shown for each known event of the Application Note figure and the supplementary gallery, chosen by annotation only
(GENCODE vM19 = Ensembl 94, GRCm38; the catalog's annotation). No usage value is read here.

    python event_annotation.py        # sbatch (reads the GTF and the genome FASTA) -> <output_dir>/data/figAN_event_junctions.tsv
                                       #                                               + figAN_event_annotation.log (transcripts used)

Rules (one per event; exon numbers are positions in the named transcript in transcription order; this GTF carries no
exon_number attribute and no MANE set exists for mouse, so the named transcript is the longest CCDS transcript holding the exon):
- Gria2 flip/flop: the MXE pair of atse_3 (two exons > 100 nt joined to a shared site on each side). Each exon is translated
  (reverse strand) and assigned flip or flop by the residues that UniProt P19491 (rat GRIA2) gives for the two isoforms:
  flop N-A ... N ... L ... SGGGD, flip T-P ... S ... V ... AKDSG. Shown: the junction from the shared genomic-left site into each exon.
- Nrxn1 SS4: the annotated 30-nt SS4 exon (c10_known_biology_fix.SS4). Shown (Karin 2026-11-10): the SS4-skipping junction, i.e.
  the junction joining the two exons that are consecutive in the annotated transcripts lacking SS4 and that straddle SS4.
- Fgfr2 IIIb: the IIIb junction of atse_3 (c10_catalog.FGFR2_IIIB). IIIb is the 3' exon of the pair on the minus strand.
- Ptprc: every junction of atse_2 labeled by its exons in Ptprc-207 (the only transcript with variable exons 4, 5 and 6); the
  figure picks the junction by the 10.4 rule.
- Cd44 v8-v10: variable exons = exons of the longest CCDS transcript (Cd44-201) absent from the standard isoform (Cd44-203, CCDS,
  no variable exon), numbered v1.. in transcription order (Bebee 2015: 10 variable exons in mouse). Shown: the Cd44s junction
  (skips every variable exon) and the junction from v10 to the next constant exon.
- Enah 11a: the 63-nt alternative exon (21 codons, the length of the 11a insert) annotated only outside the CCDS transcripts.
  GENCODE does not name it 11a: identification by length and position, UNVERIFIED. Shown: inclusion from the upstream exon, and skip.
- Ptbp2 exon 10, Dlg4 exon 18: exon 10 of Ptbp2-201 and exon 18 of Dlg4-203 (CCDS). Shown: inclusion from the upstream exon, and skip.
- Ctnnd1, Pkm: no ATSE of either gene is in the catalog; rows are written with no junction.
"""
import json, os, re, subprocess, sys
from collections import defaultdict

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "paper_config.yaml")))
OUT = os.path.join(CFG["output_dir"], "data")
GTF = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
FA = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"
PRE = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser/precomputed"
GENES = ["Gria2", "Nrxn1", "Fgfr2", "Ptprc", "Cd44", "Enah", "Ptbp2", "Dlg4", "Ctnnd1", "Pkm"]
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_lines.append(s)


def read_gtf():
    pat = "|".join(f'gene_name "{g}";' for g in GENES)
    lines = subprocess.run(["grep", "-E", pat, GTF], capture_output=True, text=True, check=True).stdout.splitlines()
    att = lambda s, k: (re.search(k + r' "?([^";]+)"?;', s) or [None, None])[1]   # noqa: E731
    tx, ex = defaultdict(dict), defaultdict(list)
    for l in lines:
        f = l.split("\t")
        g = att(f[8], "gene_name")
        if f[2] == "transcript":
            tx[g][att(f[8], "transcript_id")] = dict(name=att(f[8], "transcript_name"), chrom=f[0], strand=f[6],
                                                     ccds="CCDS" in re.findall(r'tag "([^"]+)"', f[8]), type=att(f[8], "transcript_type"))
        elif f[2] == "exon":
            ex[att(f[8], "transcript_id")].append((int(f[3]), int(f[4])))
    T = {}
    for g in tx:
        for t, m in tx[g].items():
            m["exons"] = sorted(ex[t], key=lambda e: e[0], reverse=m["strand"] == "-")   # transcription order
            T[t] = dict(m, gene=g, id=t)
    return T


def faidx():
    fai = {l.split("\t")[0]: list(map(int, l.split("\t")[1:5])) for l in open(FA + ".fai")}

    def get(ch, a, b):
        ln, off, lb, lB = fai[ch]
        with open(FA, "rb") as f:
            return "".join((f.seek(off + (p // lb) * lB + p % lb), f.read(1).decode())[1] for p in range(a - 1, b)).upper()
    return get


def translate(s):
    b = "TCAG"; aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    code = {x + y + z: aa[16 * i + 4 * j + k] for i, x in enumerate(b) for j, y in enumerate(b) for k, z in enumerate(b)}
    return [("".join(code.get(s[i:i + 3], "X") for i in range(fr, len(s) - 2, 3))) for fr in range(3)]


def jid(ch, left, right, strand):
    """catalog junction id joining genomic-left exon `left` to genomic-right exon `right` (a = last base of left exon)"""
    return f"{ch}_{left[1]}_{right[0] - 1}_{strand}"


def join(t, up, down):
    """junction from exon `up` to exon `down` (transcription order) in transcript t"""
    return jid(t["chrom"], down, up, "-") if t["strand"] == "-" else jid(t["chrom"], up, down, "+")


def longest_ccds_with(T, gene, exon):
    c = [t for t in T.values() if t["gene"] == gene and t["ccds"] and exon in t["exons"]]
    return max(c, key=lambda t: (len(t["exons"]), t["id"])) if c else None


def main():
    os.makedirs(OUT, exist_ok=True)
    T = read_gtf(); get = faidx()
    v = pd.read_parquet(os.path.join(PRE, "catalog_junctions.parquet"))
    rows = []

    def row(key, gene, ds, event, junction, label, role, transcripts, exons, rule):
        if junction is not None:
            hit = v[(v.junction_id == junction)]
            assert len(hit) == 1 and hit.event_id.iloc[0] == event, (key, junction, hit.event_id.tolist())
        rows.append(dict(key=key, gene=gene, dataset=ds, event_id=event, junction_id=junction, label=label, role=role,
                         transcripts=transcripts, exons=exons, rule=rule))
        log(f"{key:10s} {gene:7s} {junction} | {label} | {transcripts} | {exons}")

    # ---- Gria2 flip / flop
    ev = "ENSMUSG00000033981.14_atse_3"; J = v[v.event_id == ev]
    L = [(r.a, r.b) for r in J.itertuples()]
    gria2_exons = {e for t in T.values() if t["gene"] == "Gria2" for e in t["exons"]}
    pairs = set()
    for (a1, b1) in L:
        for (a2, b2) in L:
            if a1 == a2 and b1 < b2:
                for (p, rp) in L:
                    for (q, rq) in L:
                        if (b1 < p <= b2 and q > b2 and rp == rq and rp > q and p - b1 > 100 and q - b2 > 100
                                and (b1 + 1, p) in gria2_exons and (b2 + 1, q) in gria2_exons):   # both annotated exons
                            pairs.add(((b1 + 1, p), (b2 + 1, q), a1))
    assert len(pairs) == 1, pairs
    e1, e2, Lsite = pairs.pop()
    sig = {"flop": ("NA", "LKLNEQGLLDK", "SGGGD"), "flip": ("TP", "LKLSEQGVLDK", "AKDSG")}   # UniProt P19491 VSP_000111-114
    lab = {}
    for e in (e1, e2):
        s = get("chr3", *e)[::-1].translate(str.maketrans("ACGT", "TGCA"))
        frames = [p for p in translate(s) if "*" not in p]
        hits = [k for k, (x, y, z) in sig.items() for p in frames if p.startswith(x) and y in p and z in p]
        assert len(hits) == 1, (e, frames)
        tr = [t for t in T.values() if t["gene"] == "Gria2" and e in t["exons"]]
        lab[e] = hits[0]
        log(f"Gria2 exon {e}: {hits[0]}; peptide {frames}; transcripts {[(t['name'], t['id'], 'CCDS' if t['ccds'] else '') for t in tr]}")
    for e in (e1, e2):
        tr = [t for t in T.values() if t["gene"] == "Gria2" and e in t["exons"] and t["ccds"]]
        row("gria2", "Gria2", "Allen", ev, f"chr3_{Lsite}_{e[0] - 1}_-", f"{lab[e]} exon", lab[e], ";".join(f"{t['name']} {t['id']}" for t in tr),
            f"chr3:{e[0]}-{e[1]} ({e[1] - e[0] + 1} nt)", "MXE pair; flip/flop by translation vs UniProt P19491")

    # ---- Nrxn1 SS4 skipping
    ss4 = (90992115, 90992144); ev = "ENSMUSG00000024109.18_atse_2"
    skip = set(); used = []
    for t in T.values():
        if t["gene"] != "Nrxn1" or ss4 in t["exons"]:
            continue
        E = t["exons"]
        for i in range(len(E) - 1):
            if E[i][0] > ss4[1] and E[i + 1][1] < ss4[0]:       # minus strand: consecutive exons straddling SS4
                skip.add(join(t, E[i], E[i + 1])); used.append(f"{t['name']} {t['id']}")
    inc = [t for t in T.values() if t["gene"] == "Nrxn1" and ss4 in t["exons"]]
    assert len(skip) == 1, skip
    j = skip.pop()
    row("nrxn1", "Nrxn1", "Allen", ev, j, "SS4-skipping junction", "skip", ";".join(used), f"SS4 chr17:{ss4[0]}-{ss4[1]} (30 nt)",
        "junction joining the exons that straddle SS4 in annotated transcripts lacking it")
    log(f"Nrxn1: {len(inc)} transcripts include SS4; skipping transcripts {used}")

    # ---- Fgfr2 IIIb
    iiib = sorted({e for t in T.values() if t["gene"] == "Fgfr2" for e in t["exons"] if e[1] == 130199905})
    tr = [f"{t['name']} {t['id']}" for t in T.values() if t["gene"] == "Fgfr2" and t["ccds"] and any(e in t["exons"] for e in iiib)]
    row("fgfr2", "Fgfr2", "TMS", "ENSMUSG00000030849.18_atse_3", "chr7_130199905_130200982_-", "IIIb junction", "IIIb", ";".join(tr),
        "IIIb exon(s) " + ", ".join(f"chr7:{a}-{b} ({b - a + 1} nt)" for a, b in iiib), "check 10 definition (c10_catalog.FGFR2_IIIB)")

    # ---- Ptprc: label every atse_2 junction by its exons in the transcript holding all three variable exons
    t207 = max((t for t in T.values() if t["gene"] == "Ptprc" and t["ccds"]), key=lambda t: len(t["exons"]))
    num = {e: i + 1 for i, e in enumerate(t207["exons"])}
    ev = "ENSMUSG00000026395.16_atse_2"
    for r in v[v.event_id == ev].itertuples():
        left = [e for e in num if e[1] == r.a]; right = [e for e in num if e[0] == r.b + 1]
        if left and right:   # minus strand: right exon is upstream
            up, dn = num[right[0]], num[left[0]]
            row("ptprc", "Ptprc", "TMS", ev, r.junction_id, f"exon {up}→{dn}", f"e{up}_e{dn}", f"{t207['name']} {t207['id']}",
                f"exon {up} chr1:{right[0][0]}-{right[0][1]}; exon {dn} chr1:{left[0][0]}-{left[0][1]}", "exons numbered in " + t207["name"])

    # ---- Cd44 v8-v10
    full = max((t for t in T.values() if t["gene"] == "Cd44" and t["ccds"]), key=lambda t: len(t["exons"]))
    std = min((t for t in T.values() if t["gene"] == "Cd44" and t["ccds"]), key=lambda t: len(t["exons"]))
    # variable region: exons of the full transcript inserted between two exons that are consecutive in the standard isoform
    gaps = [(u, d) for u, d in zip(std["exons"], std["exons"][1:]) if u in full["exons"] and d in full["exons"]
            and full["exons"].index(d) - full["exons"].index(u) > 1]
    assert len(gaps) == 1, gaps
    first = full["exons"].index(gaps[0][0]) + 1; last = full["exons"].index(gaps[0][1]) - 1
    var = full["exons"][first:last + 1]
    log(f"Cd44 variable exons ({len(var)}) in {full['name']} {full['id']} not in {std['name']} {std['id']}: {var}")
    ev = "ENSMUSG00000005087.17_atse_1"
    up, dn = full["exons"][first - 1], full["exons"][last + 1]
    row("cd44", "Cd44", "TMS", ev, join(full, up, dn), "Cd44s junction (skips v1–v10)", "skip_all", f"{std['name']} {std['id']}",
        f"constant exons chr2:{up[0]}-{up[1]} and chr2:{dn[0]}-{dn[1]}", "standard isoform junction")
    row("cd44", "Cd44", "TMS", ev, join(full, var[-1], dn), "v10 → constant exon", "v10_incl", f"{full['name']} {full['id']}",
        f"v8 {var[-3]}, v9 {var[-2]}, v10 {var[-1]}", "variable exons = exons of the longest CCDS transcript absent from the standard isoform")

    # ---- Enah 11a (63 nt, outside CCDS transcripts)
    ccds_ex = {e for t in T.values() if t["gene"] == "Enah" and t["ccds"] for e in t["exons"]}
    cand = {(e, t["id"]) for t in T.values() if t["gene"] == "Enah" for e in t["exons"] if e[1] - e[0] + 1 == 63 and e not in ccds_ex}
    assert len({e for e, _ in cand}) == 1, cand
    e11a, tid = sorted(cand)[0]; t = T[tid]; i = t["exons"].index(e11a)
    ev = v[v.junction_id == join(t, t["exons"][i - 1], e11a)].event_id.iloc[0]
    row("enah", "Enah", "TMS", ev, join(t, t["exons"][i - 1], e11a), "inclusion (63-nt exon)", "incl", f"{t['name']} {tid}",
        f"chr1:{e11a[0]}-{e11a[1]} (63 nt)", "11a by length, UNVERIFIED")
    row("enah", "Enah", "TMS", ev, join(t, t["exons"][i - 1], t["exons"][i + 1]), "skipping junction", "skip", f"{t['name']} {tid}",
        f"chr1:{e11a[0]}-{e11a[1]} (63 nt)", "11a by length, UNVERIFIED")

    # ---- Ptbp2 exon 10, Dlg4 exon 18 (numbered in the named CCDS transcript)
    for gene, name, k, key in (("Ptbp2", "Ptbp2-201", 10, "ptbp2"), ("Dlg4", "Dlg4-203", 18, "dlg4")):
        t = [x for x in T.values() if x["name"] == name][0]
        ex = t["exons"][k - 1]; upx, dnx = t["exons"][k - 2], t["exons"][k]
        ev = v[v.junction_id == join(t, upx, ex)].event_id.iloc[0]
        row(key, gene, "TMS", ev, join(t, upx, ex), f"exon {k - 1}→{k} (inclusion)", "incl", f"{name} {t['id']}",
            f"exon {k} {t['chrom']}:{ex[0]}-{ex[1]} ({ex[1] - ex[0] + 1} nt)", f"exon {k} of {name}")
        row(key, gene, "TMS", ev, join(t, upx, dnx), f"exon {k - 1}→{k + 1} (skip)", "skip", f"{name} {t['id']}",
            f"exon {k} {t['chrom']}:{ex[0]}-{ex[1]} ({ex[1] - ex[0] + 1} nt)", f"exon {k} of {name}")

    # ---- not in the catalog
    for gene, key in (("Ctnnd1", "ctnnd1"), ("Pkm", "pkm")):
        assert not v.gene_name.eq(gene).any()
        row(key, gene, "TMS", None, None, "no ATSE of this gene in the catalog", "none", "", "", "catalog build filters")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "figAN_event_junctions.tsv"), sep="\t", index=False)
    open(os.path.join(OUT, "figAN_event_annotation.log"), "w").write("\n".join(log_lines) + "\n")
    log("wrote", os.path.join(OUT, "figAN_event_junctions.tsv"))


if __name__ == "__main__":
    sys.exit(main())
