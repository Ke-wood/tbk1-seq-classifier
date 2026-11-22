#!/usr/bin/env python3
"""57_features.py —— T-02 第 3 步：9 维特征

算子逐条照抄涡虫侧（10_features_seq.sh / 14_features_cds.py），不另起炉灶：
  prom_gc / prom_cpg_oe / prom_lowcomp   TSS 上游 1 kb（链感知）
  mfe_per_nt                             RNAfold MFE ÷ 启动子长度
  gene_span / n_transcripts              基因坐标 / 转录本数
  cds_gc / cds_cpg_oe / cai              最长 CDS

⚠️ CAI 参考表【自算】—— 用本注释的 CDS 全集，不沿用外置人类高频密码子表。
   外置表只另出一列 cai_ext 作敏感性，不进主模型。
⚠️ gffread 0.12.9 的 CDS 头是【裸转录本 ID】（`>ENST00000378513`），既不带版本
   也不带 gene_id —— 和涡虫侧的 `gene_id=xxx;` 完全不同。照抄老解析函数会拿转录本
   ID 当基因 ID，join 全落空、cds_* 三列全 NaN，而且一声不响。
   故这里显式用 results/tx2gene.tsv 做映射。

用法: env/bin/python 57_features.py
"""
import os, re
from collections import Counter
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = f"{R}/ref"
OUT = f"{R}/03_features"
os.makedirs(OUT, exist_ok=True)

BASES = "TCAG"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"


def read_fa(path, attr_re=None):
    name, buf = None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name:
                    yield name, "".join(buf).upper()
                if attr_re:
                    m = re.search(attr_re, ln)
                    name = re.sub(r"\.\d+$", "", m.group(1)) if m else ln[1:].strip()
                else:
                    name = ln[1:].split("::")[0].strip()
                buf = []
            else:
                buf.append(ln.strip())
    if name:
        yield name, "".join(buf).upper()


def cpg_oe(s):
    n = len(s)
    if n < 2:
        return np.nan
    c, g, cg = s.count("C"), s.count("G"), s.count("CG")
    return (cg * n) / (c * g) if c and g else np.nan


def lowcomp(s):
    if len(s) < 10:
        return np.nan
    k = 2
    cnt = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    p = np.array(list(cnt.values()), float); p /= p.sum()
    return float(1 + (p * np.log2(p)).sum() / np.log2(16))


def main():
    # ── CDS 侧 ────────────────────────────────────────────────
    # cds.fa 的键是转录本 ID，先映射到基因，再取每个基因最长的 CDS
    t2g = pd.read_csv(f"{R}/results/tx2gene.tsv", sep="\t",
                      index_col="transcript_id")["gene_id"]
    best = {}
    unmapped = 0
    for tx, s in read_fa(f"{REF}/cds.fa"):
        gid = t2g.get(tx)
        if gid is None:
            unmapped += 1
            continue
        s = re.sub(r"[^ACGT]", "", s)
        if len(s) > len(best.get(gid, "")):
            best[gid] = s
    print(f"CDS 转录本未映射到基因: {unmapped}")
    assert unmapped == 0, "cds.fa 的 ID 与 tx2gene 对不上"
    print(f"有 CDS 的基因: {len(best)}")

    cnt = dict.fromkeys(CODONS, 0)
    for s in best.values():
        for i in range(0, len(s) - 2, 3):
            if s[i:i + 3] in cnt:
                cnt[s[i:i + 3]] += 1
    aa_of = {c: AA[i] for i, c in enumerate(CODONS)}
    fmax = {}
    for c, n in cnt.items():
        a = aa_of[c]
        if a != "*":
            fmax[a] = max(fmax.get(a, 0), n)
    w = {c: (cnt[c] / fmax[aa_of[c]] if fmax.get(aa_of[c]) else np.nan)
         for c in CODONS if aa_of[c] != "*"}
    print(f"参考密码子表已自算（{sum(cnt.values()):,} 密码子，{len(fmax)} 个氨基酸）")

    rows = []
    for gid, s in best.items():
        n = len(s)
        c_, g_ = s.count("C"), s.count("G")
        ws = [w[s[i:i + 3]] for i in range(0, n - 2, 3)
              if s[i:i + 3] in w and w[s[i:i + 3]] > 0]
        rows.append((gid, n, (c_ + g_) / n if n else np.nan, cpg_oe(s),
                     float(np.exp(np.mean(np.log(ws)))) if ws else np.nan))
    cds = pd.DataFrame(rows, columns=["gene_id", "cds_len_longest", "cds_gc",
                                      "cds_cpg_oe", "cai"])

    # ── 启动子侧 ──────────────────────────────────────────────
    prom = pd.DataFrame(
        [(g, len(s), (s.count("G") + s.count("C")) / len(s) if s else np.nan,
          cpg_oe(s), lowcomp(s)) for g, s in read_fa(f"{REF}/promoters.fa")],
        columns=["gene_id", "prom_len", "prom_gc", "prom_cpg_oe", "prom_lowcomp"])
    mfe = pd.read_csv(f"{REF}/mfe.tsv", sep="\t", header=None,
                      names=["gene_id", "mfe"])

    # ── 基因模型 ──────────────────────────────────────────────
    g = pd.read_csv(f"{REF}/genes.bed", sep="\t", header=None,
                    names=["chrom", "start", "end", "gene_id", "score", "strand"])
    g["gene_span"] = g.end - g.start
    ntx = pd.read_csv(f"{REF}/n_tx.tsv", sep="\t", header=None,
                      names=["gene_id", "n_transcripts"])
    f = g[["gene_id", "gene_span"]].merge(ntx, on="gene_id", how="left")
    f["n_transcripts"] = f.n_transcripts.fillna(0).astype(int)
    f = f.merge(prom, on="gene_id", how="left") \
         .merge(mfe, on="gene_id", how="left") \
         .merge(cds, on="gene_id", how="left")
    f["mfe_per_nt"] = f.mfe / f.prom_len
    f.to_csv(f"{OUT}/features.tsv", sep="\t", index=False)

    # ⚠️ 静默失败检查：v2 的坑是「行数对、值全 NaN」。这里查值域而不是行数。
    n = len(f)
    print(f"\n写出 features.tsv: {n} 基因 × {f.shape[1]} 列")
    chk = [("prom_gc", 0, 1), ("prom_cpg_oe", 0, 5), ("prom_lowcomp", 0, 1),
           ("mfe_per_nt", -1, 0), ("gene_span", 1, np.inf),
           ("n_transcripts", 1, np.inf), ("cds_gc", 0, 1),
           ("cds_cpg_oe", 0, 5), ("cai", 0, 1.01)]
    bad = []
    for c, lo, hi in chk:
        v = f[c].dropna()
        ok = len(v) > 0 and v.between(lo, hi).mean() > 0.99
        print(f"  {c:16s} 非空 {len(v):>6d}/{n}  中位 {v.median():>10.4f}  "
              f"{'✓' if ok else '✗ 值域异常'}")
        if not ok:
            bad.append(c)
    if bad:
        raise SystemExit(f"!! 特征值域异常: {bad} —— 不要往下跑")
    if abs(f.prom_len.median() - 1000) > 5:      # getfasta 取错方向/取空的典型症状
        raise SystemExit(f"!! prom_len 中位 {f.prom_len.median():.0f}，应约 1000")
    # ⚠️ 中位数对「只有几百条」完全无感：曾经因为两个 run_all 同时跑，
    #    57 读到正在写入的 promoters.fa，只匹配上 571/62754 基因，
    #    而 prom_len 中位依然是 1000、值域检查全 ✓ —— 静默通过。
    n_prom = int(f.prom_len.notna().sum())
    if n_prom < 0.9 * len(f):
        raise SystemExit(f"!! 只有 {n_prom}/{len(f)} 个基因拿到启动子序列，promoters.fa 不完整")
    print("值域自检通过")


if __name__ == "__main__":
    main()
