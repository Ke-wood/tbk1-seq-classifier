#!/usr/bin/env python3
"""
14_features_cds.py —— §5.4 CDS 侧特征：GC / CpG o/e / CAI

10_features_seq.sh 只做了启动子侧（上游 1 kb）+ 长度/转录本数。这里补 CDS 侧。
每个基因取【最长 CDS】为代表（多转录本时）。

CAI（密码子适应指数）：w_c = f_c / f_max(该氨基酸)，CAI = 几何平均(w_c)。
参考密码子使用频率从本物种 CDS 全集自算（非模式生物没有现成表，自算才是对的）。
终止密码子不计入。

产出 features/features_cds.tsv    gene_id / cds_gc / cds_cpg_oe / cai / cds_len_longest
用法: $IETX_ROOT/env/bin/python 14_features_cds.py
"""
import os
import re
import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
FEAT = f"{IETX}/features"

BASES = "TCAG"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
# 标准遗传密码（核基因；GFF3 里 transl_table=9 只用于线粒体）
AA = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")


def read_cds(path):
    name, buf = None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name:
                    yield name, "".join(buf).upper()
                m = re.search(r"gene_id=([^;]+)", ln)
                name, buf = (m.group(1) if m else ln[1:].strip()), []
            else:
                buf.append(ln.strip())
    if name:
        yield name, "".join(buf).upper()


def main():
    best = {}
    for gid, s in read_cds(f"{IETX}/ref/cds.fa"):
        s = re.sub(r"[^ACGT]", "", s)
        if len(s) > len(best.get(gid, "")):
            best[gid] = s
    print(f"基因（有 CDS）: {len(best)}")

    # 参考密码子频率（自算，只统计完整密码子、排除终止子）
    cnt = dict.fromkeys(CODONS, 0)
    for s in best.values():
        for i in range(0, len(s) - 2, 3):
            c = s[i:i + 3]
            if c in cnt:
                cnt[c] += 1
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
        gc = (s.count("G") + s.count("C")) / n if n else np.nan
        c_, g_ = s.count("C"), s.count("G")
        cpg = (s.count("CG") * n) / (c_ * g_) if c_ and g_ else np.nan
        ws = [w[s[i:i + 3]] for i in range(0, n - 2, 3)
              if s[i:i + 3] in w and w[s[i:i + 3]] > 0]
        cai = float(np.exp(np.mean(np.log(ws)))) if ws else np.nan
        rows.append((gid, n, gc, cpg, cai))
    df = pd.DataFrame(rows, columns=["gene_id", "cds_len_longest", "cds_gc",
                                     "cds_cpg_oe", "cai"])
    df.to_csv(f"{FEAT}/features_cds.tsv", sep="\t", index=False)
    print(f"写出 features_cds.tsv: {len(df)} 基因")
    print(df[["cds_len_longest", "cds_gc", "cds_cpg_oe", "cai"]].describe().T.to_string())


if __name__ == "__main__":
    main()
