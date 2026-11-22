#!/usr/bin/env python3
"""62_utr_extract.py —— 从 Ensembl GTF 抽 5′/3′UTR，取「最长」

规则（任务书 §2.3）：
  1) 同一个转录本的 UTR 可能被内含子切成多段 → 先按转录本收集全部分段、长度求和
  2) 每个基因在【转录本】层面取最长的那一条（不是取并集，不是取第一个）
  3) 输出该转录本的分段 BED，name = `gene|段序号`，段序号按转录方向（5′→3′）排

产出
  ref/utr5.bed  ref/utr3.bed        供 bedtools getfasta -s -name 取序列
  ref/utr_len.tsv                   gene_id / utr5_len / utr3_len / 选中的转录本

用法: python 62_utr_extract.py
"""
import os
import pandas as pd

R = os.environ["IETX_HUMAN"]      # 复用 T-02 的注释
O = os.environ["IETX_T03"]
GTF = f"{R}/ref/annotation.gtf"
KINDS = {"five_prime_utr": "utr5", "three_prime_utr": "utr3"}


def attr(s, k):
    pat = k + ' "'
    p = s.find(pat)
    if p < 0:
        return ""
    p += len(pat)
    q = s.find('"', p)
    return s[p:q]


def main():
    # tx[(kind, tx_id)] = [gene, chrom, strand, [(start,end), ...]]
    tx = {}
    n = 0
    with open(GTF) as fh:
        for ln in fh:
            if ln[0] == "#":
                continue
            f = ln.split("\t", 8)
            kind = KINDS.get(f[2])
            if kind is None:
                continue
            a = f[8]
            tid = attr(a, "transcript_id")
            gid = attr(a, "gene_id")
            st, en = int(f[3]), int(f[4])
            key = (kind, tid)
            if key not in tx:
                tx[key] = [gid, f[0], f[6], []]
            tx[key][3].append((st, en))
            n += 1
    print(f"UTR 区间 {n} 条，转录本 {len(tx)} 个")

    # 转录本层面：长度 + 按转录方向排好段序
    best = {}      # (kind, gene) -> (len, tid, chrom, strand, segs)
    for (kind, tid), (gid, chrom, strand, segs) in tx.items():
        segs.sort(reverse=(strand == "-"))
        L = sum(e - s + 1 for s, e in segs)
        k = (kind, gid)
        if k not in best or L > best[k][0]:
            best[k] = (L, tid, chrom, strand, segs)

    rows, out = [], {"utr5": [], "utr3": []}
    for (kind, gid), (L, tid, chrom, strand, segs) in best.items():
        for i, (s, e) in enumerate(segs, 1):
            out[kind].append(f"{chrom}\t{s-1}\t{e}\t{gid}|{i}\t0\t{strand}")
    for kind, lines in out.items():
        with open(f"{O}/ref/{kind}.bed", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"  {kind}.bed: {len(lines)} 段")

    for kind in ("utr5", "utr3"):
        d = {g: v[0] for (k, g), v in best.items() if k == kind}
        rows.append(pd.Series(d, name=f"{kind}_len"))
    df = pd.concat(rows, axis=1)
    df.index.name = "gene_id"
    df.to_csv(f"{O}/ref/utr_len.tsv", sep="\t")
    print(f"  基因：utr5 {df.utr5_len.notna().sum()}  utr3 {df.utr3_len.notna().sum()}")
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
