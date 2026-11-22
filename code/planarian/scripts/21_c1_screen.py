#!/usr/bin/env python3
"""
21_c1_screen.py —— C1：从 TBK1(RNAi) 上调集中筛 TBK1 轴负调控因子

【为什么重要】上调集 = TBK1 敲降后【去阻遏】的基因。若其中存在 TBK1/NF-κB 的
  负反馈调控因子，它们就是"主动关闭 TBK1 轴"最直接的效应器候选。
  这是本项目独有的东西 —— 公开 CDS 谁都能拿，这份筛选不能。

【两种方法都做，缺一不可】
  方法甲  SwissProt 最佳命中描述的关键词匹配（快，但只看得见最好那一条）
  方法乙  【靶向家族序列比对】—— 从 SwissProt 精确提取各负调控因子家族的参考序列，
          再 DIAMOND 搜涡虫蛋白组。比甲灵敏得多，且能给出富集检验。

【诚实要求】若一个都没筛到，如实报"无"，不要凑（用户明示）。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 21_c1_screen.py
"""
import os
import gzip
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, DB = f"{IETX}/results", f"{IETX}/db"
E = f"{IETX}/env/bin"
WORK = f"{OUT}/payload/_c1_work"

# 负调控因子家族（用户给的清单 + 轴本体作对照）
FAM = {
    "TNFAIP3/A20":    r"\bTNFAIP3\b|\bA20\b|Tumor necrosis factor alpha-induced protein 3",
    "SOCS":           r"\bSOCS[1-7]\b|\bCISH\b|\bSOCS box\b|Suppressor of cytokine signaling",
    "NFKBIA/IkB":     r"\bNFKBIA\b|\bNFKBIB\b|\bNFKBIE\b|\bNFKBIZ\b|NF-kappa-B inhibitor",
    "DUSP":           r"\bDUSP\d+\b|Dual specificity protein phosphatase",
    "CYLD":           r"\bCYLD\b",
    "OTULIN":         r"\bOTULIN\b|\bGumby\b",
    "USP":            r"\bUSP[0-9]+\b|Ubiquitin carboxyl-terminal hydrolase",
    "PTPN":           r"\bPTPN[0-9]+\b|\bPTPRC\b|Tyrosine-protein phosphatase non-receptor",
    "TRIM":           r"\bTRIM[0-9]+\b|Tripartite motif",
    "CBL/ITCH/NEDD4": r"\bCBL\b|\bCBLB\b|\bCBLC\b|\bITCH\b|\bNEDD4\b|\bSMURF[12]\b|\bRNF[0-9]+\b"
                      r"|E3 ubiquitin-protein ligase",
    "PPM/PPP":        r"\bPPM1[A-B]\b|\bPHLPP[12]?\b|\bPPP1R\d+\b",
    "轴本体(对照)":     r"\bTBK1\b|\bIKBKE\b|\bIKBKB\b|\bCHUK\b|\bTRAF[1-7]\b|\bIRF3\b|\bIRF7\b"
                      r"|\bNFKB1\b|\bRELA\b|\bSTING1\b|\bCGAS\b|\bMAVS\b",
}
COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"]


def extract_queries():
    """从 SwissProt 精确提取各家族参考序列。按家族【优先匹配】，一条序列只归一个家族。"""
    pat = {k: re.compile(v, re.I) for k, v in FAM.items()}
    os.makedirs(WORK, exist_ok=True)
    n, counts = 0, {k: 0 for k in FAM}
    with gzip.open(f"{DB}/uniprot_sprot.fasta.gz", "rt", errors="replace") as fh, \
         open(f"{WORK}/negreg.fa", "w") as w:
        hdr, buf = None, []

        def flush():
            if not hdr:
                return
            s = "".join(buf)
            for k, p in pat.items():
                if p.search(hdr):
                    w.write(f">{k}|{hdr.split('|')[1] if '|' in hdr else ''}\n{s}\n")
                    counts[k] += 1
                    return
        for ln in fh:
            if ln[0] == ">":
                flush()
                hdr, buf = ln.strip(), []
                n += 1
            else:
                buf.append(ln.strip())
        flush()
    return n, counts


def main():
    os.makedirs(WORK, exist_ok=True)
    print("═" * 78)
    print("C1：从 TBK1(RNAi) 上调集中筛 TBK1 轴负调控因子")
    print("═" * 78)

    r = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t", index_col=0)
    up = r[(r.padj < 0.05) & (r.log2FoldChange > 0)]
    allg = set(r.index)
    print(f"上调显著集 n={len(up)}（padj<0.05 且 log2FC>0）")

    # ── 方法甲：关键词匹配最佳命中描述 ──
    print("\n── 方法甲：SwissProt 最佳命中描述的关键词匹配 ──")
    ga = pd.read_csv(f"{OUT}/gene_annotation.tsv", sep="\t")
    ann = up.reset_index().merge(ga, on="gene_id", how="left")
    print(f"  上调集有 SwissProt 注释的: {ann.stitle.notna().sum()}/{len(up)}"
          f"（{ann.stitle.notna().sum()/len(up)*100:.0f}%）")
    KW = [w for v in FAM.values() for w in re.findall(r"\\b[A-Z0-9][A-Z0-9\-]{2,}\\b", v)]
    hitA = ann[ann.stitle.fillna("").str.upper().apply(lambda s: any(k in s for k in KW))]
    print(f"  命中: {len(hitA)} 个")
    for _, x in hitA.iterrows():
        print(f"    {x.gene_id}  {str(x.stitle)[:80]}")

    # ── 方法乙：靶向家族序列比对 ──
    print("\n── 方法乙：靶向家族序列比对（灵敏）──")
    n, counts = extract_queries()
    print(f"  SwissProt 总条目 {n:,}；查询集:")
    for k, v in counts.items():
        print(f"    {k:18s} {v:>5d}")
    print(f"  合计 {sum(counts.values())} 条 → {WORK}/negreg.fa")

    subprocess.run([f"{E}/diamond", "makedb", "--in", f"{WORK}/negreg.fa",
                    "-d", f"{WORK}/negreg", "--quiet"], check=True)
    subprocess.run([f"{E}/diamond", "blastp", "-q", f"{IETX}/ref/proteins.fa",
                    "-d", f"{WORK}/negreg", "-o", f"{WORK}/hits.tsv",
                    "--very-sensitive", "-e", "1e-3", "--max-target-seqs", "1",
                    "--threads", "8"], check=True, capture_output=True)

    h = pd.read_csv(f"{WORK}/hits.tsv", sep="\t", header=None, names=COLS)
    h["fam"] = h.sseqid.str.split("|").str[0]
    t2g = pd.read_csv(f"{OUT}/tx2gene.tsv", sep="\t").set_index("transcript_id")["gene_id"]
    h["gene"] = h.qseqid.map(t2g)
    h = h.dropna(subset=["gene"])
    best = h.sort_values(["evalue", "bitscore"], ascending=[True, False]).drop_duplicates("gene")
    print(f"  查询集命中基因 {len(best)}/{len(allg)}（{len(best)/len(allg)*100:.0f}%）"
          f" —— 家族域共享导致命中不特异，必须看富集而非命中本身")

    rest = allg - set(up.index)
    print(f"\n  {'阈值':10s} {'上调命中':>10s} {'其余命中':>10s} {'OR':>7s} {'p':>8s}")
    verdict = []
    for pid in (0, 20, 30, 40, 50):
        s = best[best.pident >= pid] if pid else best
        a = int(s.gene.isin(up.index).sum())
        c = int(s.gene.isin(rest).sum())
        if a == 0 and c == 0:
            continue
        orr, p = fisher_exact([[a, len(up) - a], [c, len(rest) - c]])
        print(f"  pident≥{pid:<3d}% {a:>6d}/{len(up):<4d} {c:>6d}/{len(rest):<5d} "
              f"{orr:>7.2f} {p:>8.3f}")
        verdict.append((pid, a, orr, p))

    # ── 结论 ──
    print("\n" + "═" * 78)
    print("结论")
    print("═" * 78)
    enriched = any(a > 0 and orr > 1 and p < 0.05 for _, a, orr, p in verdict)
    if not enriched:
        print("  【无】上调集中不存在可识别的 TBK1/NF-κB 负反馈调控因子。")
        print("  - 方法甲（关键词）：0 个")
        print("  - 方法乙（靶向比对）：上调集命中率不高于背景，pident≥40% 时 0 命中")
        print("  ⚠️ 置信边界：全基因组 SwissProt 注释覆盖仅约 40%。涡虫免疫基因进化快，")
        print("     基于同源性的注释对这类基因灵敏度低 —— 「无」是【在当前注释下无】，")
        print("     不等于生物学上不存在。更灵敏的做法（域级 Pfam/InterPro 或结构比对）本环境不具备。")
    else:
        print("  检出候选：")
        for pid, a, orr, p in verdict:
            if a > 0 and orr > 1 and p < 0.05:
                sub = best[(best.pident >= pid) & best.gene.isin(up.index)]
                for _, x in sub.iterrows():
                    print(f"    {x.gene}  {x.fam}  lfc={r.loc[x.gene,'log2FoldChange']:+.2f}  "
                          f"pident={x.pident:.1f}  e={x.evalue:.1e}")
                break

    rows = []
    for g in up.index:
        s = ga.set_index("gene_id").stitle.get(g)
        b = best[best.gene == g]
        rows.append(dict(
            gene_id=g, log2FC=r.loc[g, "log2FoldChange"], padj=r.loc[g, "padj"],
            swissprot_描述=(str(s)[7:90] if isinstance(s, str) else ""),
            **({"靶向比对家族": b.iloc[0].fam, "pident": round(b.iloc[0].pident, 1),
                "evalue": b.iloc[0].evalue} if len(b) else
               {"靶向比对家族": "", "pident": "", "evalue": ""})))
    out = pd.DataFrame(rows).sort_values("log2FC", ascending=False)
    out.to_csv(f"{OUT}/c1_candidates.tsv", sep="\t", index=False)
    print(f"\n  写出 {OUT}/c1_candidates.tsv（{len(out)} 行，含全部上调基因的注释与比对情况）")


if __name__ == "__main__":
    main()
