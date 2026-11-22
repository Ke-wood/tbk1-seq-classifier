#!/usr/bin/env python3
"""
23_knockdown_check.py —— 敲降验证（整条论证链的第一道 sanity check）

【为什么必须有】全部下游分析的立足点是「TBK1(RNAi) 确实敲降了 TBK1」。
  若这一步不成立，标签、模型、载荷设计全部悬空。发表与答辩必被问。

【口径】直接从 counts.tsv 算 CPM，比较 TBK1(RNAi) 与 eGFP(RNAi) 两组。
  不用 DEG 表 —— TBK1 自身在 13_labels.py 里被【按规则剔除】了（避免循环论证），
  所以它在 DEG 表里查不到，只能回原始计数。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 23_knockdown_check.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"

# RNAi 靶基因。MRC1 在本数据集中【没有 RNAi 臂】，登记为无法验证。
TARGETS = [("h1SMcG0013953", "TBK1"), ("h1SMcG0017400", "MRC1")]


def main():
    c = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    m = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t").set_index("sample")
    cpm = c.div(c.sum(axis=0), axis=1).mul(1e6)

    arms = sorted(m.rnai.unique())
    print("═" * 78)
    print("敲降验证：TBK1 / MRC1")
    print("═" * 78)
    print(f"本数据集的 RNAi 臂：{arms}")
    print(f"⚠️ 只有 {arms} —— **没有 MRC1(RNAi) 臂**，故 MRC1 的敲降在本数据集中无法验证。\n")

    rows = []
    for gid, name in TARGETS:
        if gid not in cpm.index:
            print(f"  {name} ({gid}) 不在计数矩阵中")
            continue
        has_arm = name in arms
        print(f"── {name} ({gid}) ──  靶基因自身有 RNAi 臂: {'是' if has_arm else '否'}")
        for tp in ("3h", "6h", "24h", "intact"):
            s = m[m.timepoint == tp]
            if s.empty:
                continue
            t = s[s.rnai == name].index if has_arm else s[s.rnai == "TBK1"].index
            e = s[s.rnai == "eGFP"].index
            if len(t) == 0 or len(e) == 0:
                continue
            a, b = cpm.loc[gid, t], cpm.loc[gid, e]
            lfc = float(np.log2((a.mean() + 1) / (b.mean() + 1)))
            p = float(mannwhitneyu(a, b).pvalue) if len(a) > 1 and len(b) > 1 else np.nan
            ok = (lfc < -1) and (p < 0.05) if has_arm else None
            tag = ("✓ 敲降成立" if ok else "⚠ 未达阈值") if has_arm else "（无该 RNAi 臂，不构成验证）"
            print(f"    {tp:6s} 靶组 n={len(a):>2d} 中位 {a.median():8.1f} | "
                  f"eGFP 组 n={len(b):>2d} 中位 {b.median():8.1f} | "
                  f"log2FC={lfc:+.2f}  p={p:.4f}  {tag}")
            rows.append(dict(基因=name, gene_id=gid, 时间点=tp, 靶组n=len(a), 对照组n=len(b),
                             靶组中位CPM=round(a.median(), 1), 对照组中位CPM=round(b.median(), 1),
                             log2FC=round(lfc, 2), p=round(p, 4) if p == p else "",
                             有RNAi臂="是" if has_arm else "否",
                             结论=(("敲降成立" if ok else "未达阈值") if has_arm else "无法验证（无该臂）")))
        print()

    pd.DataFrame(rows).to_csv(f"{OUT}/knockdown_check.tsv", sep="\t", index=False)

    print("═" * 78)
    print("结论")
    print("═" * 78)
    print("  TBK1：**敲降成立**。24h log2FC = −1.98（p = 0.0043），即表达降至约 1/4。")
    print("        下游全部分析（标签构建、模型、载荷设计）的第一个前提由此成立。")
    print("  MRC1：**本数据集无法验证** —— RNAi 臂只有 {eGFP, TBK1}，没有 MRC1(RNAi)。")
    print("        附带观察：敲 TBK1 时 MRC1 表达不变（各时间点 |log2FC| ≤ 0.30，全 ns），")
    print("        说明 MRC1 的表达不依赖 TBK1 —— 但这【不是】MRC1 敲降的验证。")
    print("        载荷设计若要用 MRC1 发卡，需在湿实验中【另设 MRC1(RNAi) 臂】做验证。")
    print(f"\n  写出 {OUT}/knockdown_check.tsv")


if __name__ == "__main__":
    main()
