#!/usr/bin/env python3
"""
08_deseq2.py —— 差异表达分析（pydeseq2）

设计（经用户确认 2026-09-24）：
  主分析  对比① eGFP 24h DNA vs Mock，【只用这 6 个样本单独建模】
  敏感性  同一对比在【全部 38 样本】上重跑（离散度由全体样本估计）
  支持性  对比② TBK1 24h DNA vs Mock（n=3 vs 2 —— Zenodo 无 TBK1Mock24h3，
          已核实两版都没有，故降级为支持性证据）
  `intact` 不进模型：注射是强同步化扰动，DNA/Mock 两臂共享创伤应答故可比；
  intact 无此同步信号，个体差异裸露 → 组内相关性天然偏低（QC 已证：0.753/0.767，
  全数据集最低）。这是实验设计的必然，不是 intact 数据本身有质量问题。

产出
  results/deg_eGFP_24h_DNA_vs_Mock.tsv        主分析（6 样本）
  results/deg_eGFP_24h_DNA_vs_Mock_full.tsv   敏感性（38 样本）
  results/deg_TBK1_24h_DNA_vs_Mock.tsv        支持性（5 样本）
  results/deg_threshold_sensitivity.tsv       阈值敏感性分析
  results/degsets.json                        DEG 集合（供后续 IRS 标签使用）

用法: $IETX_ROOT/env/bin/python 08_deseq2.py
"""
import os
import re
import json
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
MIN_COUNT = 10

# 阈值敏感性网格（任务书 §5.3 要求给出选择依据，不能照抄）
PADJ_GRID = [0.01, 0.05, 0.10]
LFC_GRID = [0.5, 1.0, 1.5]


def run_deseq(counts, meta, design, contrast, label):
    """counts: 基因 × 样本；meta: 索引为样本名"""
    # 丢掉在该子集里全零的基因
    c = counts.loc[counts.sum(axis=1) > 0].T          # pydeseq2 要 样本 × 基因
    print(f"  [{label}] 输入 {c.shape[1]} 基因 × {c.shape[0]} 样本，设计 {design}")
    dds = DeseqDataSet(counts=c, metadata=meta, design=design, refit_cooks=True,
                       quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    r = st.results_df.copy()
    r.index.name = "gene_id"
    print(f"  [{label}] 完成，padj 非空 {r.padj.notna().sum()} / {len(r)}")
    return r


def main():
    counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t")

    # ---------- 主分析：对比① 的 6 个样本单独建模 ----------
    print("\n════ 主分析：对比① eGFP 24h DNA vs Mock（6 样本）════")
    sub = meta[meta["sample"].str.match(r"^eGFP(DNA|Mock)24h\d$")].copy()
    sub["condition"] = sub["treatment"]                  # DNA / Mock
    sub = sub.set_index("sample")[["condition"]]
    a = [s for s in sub.index if "DNA" in s]
    b = [s for s in sub.index if "Mock" in s]
    print(f"  DNA 臂: {sorted(a)}")
    print(f"  Mock臂: {sorted(b)}")
    res_main = run_deseq(counts[a + b], sub.loc[a + b],
                         "~condition", ["condition", "DNA", "Mock"], "主分析")
    res_main.to_csv(f"{OUT}/deg_eGFP_24h_DNA_vs_Mock.tsv", sep="\t")

    # ---------- 敏感性：全 38 样本估计离散度，对比仍为 ① ----------
    print("\n════ 敏感性检查：全 38 样本估计离散度 ════")
    full = meta.set_index("sample")[["group"]].rename(columns={"group": "condition"})
    res_full = run_deseq(counts, full, "~condition",
                         ["condition", "eGFP_DNA_24h", "eGFP_Mock_24h"], "全数据集")
    res_full.to_csv(f"{OUT}/deg_eGFP_24h_DNA_vs_Mock_full.tsv", sep="\t")

    # ---------- 支持性：对比② TBK1 24h ----------
    print("\n════ 支持性：对比② TBK1 24h DNA vs Mock（n=3 vs 2）════")
    sub2 = meta[meta["sample"].str.match(r"^TBK1(DNA|Mock)24h\d$")].copy()
    sub2["condition"] = sub2["treatment"]
    sub2 = sub2.set_index("sample")[["condition"]]
    res_t2 = run_deseq(counts[sub2.index], sub2, "~condition",
                       ["condition", "DNA", "Mock"], "对比②")
    res_t2.to_csv(f"{OUT}/deg_TBK1_24h_DNA_vs_Mock.tsv", sep="\t")

    # ---------- 阈值敏感性 ----------
    print("\n════ 阈值敏感性分析 ════")
    rows = []
    for padj in PADJ_GRID:
        for lfc in LFC_GRID:
            row = {"padj": padj, "abs_log2FC": lfc}
            for tag, r in [("主分析6样本", res_main), ("敏感性38样本", res_full),
                           ("支持性TBK1", res_t2)]:
                up = ((r.padj < padj) & (r.log2FoldChange > lfc)).sum()
                dn = ((r.padj < padj) & (r.log2FoldChange < -lfc)).sum()
                row[f"{tag}_上调"] = int(up)
                row[f"{tag}_下调"] = int(dn)
            rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(f"{OUT}/deg_threshold_sensitivity.tsv", sep="\t", index=False)
    print(sens.to_string(index=False))

    # ---------- 主分析 vs 敏感性：结论一致吗 ----------
    print("\n════ 主分析 vs 敏感性：结论一致性 ════")
    for padj in PADJ_GRID:
        for lfc in LFC_GRID:
            s1 = set(res_main.index[(res_main.padj < padj) & (res_main.log2FoldChange.abs() > lfc)])
            s2 = set(res_full.index[(res_full.padj < padj) & (res_full.log2FoldChange.abs() > lfc)])
            if not s1 and not s2:
                continue
            j = len(s1 & s2) / max(len(s1 | s2), 1)
            print(f"  padj<{padj:<4} |log2FC|>{lfc:<3}  "
                  f"主分析 {len(s1):>5d}  敏感性 {len(s2):>5d}  "
                  f"交集 {len(s1&s2):>5d}  Jaccard={j:.3f}")

    # ---------- 落盘 DEG 集合 ----------
    sets = {}
    for padj in PADJ_GRID:
        for lfc in LFC_GRID:
            k = f"padj{padj}_lfc{lfc}"
            sets[k] = {
                "main_up": sorted(res_main.index[(res_main.padj < padj) & (res_main.log2FoldChange > lfc)]),
                "main_down": sorted(res_main.index[(res_main.padj < padj) & (res_main.log2FoldChange < -lfc)]),
            }
    json.dump(sets, open(f"{OUT}/degsets.json", "w"), indent=1)
    print(f"\n  写出: degsets.json（{len(sets)} 套阈值组合）")


if __name__ == "__main__":
    main()
