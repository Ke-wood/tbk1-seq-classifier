#!/usr/bin/env python3
"""58_labels.py —— T-02 第 4 步：标签构造（双向主口径 + 单向子集）

规则照抄涡虫侧 13_labels.py：
  正类  padj < 0.05 且 |log2FC| > 0.5（双向）；另出 down(<−0.5) 与 up(>+0.5)
  负类  不显著池（padj 缺失或 ≥0.5）中按 表达量 × 基因跨度 × 转录本数 最近邻 1:1 匹配
        权重 0.60 / 0.25 / 0.15；从高表达正类开始贪心挑
  剔除  涡虫侧剔除敲降靶 TBK1；IFN 刺激没有靶基因，本数据集不剔任何基因（方法里写明）

⚠️ sorted() 不是装饰：set 的行序随 PYTHONHASHSEED 变，距离并列时 argmin 取的
   「第一个最小」跟着变，结果就不可复现。13_labels.match() 已处理，此处直接复用。

用法: env/bin/python 58_labels.py GSE309167
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.dirname(os.path.abspath(__file__))
# 13_labels.py 在模块级就检查 IETX_ROOT；这里只借它的函数，给个占位值即可
os.environ.setdefault("IETX_ROOT", "/nonexistent")
_s = importlib.util.spec_from_file_location(
    "labels13", os.path.join(S, "13_labels.py"))
L13 = importlib.util.module_from_spec(_s); _s.loader.exec_module(L13)

PADJ, NS_PADJ, LFC = 0.05, 0.5, 0.5
KINDS = {"both": 0, "down": -1, "up": +1}   # 0 = 双向


def build(res, feats):
    """res: DEG 表（index=gene_id）；feats: index=gene_id，含 expr"""
    out, rows = {}, []
    for kind, s in KINDS.items():
        if kind == "both":
            sig = set(res.index[(res.padj < PADJ) & (res.log2FoldChange.abs() > LFC)])
            opp = set()
        else:
            sig = set(res.index[(res.padj < PADJ) & (s * res.log2FoldChange > LFC)])
            opp = set(res.index[(res.padj < PADJ) & (s * res.log2FoldChange < -LFC)])
        # ⚠️ padj 为 NaN 的是被独立过滤的基因，仍属「非显著」背景，不能 dropna 丢掉
        #    （涡虫侧 16_validation 的注释里记着：用 dropna 会把背景从 16,191 缩到 4,370）
        ns = set(res.index[res.padj.isna() | (res.padj >= NS_PADJ)])
        pool = sorted(ns - sig - opp)
        neg, _ = L13.match(sig, pool, feats)
        df = pd.DataFrame(
            [dict(gene_id=g, label=1, log2FC=res.loc[g, "log2FoldChange"],
                  padj=res.loc[g, "padj"]) for g in sorted(sig)] +
            [dict(gene_id=g, label=0, log2FC=res.loc[g, "log2FoldChange"],
                  padj=res.loc[g, "padj"]) for g in sorted(neg)])
        # 匹配质量：SMD 必须 <0.25，否则模型学到的是表达量不是序列
        smd = {}
        for cov in ("expr", "gene_span", "n_transcripts"):
            a, b = (df.merge(feats.reset_index()[["gene_id", cov]], on="gene_id",
                             how="left").query("label==1")[cov].astype(float),
                    df.merge(feats.reset_index()[["gene_id", cov]], on="gene_id",
                             how="left").query("label==0")[cov].astype(float))
            smd[cov] = abs(a.mean() - b.mean()) / pd.concat([a, b]).std()
        out[kind] = df
        rows.append(dict(kind=kind, n_pos=len(sig), n_neg=len(neg),
                         pool=len(pool), **{f"SMD_{k}": round(v, 3) for k, v in smd.items()}))
    return out, pd.DataFrame(rows)


def main(gse):
    D = f"{R}/人类侧_{gse}"
    outdir = f"{D}/05_label"
    os.makedirs(outdir, exist_ok=True)
    feats = pd.read_csv(f"{R}/03_features/features.tsv", sep="\t")
    expr = pd.read_csv(f"{D}/01_qc/cpm_mean.tsv", sep="\t", index_col=0)["expr"]
    feats["expr"] = feats.gene_id.map(expr)
    if feats.expr.isna().mean() > 0.05:
        print(f"  ⚠️ {feats.expr.isna().mean():.1%} 基因没有表达量（不在计数矩阵里）")
    # ⚠️ 必须丢掉没有表达量的基因。GTF 有 62,754 个基因，计数矩阵只有 3.7 万/5.8 万，
    #    留下的 expr=NaN 会让匹配距离变成 NaN，argmin 取到 NaN 行 —— 匹配静默失效。
    d = L13.strata_table(feats.dropna(subset=["expr"]))

    tag = sys.argv[2] if len(sys.argv) > 2 else "pooled"
    res = pd.read_csv(f"{D}/02_deg/{tag}.tsv", sep="\t", index_col=0)
    sets, qc = build(res, d)
    print(f"\n【{gse} / {tag}】  DEG 表 {len(res)} 基因")
    for kind, df in sets.items():
        df.to_csv(f"{outdir}/label_{kind}_{tag}.tsv", sep="\t", index=False)
        if tag == "pooled":      # 主口径同时写一份无后缀，下游默认读它
            df.to_csv(f"{outdir}/label_{kind}.tsv", sep="\t", index=False)
    qc.insert(0, "contrast", tag)
    print(qc.to_string(index=False))
    # 每个对照单独一份 —— 用 append 会在重跑时叠出重复行（第一次就踩过）
    qc.to_csv(f"{outdir}/label_matching_qc_{tag}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
