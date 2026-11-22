#!/usr/bin/env python3
"""61_qc.py —— 样本与分组核对（§7 步骤 1 的 qc.tsv）

只做三件事：文库大小、检出基因数、样本间 log2 CPM 相关中位。
分组与扰动列从 56_deseq.py 写的 sample_meta.tsv 取。

用法: env/bin/python 61_qc.py
"""
import os
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{R}/raw"
FILES = {"GSE351189": "GSE351189_PHH_Count.csv.gz",
         "GSE309167": "GSE309167_iPSC_Macrophage_Treatment_umi_counts.csv.gz"}

for gse, fn in FILES.items():
    d = pd.read_csv(f"{RAW}/{fn}", index_col=0)
    meta = pd.read_csv(f"{R}/人类侧_{gse}/01_qc/sample_meta.tsv",
                       sep="\t", index_col=0)
    # ⚠️ 不能靠「第一行是不是字符串」来挑计数列 —— GSE309167 的注释列
    #    (entrezgene_id 等) 首值是数字，会被误当成样本列（实测多出 3 列）。
    #    样本列以 sample_meta 的索引为准。
    d = d[meta.index]
    lc = np.log2(d.div(d.sum(0), axis=1).mul(1e6) + 1)
    cm = lc.corr()
    q = pd.DataFrame({
        "sample": d.columns, "group": meta.iloc[:, 0].values,
        "lib_size": d.sum(0).astype(int).values,
        "n_detected": (d > 0).sum(0).values,
        "corr_median": [cm.loc[c].drop(c).median().round(4) for c in d.columns],
    })
    out = f"{R}/人类侧_{gse}/01_qc/qc.tsv"
    q.to_csv(out, sep="\t", index=False)
    print(f"{gse}: {len(q)} 样本  文库 {q.lib_size.min()/1e6:.1f}–{q.lib_size.max()/1e6:.1f} M  "
          f"检出基因 {q.n_detected.min()}–{q.n_detected.max()}  "
          f"相关中位 {q.corr_median.min():.3f}–{q.corr_median.max():.3f}")
    assert (q.corr_median > 0.8).all(), "有样本与其余样本相关性 <0.8，检查分组/批次"
print("QC 通过")
