#!/usr/bin/env python3
"""56_deseq.py —— T-02 第 2 步：差异表达 + CPM 表达量

两个数据集各跑一遍，对照如下：
  GSE351189  PHH，18 样本（neg/IFNa/IFNγ × 6h/72h × 3）
    ifna_6h / ifng_6h / ifna_72h / ifng_72h   每个 n=3 vs 3，design ~condition
    pooled                                    IFN(α或γ) vs neg，design ~timepoint + treated（n=9 vs 9）
  GSE309167  iMacs，12 样本（IFNa2/IFNb/IFNw/UT × 3）
    ifna2 / ifnb / ifnw                      每个 n=3 vs 3
    pooled                                    任一 I 型 IFN vs UT（n=9 vs 3）

⚠️ 主口径取 pooled：与涡虫侧「一个扰动 vs 一个对照」最可比，且样本量最大。
   逐个扰动只作稳健性，不替代主口径。两层都出，不挑。
⚠️ expr 用 counts 的平均 CPM（与 13_labels.py 同定义），不是 DEG 的 baseMean。
   涡虫侧踩过这个坑：baseMean 会让负类匹配与原始口径只重合 19/83。

用法: env/bin/python 56_deseq.py GSE351189
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{R}/raw"

SETS = {
    "GSE351189": dict(file="GSE351189_PHH_Count.csv.gz",
                      samples=None),   # 列名即样本名
    "GSE309167": dict(file="GSE309167_iPSC_Macrophage_Treatment_umi_counts.csv.gz",
                      samples="Kolf2"),
}


def load_counts(gse):
    cfg = SETS[gse]
    d = pd.read_csv(f"{RAW}/{cfg['file']}", index_col=0)
    if cfg["samples"]:
        d = d[[c for c in d.columns if str(c).startswith(cfg["samples"])]]
    d.index = d.index.astype(str).str.split(".").str[0]
    d.index.name = "gene_id"
    return d.astype(float)


def run(counts, meta, design, contrast, tag, outdir):
    c = counts.loc[counts.sum(axis=1) > 0].T
    dds = DeseqDataSet(counts=c, metadata=meta, design=design,
                       refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    r = st.results_df.copy()
    r.index.name = "gene_id"
    r.to_csv(f"{outdir}/{tag}.tsv", sep="\t")
    sig = ((r.padj < 0.05) & (r.log2FoldChange.abs() > 0.5))
    print(f"  [{tag}] {c.shape[1]} 基因 × {c.shape[0]} 样本  "
          f"padj 非空 {r.padj.notna().sum()}  显著 {int(sig.sum())}"
          f"（上 {int((sig & (r.log2FoldChange > 0)).sum())} / "
          f"下 {int((sig & (r.log2FoldChange < 0)).sum())}）")
    return r


def main(gse):
    out = f"{R}/人类侧_{gse}/02_deg"
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{R}/人类侧_{gse}/01_qc", exist_ok=True)
    if os.path.exists(f"{out}/pooled.tsv"):
        print(f"{gse}: pooled.tsv 已存在，跳过（删掉可重跑）")
        return
    counts = load_counts(gse)
    print(f"{gse}: {counts.shape[0]} 基因 × {counts.shape[1]} 样本")

    # CPM 表达量（全数据集平均）—— 负类匹配与面板里的 expr 都用这一列
    cpm = counts.div(counts.sum(axis=0), axis=1).mul(1e6)
    expr = cpm.mean(axis=1).rename("expr")
    expr.to_csv(f"{R}/人类侧_{gse}/01_qc/cpm_mean.tsv", sep="\t")

    if gse == "GSE351189":
        meta = pd.DataFrame({
            "sample": counts.columns,
            "timepoint": [c.split("_")[-1] for c in counts.columns],
            "treatment": [("neg" if "neg" in c else
                           "IFNa" if "IFNa" in c else "IFNg") for c in counts.columns],
        }).set_index("sample")
        # 逐个扰动 / 时点
        for tp in ("6h", "72h"):
            for tr in ("IFNa", "IFNg"):
                sub = meta[(meta.timepoint == tp) &
                           meta.treatment.isin([tr, "neg"])].copy()
                r = run(counts[sub.index], sub[["treatment"]].rename(columns={"treatment": "condition"}),
                        "~condition", ["condition", tr, "neg"],
                        f"{tr.lower()}_{tp}", out)
        # pooled：IFN vs neg，时点作协变量
        p = meta.copy(); p["treated"] = np.where(p.treatment == "neg", "neg", "IFN")
        run(counts[p.index], p[["treated", "timepoint"]], "~timepoint + treated",
            ["treated", "IFN", "neg"], "pooled", out)
        meta.to_csv(f"{R}/人类侧_{gse}/01_qc/sample_meta.tsv", sep="\t")
    else:
        meta = pd.DataFrame({
            "sample": counts.columns,
            "treatment": [("IFNa2" if "IFNa2" in c else "IFNb" if "IFNb" in c
                           else "IFNw" if "IFNw" in c else "UT") for c in counts.columns],
        }).set_index("sample")
        for tr in ("IFNa2", "IFNb", "IFNw"):
            sub = meta[meta.treatment.isin([tr, "UT"])].copy()
            run(counts[sub.index], sub[["treatment"]].rename(columns={"treatment": "condition"}),
                "~condition", ["condition", tr, "UT"], tr.lower(), out)
        p = meta.copy(); p["treated"] = np.where(p.treatment == "UT", "UT", "IFN")
        run(counts[p.index], p[["treated"]], "~treated",
            ["treated", "IFN", "UT"], "pooled", out)
        meta.to_csv(f"{R}/人类侧_{gse}/01_qc/sample_meta.tsv", sep="\t")

    print(f"→ {out}")


if __name__ == "__main__":
    main(sys.argv[1])
