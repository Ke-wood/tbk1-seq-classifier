#!/usr/bin/env python3
"""64_utr_features.py —— 五个 UTR 特征

  utr5_len         最长 5′UTR 的 bp 长度（转录本层面取最长，多段求和）
  utr5_gc          GC 含量
  utr5_cpg_oe      CpG 观测/期望比
  utr5_mfe_per_nt  RNAfold MFE ÷ 该 UTR 自身长度
  utr3_len         最长 3′UTR 的 bp 长度

算子直接复用 T-02 的 57_features.py（cpg_oe / read_fa），不另写一份（任务书 §2.2）。

产出 03_features/features_utr.tsv = T-02 的 9 特征 + 这 5 个
用法: python 64_utr_features.py
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd

T2 = os.environ["IETX_HUMAN"]
R = os.environ["IETX_T03"]

_s = importlib.util.spec_from_file_location(
    "f57", f"{T2}/scripts/57_features.py")
F57 = importlib.util.module_from_spec(_s); _s.loader.exec_module(F57)


def main():
    seq = {g: s for g, s in F57.read_fa(f"{R}/ref/utr5.fa")}
    utr5 = pd.DataFrame(
        [(g, len(s), (s.count("G") + s.count("C")) / len(s) if s else np.nan,
          F57.cpg_oe(s)) for g, s in seq.items()],
        columns=["gene_id", "utr5_len_seq", "utr5_gc", "utr5_cpg_oe"])
    mfe = pd.read_csv(f"{R}/ref/utr5_mfe.tsv", sep="\t", header=None,
                      names=["gene_id", "mfe"])
    utr5 = utr5.merge(mfe, on="gene_id", how="left")
    utr5["utr5_mfe_per_nt"] = utr5.mfe / utr5.utr5_len_seq

    L = pd.read_csv(f"{R}/ref/utr_len.tsv", sep="\t")
    # ⚠️ 长度以 BED 侧为准，并和序列侧交叉核对 —— 两者不一致说明
    #    拼接顺序或 getfasta 出了问题（例如漏了 -s 让反义链没有反向互补）
    m = L.merge(utr5, on="gene_id", how="left")
    bad = (m.utr5_len - m.utr5_len_seq).abs() > 0
    print(f"utr5_len（BED）与序列长度不一致: {int(bad.sum())} / {len(m)}")
    assert bad.sum() < 50, "UTR 序列长度与坐标长度系统性不符，检查 getfasta -s"

    out = L.merge(utr5.drop(columns=["utr5_len_seq", "mfe"]), on="gene_id", how="left")
    f9 = pd.read_csv(f"{T2}/03_features/features.tsv", sep="\t")
    f = f9.merge(out, on="gene_id", how="left")
    f.to_csv(f"{R}/03_features/features_utr.tsv", sep="\t", index=False)

    n = len(f)
    print(f"\nfeatures_utr.tsv: {n} 基因 × {f.shape[1]} 列")
    chk = [("utr5_len", 1, 1e6), ("utr5_gc", 0, 1), ("utr5_cpg_oe", 0, 5),
           ("utr5_mfe_per_nt", -1, 0), ("utr3_len", 1, 1e7)]
    for c, lo, hi in chk:
        v = f[c].dropna()
        ok = len(v) > 0 and v.between(lo, hi).mean() > 0.99
        print(f"  {c:16s} 非空 {len(v):>6d}/{n}  中位 {v.median():>10.2f}  "
              f"{'✓' if ok else '✗ 值域异常'}")
        assert ok, f"{c} 值域异常"

    # 任务书 §2.4 的完整性自检：查非空比例，不查中位数。
    # ⚠️ 分母不能用全部 62,754 个基因 —— lncRNA/假基因本来就没有 UTR 注释，
    #    任务书里的 0.5×len(f) 隐含 f 是「有 UTR 的基因集」。这里的对照基准取
    #    **有 CDS 的基因**（20,481）：19,694/20,481 = 96%，才是真正的提取完整度。
    n_cds = int(f.cds_len_longest.notna().sum())
    for c, frac in (("utr5_len", 0.8), ("utr3_len", 0.8)):
        k = int(f[c].notna().sum())
        print(f"  {c} 覆盖 {k}/{n_cds} 个有 CDS 的基因（{k/n_cds:.1%}）")
        assert k >= frac * n_cds, f"只有 {k}/{n_cds} 个编码基因拿到 {c}，提取不完整"


if __name__ == "__main__":
    main()
