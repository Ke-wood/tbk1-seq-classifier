#!/usr/bin/env python3
"""60_compare.py —— T-02 第 6 步：与涡虫侧同口径对比表 + 判据分档

读两个人类数据集的 06_model/model_cv.tsv，拼上涡虫侧基准（任务书 §12），
按双向口径的「9 特征 − 长度」差值分档 A/B/C/D（§4）。

用法: env/bin/python 60_compare.py
"""
import os
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 涡虫侧基准（任务书 §12；来源 rerun_chain2_20260927.log）
PLANARIA = [
    dict(dataset="涡虫（单向 down）", label="down", auc9=0.7254, auc_len=0.7375,
         auc_len_common=0.6379, delta_9_minus_len=-0.0118,
         common="[697, 13615]", n_pos=72, n_neg=69),
    dict(dataset="涡虫（双向 both）", label="both", auc9=0.6731, auc_len=0.6571,
         auc_len_common=0.5903, delta_9_minus_len=+0.0160,
         common="[412, 13615]", n_pos=108, n_neg=112),
]


def grade(delta):
    if delta is None or delta != delta:
        return "—"
    if delta <= 0.05:
        return "A 同样长度主导"
    if delta <= 0.15:
        return "B 部分长度主导"
    return "C 不长度主导"


def main():
    rows = list(PLANARIA)
    for gse in ("GSE351189", "GSE309167"):
        p = f"{R}/人类侧_{gse}/06_model/model_cv.tsv"
        if not os.path.exists(p):
            print(f"  跳过 {gse}（缺 {p}）"); continue
        m = pd.read_csv(p, sep="\t")
        for _, r in m.iterrows():
            rows.append(dict(
                dataset=f"{gse}（{r.label}）", label=r.label,
                auc9=round(r.auc9, 4), auc_len=round(r.auc_len, 4),
                auc_len_common=round(r.auc_len_common, 4),
                delta_9_minus_len=round(r.delta_9_minus_len, 4),
                common=f"[{r.common_lo:.0f}, {r.common_hi:.0f}]",
                n_pos=int(r.n_pos), n_neg=int(r.n_neg),
                auc_ci=f"[{r.auc9_ci_lo:.3f}, {r.auc9_ci_hi:.3f}]",
                perm_p=r.auc9_vs_perm_p))
    d = pd.DataFrame(rows)
    d["判定"] = [grade(x) for x in d.delta_9_minus_len]
    # §4 的 D 档：9 特征 AUC < 0.6
    d.loc[(d.auc9 < 0.6), "判定"] = "D 模型接近随机"
    out = f"{R}/07_compare"
    os.makedirs(out, exist_ok=True)
    d.to_csv(f"{out}/cross_species_comparison.tsv", sep="\t", index=False)
    print(d.to_string(index=False))
    print(f"\n→ {out}/cross_species_comparison.tsv")


if __name__ == "__main__":
    main()
