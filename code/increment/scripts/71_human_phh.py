#!/usr/bin/env python3
"""71_human_phh.py —— 跨数据集扩展（任务书 §5 末条）

在人类原代肝细胞双向标签上，用与涡虫完全相同的三模型方案估计：
  LEN / 9F / JOINT 的折外 AUC，以及两个配对 ΔAUC 及其 95% 区间。

核心交付是涡虫两套标签；本结果是并列参考。
输入（随包）：data_human/{features.tsv, clusters.pkl, label_both.tsv}

用法: python scripts/71_human_phh.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "data_human")
OUT = os.path.join(HERE, "out")
COV_MIN = 0.5                     # 与 T-02 的 16_validation.py 一致

_s = importlib.util.spec_from_file_location("m70", os.path.join(HERE, "scripts", "70_len_increment.py"))
M70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M70)
FEATS = M70.FEATS


def main():
    feats = pd.read_csv(f"{D}/features.tsv", sep="\t").set_index("gene_id")
    clust = pd.read_pickle(f"{D}/clusters.pkl")
    lab = pd.read_csv(f"{D}/label_both.tsv", sep="\t")
    df = lab.merge(feats.reset_index(), on="gene_id", how="left")
    assert not df.gene_id.duplicated().any(), "gene_id 重复"
    print(f"GSE351189 标注 {len(df)}（正 {int(df.label.sum())}）")

    keep = [c for c in FEATS if df[c].notna().sum() > COV_MIN * len(df)]
    assert len(keep) == 9, f"覆盖率过滤掉了特征: {[c for c in FEATS if c not in keep]}"

    X9 = df[keep].astype(float).values
    e = np.log10(pd.to_numeric(df.cds_len_longest, errors="coerce").values + 1)
    ok = ~np.isnan(X9).any(axis=1) & np.isfinite(e)
    X9, e, y = X9[ok], e[ok], df.label.values[ok]

    # 簇：无同源的基因各自成组
    c = df.gene_id.values[ok]
    g = M70.group_ids(clust.reindex(c).fillna(-1).values)

    Xs = [e.reshape(-1, 1), X9, np.column_stack([X9, e])]
    sp = M70.splits_of(X9, y, g)
    oof = [M70.oof_of(X, y, sp) for X in Xs]
    auc = [float(roc_auc_score(y, o)) for o in oof]
    assert round(auc[1], 4) == 0.6789, (
        f"PHH 九特征未复现 T-02 的 0.6789：得 {auc[1]:.4f}，检查覆盖率过滤与行掩码")
    print(f"n={len(y)} 正{int(y.sum())}/负{int((y == 0).sum())} 簇 {len(np.unique(g))}  "
          f"LEN {auc[0]:.4f}  9F {auc[1]:.4f}（✓ 复现 0.6789）  JOINT {auc[2]:.4f}  "
          f"Δ(J−L) {auc[2]-auc[0]:+.4f}  Δ(J−9F) {auc[2]-auc[1]:+.4f}")

    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(sp):
        fold[te] = k
    assert (fold >= 0).all(), "有基因没分到折"
    pd.DataFrame({"gene_id": c, "label": y, "cluster_id": g, "fold": fold,
                  "p_length": oof[0], "p_nine": oof[1], "p_joint": oof[2]}) \
        .to_csv(f"{OUT}/oof_genes_GSE351189.tsv", sep="\t", index=False)

    rng = np.random.default_rng(M70.SEED)
    uc = np.unique(g)
    bucket = {u: np.flatnonzero(g == u) for u in uc}
    rows = []
    for _ in range(M70.N_BOOT):
        idx = np.concatenate([bucket[u] for u in rng.choice(uc, len(uc), replace=True)])
        st, a, why = M70.attempt(idx, Xs, y, g)
        rows.append({"status": st,
                     "auc_length": a[0] if a else np.nan,
                     "auc_nine": a[1] if a else np.nan,
                     "auc_joint": a[2] if a else np.nan,
                     "d_joint_minus_length": a[2] - a[0] if a else np.nan,
                     "d_joint_minus_nine": a[2] - a[1] if a else np.nan,
                     "reason": why or ""})
    bt = pd.DataFrame(rows)
    bt.insert(0, "attempt", np.arange(1, M70.N_BOOT + 1))
    bt.to_csv(f"{OUT}/bootstrap_attempts_GSE351189.tsv", sep="\t", index=False)
    v = bt[bt.status == "valid"]
    print(f"重采样 {M70.N_BOOT} 次，有效 {len(v)}"
          + (f"，失败原因 {bt.reason[bt.status != 'valid'].value_counts().to_dict()}"
             if len(v) < M70.N_BOOT else ""))

    def ci(col):
        return float(np.percentile(v[col], 2.5)), float(np.percentile(v[col], 97.5))

    lo1, hi1 = ci("d_joint_minus_length")
    lo2, hi2 = ci("d_joint_minus_nine")
    pd.DataFrame([{
        "label_set": "PHH_both", "n": len(y), "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()), "n_clusters": len(np.unique(g)),
        "auc_length": round(auc[0], 4), "auc_nine": round(auc[1], 4),
        "auc_joint": round(auc[2], 4),
        "d_joint_minus_length": round(auc[2] - auc[0], 4),
        "d_jml_lo": round(lo1, 4), "d_jml_hi": round(hi1, 4),
        "d_joint_minus_nine": round(auc[2] - auc[1], 4),
        "d_jmn_lo": round(lo2, 4), "d_jmn_hi": round(hi2, 4),
        "n_boot": M70.N_BOOT, "n_valid": len(v)}]) \
        .to_csv(f"{OUT}/summary_GSE351189.tsv", sep="\t", index=False)
    print(f"Δ(J−L) {auc[2]-auc[0]:+.4f} 95%CI [{lo1:+.4f}, {hi1:+.4f}]  "
          f"Δ(J−9F) {auc[2]-auc[1]:+.4f} 95%CI [{lo2:+.4f}, {hi2:+.4f}]")


if __name__ == "__main__":
    main()
