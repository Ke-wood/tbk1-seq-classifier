#!/usr/bin/env python3
"""补充实验 2 —— 最长 CDS 长度与九特征模型的增量预测价值

输入（随包）：data_t01/label_down_tp3_cpm.tsv、data_t01/label_both_tp3_cpm.tsv
三个模型共用一次分折：
  LEN    log10(cds_len_longest + 1)
  9F     prom_gc prom_cpg_oe prom_lowcomp mfe_per_nt gene_span
         n_transcripts cds_gc cds_cpg_oe cai
  JOINT  9F + 同一条对数长度

流程：复现六个点估计 → 按同源簇配对重采样 2000 次，每次重新分折并拟合三模型，
在同一次抽样内算 Δ(JOINT−LEN) 与 Δ(JOINT−9F)。

用法: python scripts/70_len_increment.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data_t01")
OUT = os.path.join(HERE, "out")

FEATS = ["prom_gc", "prom_cpg_oe", "prom_lowcomp", "mfe_per_nt", "gene_span",
         "n_transcripts", "cds_gc", "cds_cpg_oe", "cai"]
N_BOOT, SEED = 2000, 0
# 归档点估计（四位小数），用于硬断言复现
EXPECT = {
    "down": dict(n=141, n_pos=72, n_neg=69, n_clu=115,
                 len=0.7375, nine=0.7254, joint=0.8937),
    "both": dict(n=220, n_pos=108, n_neg=112, n_clu=188,
                 len=0.6571, nine=0.6731, joint=0.7563),
}


def mk():
    """归档模型：StandardScaler + L2 逻辑回归，C=1.0、balanced、max_iter=3000。"""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))


def group_ids(cluster):
    """cluster<0（无同源）各自成组，编号 1e6+序号。
    ⚠️ 全部填 -1 会把单例并成一个巨组，GroupKFold 退化成按簇切，AUC 虚高。
    ⚠️ 编号必须在重采样前一次性固定；重采样里同一原簇的副本共用该编号。"""
    c = np.asarray(cluster, dtype=float)
    g = c.copy()
    sing = np.flatnonzero(c < 0)
    g[sing] = 10 ** 6 + np.arange(len(sing))
    return g.astype(int)


def splits_of(X, y, g):
    """唯一一次分折。三模型共用；重采样时每抽一次也只用这一次。"""
    return list(GroupKFold(n_splits=5).split(X, y, g))


def oof_of(X, y, splits):
    """折外正类概率。每折只在训练集上拟合并标准化。"""
    oof = np.zeros(len(y))
    cnt = np.zeros(len(y), dtype=int)
    for tr, te in splits:
        oof[te] += mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        cnt[te] += 1
    assert (cnt == 1).all(), "有基因没拿到折外预测或拿到多次"
    return oof


def one_fit_set(Xs, y, splits):
    """输入矩阵列表 → 每个模型的折外 AUC。"""
    return [roc_auc_score(y, oof_of(X, y, splits)) for X in Xs]


def attempt(idx, Xs, y, g):
    """一次配对抽样。返回 (状态, [三个 AUC] 或 None, 失败原因 或 None)。

    idx 里同一原簇的副本保留相同簇编号，GroupKFold 因此不会把它们拆到训练/测试两侧。
    """
    yy = y[idx]
    if len(np.unique(g[idx])) < 5:
        return "invalid", None, "独立簇不足5个"
    if len(np.unique(yy)) < 2:
        return "invalid", None, "汇总标签只剩一类"
    sp = splits_of(Xs[0][idx], yy, g[idx])
    for tr, te in sp:
        if len(np.unique(yy[tr])) < 2:
            return "invalid", None, "训练折只有一类"
    try:
        auc = one_fit_set([X[idx] for X in Xs], yy, sp)
    except Exception:
        return "invalid", None, "预测不完整"
    if not np.all(np.isfinite(auc)):
        return "invalid", None, "预测不完整"
    return "valid", auc, None


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, boot_all, reason_rows = [], [], []

    for tag in ("down", "both"):
        lb = pd.read_csv(f"{SRC}/label_{tag}_tp3_cpm.tsv", sep="\t")
        assert not lb.gene_id.duplicated().any(), "gene_id 重复"

        # ── 固定行：九特征与长度共同取完整、有限值。归档表在这两套标签上
        #    两套过滤的分母恰好相同，但仍显式同时判定（任务书 §3.1）。
        X9 = lb[FEATS].astype(float).values
        e = np.log10(pd.to_numeric(lb.cds_len_longest, errors="coerce").values + 1)
        ok = ~np.isnan(X9).any(axis=1) & np.isfinite(e)
        assert (e[ok] > 0).all(), "长度变换后出现 <=0，检查变换口径"

        X9, e, y = X9[ok], e[ok], lb.label.values[ok]
        g = group_ids(lb.cluster.values[ok])
        Xs = [e.reshape(-1, 1), X9, np.column_stack([X9, e])]

        # ── 复现点估计 + 折号/折外预测存档
        sp = splits_of(X9, y, g)
        fold = np.full(len(y), -1)
        for k, (_, te) in enumerate(sp):
            fold[te] = k
        assert (fold >= 0).all(), "有基因没分到折"
        for c in np.unique(g):
            assert len(np.unique(fold[g == c])) == 1, f"簇 {c} 被拆到多个测试折"

        oof = [oof_of(X, y, sp) for X in Xs]
        auc = [roc_auc_score(y, o) for o in oof]
        ex = EXPECT[tag]
        assert len(y) == ex["n"] and int(y.sum()) == ex["n_pos"] and len(np.unique(g)) == ex["n_clu"], "行/簇数不符"
        for nm, a, want in zip(("length", "nine", "joint"), auc,
                               (ex["len"], ex["nine"], ex["joint"])):
            assert round(a, 4) == want, f"{tag} {nm} 复现失败: {a:.4f} != {want}"
        print(f"[{tag}] n={len(y)} 正{int(y.sum())}/负{int((y == 0).sum())} 簇 {len(np.unique(g))}  "
              f"LEN {auc[0]:.4f}  9F {auc[1]:.4f}  JOINT {auc[2]:.4f}  "
              f"Δ(J−L) {auc[2]-auc[0]:+.4f}  Δ(J−9F) {auc[2]-auc[1]:+.4f}  ✓ 复现")

        pd.DataFrame({"gene_id": lb.gene_id.values[ok], "label": y, "cluster_id": g,
                      "fold": fold, "p_length": oof[0], "p_nine": oof[1],
                      "p_joint": oof[2]}) \
            .to_csv(f"{OUT}/oof_genes_{tag}.tsv", sep="\t", index=False)

        # ── 配对簇级重采样
        rng = np.random.default_rng(SEED)
        uc = np.unique(g)
        bucket = {c: np.flatnonzero(g == c) for c in uc}
        rows = []
        for _ in range(N_BOOT):
            draw = rng.choice(uc, len(uc), replace=True)
            idx = np.concatenate([bucket[c] for c in draw])
            st, a, why = attempt(idx, Xs, y, g)
            rows.append({"status": st,
                         "auc_length": a[0] if a else np.nan,
                         "auc_nine": a[1] if a else np.nan,
                         "auc_joint": a[2] if a else np.nan,
                         "d_joint_minus_length": a[2] - a[0] if a else np.nan,
                         "d_joint_minus_nine": a[2] - a[1] if a else np.nan,
                         "reason": why or ""})
        bt = pd.DataFrame(rows)
        bt.insert(0, "attempt", np.arange(1, N_BOOT + 1))
        bt.to_csv(f"{OUT}/bootstrap_attempts_{tag}.tsv", sep="\t", index=False)
        boot_all.append(bt)
        n_ok = int((bt.status == "valid").sum())
        print(f"    重采样 {N_BOOT} 次，有效 {n_ok}，失败 {(bt.status != 'valid').sum()}"
              + (f" {bt.reason[bt.status != 'valid'].value_counts().to_dict()}" if n_ok < N_BOOT else ""))
        for why, k in bt.reason[bt.status != "valid"].value_counts().items():
            reason_rows.append({"label_set": tag, "reason": why, "n": int(k)})

        v = bt[bt.status == "valid"]

        def ci(col):
            return float(np.percentile(v[col], 2.5)), float(np.percentile(v[col], 97.5))

        lo1, hi1 = ci("d_joint_minus_length")
        lo2, hi2 = ci("d_joint_minus_nine")
        summary.append({
            "label_set": tag, "n": len(y), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "n_clusters": len(np.unique(g)),
            "auc_length": round(auc[0], 4), "auc_nine": round(auc[1], 4), "auc_joint": round(auc[2], 4),
            "d_joint_minus_length": round(auc[2] - auc[0], 4),
            "d_jml_lo": round(lo1, 4), "d_jml_hi": round(hi1, 4),
            "d_joint_minus_nine": round(auc[2] - auc[1], 4),
            "d_jmn_lo": round(lo2, 4), "d_jmn_hi": round(hi2, 4),
            "n_boot": N_BOOT, "n_valid": n_ok,
        })
        print(f"    Δ(J−L) {auc[2]-auc[0]:+.4f} 95%CI [{lo1:+.4f}, {hi1:+.4f}]  "
              f"Δ(J−9F) {auc[2]-auc[1]:+.4f} 95%CI [{lo2:+.4f}, {hi2:+.4f}]")

    s = pd.DataFrame(summary)
    s.to_csv(f"{OUT}/summary.tsv", sep="\t", index=False)
    pd.DataFrame(reason_rows or [{"label_set": "-", "reason": "无失败", "n": 0}]) \
        .to_csv(f"{OUT}/bootstrap_reasons.tsv", sep="\t", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
