#!/usr/bin/env python3
"""72_expression_baseline.py —— 涡虫 expression-only 基线（A 任务书 §0）

回答：九特征信号是否只是基础表达量/易检出性的代理。
三个模型共用一套分折：
  EXPR   log10(平均 CPM + 1)，1 列
  NINE   九个特征，9 列
  JOINT  九特征 + 同一个表达量，10 列

流程与 70_len_increment.py 同构：复现点估计 → 配对簇重采样 2000 次，
同一次抽样内算 d(joint−expr) 与 d(joint−nine)。

用法:
  python scripts/72_expression_baseline.py \
    --labels data_t01 --counts "$COUNTS_TSV" --out out_expression_20261122
三个参数都有默认值，直接运行也可。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib.util
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_s = importlib.util.spec_from_file_location("m70", os.path.join(HERE, "scripts", "70_len_increment.py"))
M70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M70)
FEATS = M70.FEATS

EXPECT = {  # 归档值（70 已硬断言复现过）
    "down": dict(n=141, n_pos=72, n_neg=69, n_clu=115, nine=0.7254),
    "both": dict(n=220, n_pos=108, n_neg=112, n_clu=188, nine=0.6731),
}
RAW_ROWS = {"down": 166, "both": 288}


def load_expr(a, gene_ids):
    """平均 CPM：每样本先转 CPM，再对样本取平均（与 run_t01 匹配口径一致）。
    baseMean 仅作可选敏感性（任务书 §9），不是主分析口径。"""
    if a.expr_source == "baseMean":
        fm = pd.read_csv(a.features_merged, sep="\t").set_index("gene_id")
        v = pd.to_numeric(fm.reindex(gene_ids).expr, errors="coerce")
        print(f"  baseMean: 标签基因缺失 {int(v.isna().sum())}")
        assert not v.isna().any(), "features_merged 中缺表达式值"
        return v
    counts = pd.read_csv(a.counts, sep="\t", index_col=0)
    assert counts.shape[1] == 38, f"counts 列数 {counts.shape[1]}，应为 38 个样本"
    assert counts.index.is_unique, "counts 的 gene_id 有重复"
    s = counts.sum(axis=0)
    assert (s > 0).all(), f"有样本列之和为 0: {list(s[s <= 0].index)}"
    cpm = counts.div(s, axis=1).mul(1e6).mean(axis=1)
    miss = [g for g in gene_ids if g not in cpm.index]
    print(f"  counts: {counts.shape[0]} 基因 × {counts.shape[1]} 样本，标签基因缺失 {len(miss)}")
    assert not miss, f"标签基因不在 counts.tsv 中: {miss[:10]}"
    return cpm


def main():
    ap = argparse.ArgumentParser()
    def pick(*paths):  # 先找随包文件，再退回项目内/家目录的原始位置
        return next((p for p in paths if os.path.exists(p)), paths[-1])

    ap.add_argument("--labels", default=os.path.join(HERE, "data_t01"))
    ap.add_argument("--counts", default=os.environ.get(
        "COUNTS_TSV", pick(os.path.join(HERE, "counts.tsv"),
                           os.path.expanduser("~/workspace/database/IE-Tx/results/counts.tsv"))))
    ap.add_argument("--out", default=os.path.join(HERE, "out_expression_20261122"))
    ap.add_argument("--expr-source", choices=["cpm", "baseMean"], default="cpm",
                    help="baseMean 仅作敏感性分析（任务书 §9）")
    ap.add_argument("--features-merged", default=pick(
        os.path.join(HERE, "features_merged.tsv"),
        os.path.join(HERE, "..", "交付_20261031", "02_T01双向口径", "data", "features_merged.tsv")))
    a = ap.parse_args()
    sfx = "" if a.expr_source == "cpm" else "_baseMean"
    os.makedirs(a.out, exist_ok=True)
    print(f"labels={a.labels}\nexpr_source={a.expr_source}\ncounts={a.counts}\nout={a.out}")
    print(f"python={__import__('sys').version.split()[0]} sklearn={sklearn.__version__} "
          f"numpy={np.__version__} pandas={pd.__version__}")

    # ── 预检：先把两套标签都读入并核对行数
    label_tabs = {}
    for tag in ("down", "both"):
        lb = pd.read_csv(f"{a.labels}/label_{tag}_tp3_cpm.tsv", sep="\t")
        assert not lb.gene_id.duplicated().any(), "gene_id 重复"
        assert len(lb) == RAW_ROWS[tag], f"{tag} 标签表 {len(lb)} 行，应为 {RAW_ROWS[tag]}"
        label_tabs[tag] = lb
        print(f"  标签 {tag}: {len(lb)} 行，gene_id 无重复 ✓")
    all_ids = pd.concat([t.gene_id for t in label_tabs.values()]).unique()
    cpm = load_expr(a, all_ids)

    summary, reason_rows = [], []
    for tag, lb in label_tabs.items():
        # ── 与长度分析完全同一行掩码；不按 expression 过滤
        x_all = np.log10(cpm.reindex(lb.gene_id).values + 1)
        assert np.isfinite(x_all).all(), "有标签基因的表达量非有限"
        assert (x_all >= 0).all(), "log10(CPM+1) 出现负值"
        X9 = lb[FEATS].astype(float).values
        e = np.log10(pd.to_numeric(lb.cds_len_longest, errors="coerce").values + 1)
        ok = ~np.isnan(X9).any(axis=1) & np.isfinite(e)

        X9, e, x, y = X9[ok], e[ok], x_all[ok], lb.label.values[ok]
        g = M70.group_ids(lb.cluster.values[ok])
        ex = EXPECT[tag]
        assert len(y) == ex["n"] and int(y.sum()) == ex["n_pos"] \
            and int((y == 0).sum()) == ex["n_neg"] \
            and len(np.unique(g)) == ex["n_clu"], \
            f"{tag} 行/簇数不符: n={len(y)} pos={int(y.sum())} neg={int((y == 0).sum())} " \
            f"clu={len(np.unique(g))}"

        Xs = [x.reshape(-1, 1), X9, np.column_stack([X9, x])]
        sp = M70.splits_of(X9, y, g)
        fold = np.full(len(y), -1)
        for k, (_, te) in enumerate(sp):
            fold[te] = k
        assert (fold >= 0).all(), "有基因没分到折"
        for c in np.unique(g):
            assert len(np.unique(fold[g == c])) == 1, f"簇 {c} 被拆到多个测试折"

        oof = [M70.oof_of(X, y, sp) for X in Xs]
        auc = [float(roc_auc_score(y, o)) for o in oof]
        assert abs(auc[1] - ex["nine"]) <= 0.002, (
            f"{tag} 九特征未复现 {ex['nine']}：得 {auc[1]:.4f}")
        print(f"[{tag}] n={len(y)} 正{int(y.sum())}/负{int((y == 0).sum())} 簇 {len(np.unique(g))}  "
              f"EXPR {auc[0]:.4f}  9F {auc[1]:.4f}（✓ 复现 {ex['nine']}）  JOINT {auc[2]:.4f}  "
              f"Δ(J−E) {auc[2]-auc[0]:+.4f}  Δ(J−9F) {auc[2]-auc[1]:+.4f}")

        pd.DataFrame({"gene_id": lb.gene_id.values[ok], "label": y, "cluster_id": g,
                      "fold": fold, "p_expr": oof[0], "p_nine": oof[1],
                      "p_joint": oof[2]}) \
            .to_csv(f"{a.out}/oof_genes_expression_{tag}{sfx}.tsv", sep="\t", index=False)

        # ── 配对簇重采样
        rng = np.random.default_rng(M70.SEED)
        uc = np.unique(g)
        bucket = {c: np.flatnonzero(g == c) for c in uc}
        rows = []
        for _ in range(M70.N_BOOT):
            idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
            st, au, why = M70.attempt(idx, Xs, y, g)
            rows.append({"status": st,
                         "auc_expr": au[0] if au else np.nan,
                         "auc_nine": au[1] if au else np.nan,
                         "auc_joint": au[2] if au else np.nan,
                         "d_joint_minus_expr": au[2] - au[0] if au else np.nan,
                         "d_joint_minus_nine": au[2] - au[1] if au else np.nan,
                         "reason": why or ""})
        bt = pd.DataFrame(rows)
        bt.insert(0, "attempt", np.arange(1, M70.N_BOOT + 1))
        bt.to_csv(f"{a.out}/bootstrap_expression_{tag}{sfx}.tsv", sep="\t", index=False)
        v = bt[bt.status == "valid"]
        n_ok = len(v)
        print(f"    重采样 {M70.N_BOOT} 次，有效 {n_ok}"
              + (f"，失败 {bt.reason[bt.status != 'valid'].value_counts().to_dict()}"
                 if n_ok < M70.N_BOOT else "，失败 0"))
        if n_ok < 1900:
            print(f"    ⚠️ 有效抽样 {n_ok} < 1900，报告需标注")
        for why, k in bt.reason[bt.status != "valid"].value_counts().items():
            reason_rows.append({"label_set": tag, "reason": why, "n": int(k)})
        assert np.isfinite(v.auc_joint.values).all(), "有效抽样出现 NaN auc_joint"

        def ci(col):
            return float(np.percentile(v[col], 2.5)), float(np.percentile(v[col], 97.5))

        (el, eh), (nl, nh), (jl, jh) = ci("auc_expr"), ci("auc_nine"), ci("auc_joint")
        (l1, h1), (l2, h2) = ci("d_joint_minus_expr"), ci("d_joint_minus_nine")
        summary.append({
            "label_set": tag, "n": len(y), "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()), "n_clusters": len(np.unique(g)),
            "auc_expr": round(auc[0], 4), "auc_expr_lo": round(el, 4), "auc_expr_hi": round(eh, 4),
            "auc_nine": round(auc[1], 4), "auc_nine_lo": round(nl, 4), "auc_nine_hi": round(nh, 4),
            "auc_joint": round(auc[2], 4), "auc_joint_lo": round(jl, 4), "auc_joint_hi": round(jh, 4),
            "d_joint_minus_expr": round(auc[2] - auc[0], 4),
            "d_jme_lo": round(l1, 4), "d_jme_hi": round(h1, 4),
            "d_joint_minus_nine": round(auc[2] - auc[1], 4),
            "d_jmn_lo": round(l2, 4), "d_jmn_hi": round(h2, 4),
            "n_boot": M70.N_BOOT, "n_valid": n_ok, "sklearn_version": sklearn.__version__,
        })
        print(f"    Δ(J−E) {auc[2]-auc[0]:+.4f} 95%CI [{l1:+.4f}, {h1:+.4f}]  "
              f"Δ(J−9F) {auc[2]-auc[1]:+.4f} 95%CI [{l2:+.4f}, {h2:+.4f}]")
        print(f"    P(Δ(J−E)≤0) = {(v.d_joint_minus_expr <= 0).mean()*100:.2f}%  "
              f"P(Δ(J−9F)≤0) = {(v.d_joint_minus_nine <= 0).mean()*100:.2f}%  "
              f"中位数 Δ(J−E) = {v.d_joint_minus_expr.median():+.4f}")

    s = pd.DataFrame(summary)
    s.to_csv(f"{a.out}/summary_expression{sfx}.tsv", sep="\t", index=False)
    pd.DataFrame(reason_rows or [{"label_set": "-", "reason": "无失败", "n": 0}]) \
        .to_csv(f"{a.out}/bootstrap_reasons_expression{sfx}.tsv", sep="\t", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
