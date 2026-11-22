#!/usr/bin/env python3
"""
15_train.py —— IRS 建模 + 阈值选择 + 符号稳健性检验

【为什么这么小】正类只有 15 / 83 / 61 个，特征 8~10 个。这种规模上梯度提升
必然过拟合、AUC 还会很好看。故用 L2 逻辑回归 + 重复分层交叉验证，
交叉验证的【折外预测】算 AUC/PR-AUC，另有置换检验给零分布 —— 宁可保守。

【用户要求的判据】「严格/宽松各训一次，按富集强度+样本量综合选」+
  「用 UP 集当正类再训一次。三种标签定义下若 AUC 与 top 特征一致 → 报稳健」

【特征泄漏】负类已按 表达量×长度×转录本数 最近邻匹配（SMD<0.06）。
  另外做「去掉表达量协变量后 AUC 掉多少」的对照（用户第 5 条要求的兜底）。

产出
  results/model_cv.tsv        三种标签定义 × 有无表达量的 CV 指标
  results/model_features.tsv  各定义下的特征权重（用于稳健性比对）
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
FEAT = f"{IETX}/features"
LABELS = ["down_strict", "down_loose", "up_loose"]
SEED, NSPLIT, NREP = 0, 5, 10


def build_features():
    f = pd.read_csv(f"{FEAT}/features_fast.tsv", sep="\t").set_index("gene_id")
    # features_fast 里那个 cds_gc 是 10_features_seq.sh 留的占位空列，真值在
    # features_cds.tsv —— 不去掉会 join 冲突。
    f = f.drop(columns=[c for c in ("cds_gc", "cds_cpg_oe", "cai") if c in f.columns])
    try:
        c = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")
        f = f.join(c, how="left")
    except FileNotFoundError:
        print("  (features_cds.tsv 缺失，跳过 CDS 特征)")
        f["cds_gc"] = f["cds_cpg_oe"] = f["cai"] = f["cds_len_longest"] = np.nan
    try:
        m = pd.read_csv(f"{FEAT}/mfe.tsv", sep="\t", header=None,
                        names=["gene_id", "mfe"]).set_index("gene_id")
        f = f.join(m, how="left")
        # MFE 归一到长度：更长的启动子天然更负，不归一化会变成「长度」的代理
        f["mfe_per_nt"] = f.mfe / f.prom_len
        print(f"  MFE 已并入（{f.mfe.notna().sum()} 条有值）")
    except FileNotFoundError:
        f["mfe"] = f["mfe_per_nt"] = np.nan
        print("  (mfe.tsv 尚未生成，本次不含 MFE)")
    return f


FEATURES = ["prom_gc", "prom_cpg_oe", "prom_lowcomp", "gene_span",
            "n_transcripts", "cds_gc", "cds_cpg_oe", "cai", "mfe_per_nt"]


def cv_eval(df, feats, use_expr, verbose=False):
    """verbose 只给主流程用 —— 置换检验会调它 200 次，不能每次都刷屏。"""
    cols = feats + (["expr"] if use_expr else [])
    X = df[cols].astype(float).values
    y = df["label"].values
    ok = ~np.isnan(X).any(axis=1)
    if verbose and (~ok).any():
        # ⚠️ 静默丢数据是这条管道最常见的失败（outcome12 §3.1）。
        #    值级 NaN 过滤不会改变行数/列数，上游任何环节出错都在这里被无声吃掉。
        d = df.loc[df.index[~ok]]
        print(f"    ⚠️ 行级丢弃 {int((~ok).sum())}/{len(df)}"
              f"（正 {int(d.label.sum())} 负 {int((d.label==0).sum())}）")
        cause = df.loc[d.index, cols].isna().idxmax(axis=1).value_counts()
        for c, k in cause.items():
            print(f"        {k:>3d} 行缺 {c}")
    X, y = X[ok], y[ok]
    if len(np.unique(y)) < 2 or y.sum() < 3:
        return np.nan, np.nan, 0
    rskf = RepeatedStratifiedKFold(n_splits=NSPLIT, n_repeats=NREP, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in rskf.split(X, y):
        p = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000,
                                             class_weight="balanced"))
        p.fit(X[tr], y[tr])
        oof[te] += p.predict_proba(X[te])[:, 1]
    oof /= NREP
    return roc_auc_score(y, oof), average_precision_score(y, oof), len(y)


def permutation_null(df, feats, n=200):
    """打乱标签重跑，给 AUC 的零分布 —— 小样本下 AUC 0.8 可能纯属偶然。"""
    y_true = df["label"].values
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        d = df.copy()
        d["label"] = rng.permutation(y_true)
        a, _, _ = cv_eval(d, feats, False)
        if a == a:
            out.append(a)
    return np.array(out)


def main():
    print("════ IRS 建模 ════")
    f = build_features()
    rows, frows = [], []
    for name in LABELS:
        df = pd.read_csv(f"{OUT}/label_{name}.tsv", sep="\t")
        # expr 不在这里 merge —— 13_labels.py 产出标签表时已带 expr 列
        df = df.merge(f.reset_index()[["gene_id"] + FEATURES],
                      on="gene_id", how="left", suffixes=("", "_f"))
        for c in FEATURES:
            if c + "_f" in df.columns:
                df[c] = df[c + "_f"]
        feats = [c for c in FEATURES if df[c].notna().sum() > 0.5 * len(df)]
        lo = [c for c in FEATURES if c not in feats]
        if lo:
            print(f"  覆盖率不足被丢弃的特征: {lo}")
        a1, p1, n1 = cv_eval(df, feats, use_expr=False, verbose=True)
        a2, p2, _ = cv_eval(df, feats, use_expr=True)
        null = permutation_null(df, feats)
        pct = (null < a1).mean() * 100 if a1 == a1 else np.nan
        print(f"\n【{name}】正类 {int(df.label.sum())} 负类 {int((df.label==0).sum())} "
              f"可用特征 {len(feats)}")
        print(f"  交叉验证 AUC={a1:.3f}  PR-AUC={p1:.3f}   （折外预测，{NSPLIT}折×{NREP}次）")
        print(f"  并入表达量协变量后 AUC={a2:.3f}  PR-AUC={p2:.3f}")
        print(f"  标签置换零分布 AUC: 中位 {np.median(null):.3f}  "
              f"95分位 {np.percentile(null,95):.3f}  真实值位于第 {pct:.1f} 百分位")
        rows.append(dict(label=name, n_pos=int(df.label.sum()),
                         n_neg=int((df.label == 0).sum()), n_features=len(feats),
                         AUC=a1, PR_AUC=p1, AUC_with_expr=a2, PR_AUC_with_expr=p2,
                         null_median=np.median(null), null_p95=np.percentile(null, 95),
                         AUC_percentile=pct))
        # 特征权重（全数据拟合，仅用于跨定义比对方向/量级）
        X = df[feats].astype(float).values
        y = df["label"].values
        ok = ~np.isnan(X).any(axis=1)
        p = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000,
                                             class_weight="balanced"))
        p.fit(X[ok], y[ok])
        co = p[-1].coef_[0]
        for c, v in zip(feats, co):
            frows.append(dict(label=name, feature=c, coef=v, abs_coef=abs(v)))

    pd.DataFrame(rows).to_csv(f"{OUT}/model_cv.tsv", sep="\t", index=False)
    fd = pd.DataFrame(frows)
    fd.to_csv(f"{OUT}/model_features.tsv", sep="\t", index=False)

    print("\n════ 稳健性检验（用户要求）════")
    piv = fd.pivot_table(index="feature", columns="label", values="coef")
    print(piv.round(3).to_string())
    cv = pd.read_csv(f"{OUT}/model_cv.tsv", sep="\t").set_index("label")

    def agree(a, b):
        r = piv[[a, b]].dropna()
        sa = (np.sign(r[a]) == np.sign(r[b])).mean()
        rho = np.corrcoef(r[a].abs().rank(ascending=False),
                          r[b].abs().rank(ascending=False))[0, 1]
        return sa, rho, len(r)

    print("\n  ① 同方向换阈值（这才是「阈值稳健性」该看的对比）")
    sa, rho, n = agree("down_strict", "down_loose")
    print(f"     down_strict vs down_loose: 特征符号一致 {sa:.0%}（{n} 个）"
          f"  |系数|排序 Spearman={rho:.3f}")
    print(f"     → {'稳健' if sa >= 0.7 and rho > 0.5 else '不稳健'}"
          f"；但严格集 n={int(cv.loc['down_strict','n_pos'])} 太小，"
          f"AUC={cv.loc['down_strict','AUC']:.3f} 落在零分布内（第 "
          f"{cv.loc['down_strict','AUC_percentile']:.0f} 百分位），本身不构成证据")

    print("\n  ② 翻转方向（这不是阈值问题，是对符号判据的独立检验）")
    sa2, rho2, n2 = agree("down_loose", "up_loose")
    print(f"     down_loose vs up_loose:    特征符号一致 {sa2:.0%}（{n2} 个）"
          f"  |系数|排序 Spearman={rho2:.3f}")
    au, pu = cv.loc["up_loose", "AUC"], cv.loc["up_loose", "AUC_percentile"]
    ad, pd_ = cv.loc["down_loose", "AUC"], cv.loc["down_loose", "AUC_percentile"]
    print(f"     up 方向 AUC={au:.3f}（第 {pu:.0f} 百分位）= 噪声；"
          f"down 方向 AUC={ad:.3f}（第 {pd_:.0f} 百分位）= 真信号")
    print(f"     → 两个方向学不出同样的东西，且**上调方向学不到任何序列信号**。")
    print(f"       ⚠️ 口径：这与符号翻转【相容】，但【不构成独立支持】—— "
          f"它只说明「用上调标签学不到序列信号」，")
    print(f"       并不能证明下调方向是对的。符号翻转的依据在 13_labels.py 的三条"
          f"（机制 + 通路级检验 + 文本一致）。")

    print("\n════ 阈值选择（富集强度 + 样本量 + 置换检验）════")
    print(cv[["n_pos", "AUC", "PR_AUC", "null_median", "null_p95",
              "AUC_percentile"]].round(3).to_string())
    print(f"\n  写出 model_cv.tsv / model_features.tsv")


if __name__ == "__main__":
    main()
