#!/usr/bin/env python3
"""
29_capacity_ladder.py —— 模型容量阶梯：把"加容量没用"从立场变成实验

【为什么要做】§1.6① 说「信号是碱基组成/结构，加容量只是过拟合」，
  但**全部现有证据都来自 `PCA + 逻辑回归`** ⇒ 目前只排除了【表征】，没排除【假设空间】。
  批评者一句话就能戳：「你只用线性模型，凭什么断言容量无益？」

【设计】**表征与维数固定，只变假设空间。**
  表征       人工 9 维 / 4-mer 256→PCA(9) / 4-mer 256 原维
  模型类     LR（基线）/ RF / GBM / MLP(16) / MLP(64)
  口径       同一簇级 GroupKFold、同一簇级自助，全域与适用域各一套
  超参       **全部写死，不做任何搜索**

【⚠️ 泄漏纪律】StandardScaler 与 PCA 一律**放在 pipeline 里、按折拟合**
  —— 与 `19_ceiling.eval_rep` 同规矩。
  （把 PCA 先在全集上 fit 好再切 CV，属无监督泄漏，会让 CI 虚假收窄；
   本脚本首版就是那么写的，已改。改回外部 fit 即为回退。）

【判据】见 判据预注册_outcome15_prereg.md §29。要点：
  A：所有非线性模型 ΔAUC<0.03 或 CI 重叠 → 「容量无益」成立
  B：任一模型 ΔAUC≥0.03 且 CI 不重叠 → **§1.6① 必须改写，答辩口径同步修订**
  + 负对照：打乱标签后各模型零分布应 ≈0.5，明显偏高说明该模型 CV 内泄漏，其真值作废

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 29_capacity_ladder.py
"""
import os
# ⚠️ 必须在 import numpy 之前设 —— 理由见 16_validation.py 头部
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
PIPE = os.environ.get("IETX_SCRIPTS", f"{IETX}/scripts")
NBOOT = 200          # 簇级自助轮数；定 CI 形状够用
NNULL = 200
SEED = 0
PCA_DIM = 9          # 与人工特征同为 9 维 —— 这是「只变假设空间」的关键约束


def _mod(name, path):
    """载入主流程脚本为模块。找不到时给一句人话，而不是抛原始 traceback。"""
    if not os.path.exists(path):
        raise SystemExit(f"!! 找不到主流程脚本 {path} —— 用 IETX_SCRIPTS 指定它的目录"
                         f"（默认 $IETX_ROOT/scripts）")
    from importlib.util import spec_from_file_location, module_from_spec
    sp = spec_from_file_location(name, path)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def read_promoters(path, want):
    """与 18_embed.read_fa **同构**：头形如 >h1SMcG0000001::chr1_h1:37304-38304(+)。
    ⚠️ promoters.fa 是给 RNAfold 的，序列里是【U 不是 T】——
       4-mer 组成按 DNA 处理，不换回 T 会整段失真（18_embed.py 已记录此坑）。"""
    seq, name, buf = {}, None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name in want:
                    seq[name] = "".join(buf).upper().replace("U", "T")
                name, buf = ln[1:].split("::")[0].strip(), []
            else:
                buf.append(ln.strip())
    if name in want:
        seq[name] = "".join(buf).upper().replace("U", "T")
    return seq


def kmer_mat(seqs, k):
    """k-mer 组成（行归一）。与 19_ceiling.kmer_mat 同构。"""
    idx = {"".join(t): i for i, t in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), len(idx)))
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                M[r, j] += 1
    tot = M.sum(1, keepdims=True)
    return M / np.where(tot == 0, 1, tot)


# ── 模型类：**超参全部写死**，不搜索 ────────────────────────────────────────
def fit_model(name, Xtr, ytr, pca=False):
    """返回一个已 fit、带 predict_proba 的 **pipeline**。

    ⚠️ StandardScaler / PCA 都在 pipeline 内 —— 按折拟合，不跨折。
       把 PCA 挪到外面先在全集上 fit 就是泄漏，不许改回去。"""
    pre = [StandardScaler()]
    if pca:
        pre.append(PCA(n_components=min(PCA_DIM, Xtr.shape[1], Xtr.shape[0] - 1),
                       random_state=SEED))
    if name == "LR":
        est = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    elif name == "RF":
        est = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=1)
    elif name == "GBM":
        est = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=2, random_state=SEED)
    elif name.startswith("MLP"):
        est = MLPClassifier(hidden_layer_sizes=(int(name[3:]),), alpha=1e-3,
                            max_iter=3000, random_state=SEED)
    else:
        raise KeyError(name)
    pipe = make_pipeline(*pre, est)
    if name == "GBM":
        # GBM 无 class_weight，用平衡样本权重代替（与 LR/RF 的 class_weight 同义）
        w = np.where(ytr == 1, len(ytr) / (2 * max(1, int((ytr == 1).sum()))),
                     len(ytr) / (2 * max(1, int((ytr == 0).sum()))))
        return pipe.fit(Xtr, ytr, gradientboostingclassifier__sample_weight=w)
    return pipe.fit(Xtr, ytr)


def oof_auc(X, y, g, name, pca=False):
    fits = GroupKFold(n_splits=min(5, len(np.unique(g))))
    oof, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in fits.split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        p = fit_model(name, X[tr], y[tr], pca=pca)
        oof[te] += p.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    msk = cnt > 0
    if len(np.unique(y[msk])) < 2:
        return np.nan
    return roc_auc_score(y[msk], oof[msk] / cnt[msk])


def boot_ci(X, y, g, name, pca=False, n=NBOOT, seed=SEED):
    """簇级自助 —— 与 16_validation.boot_cluster 同口径，但模型可换。"""
    uc = np.unique(g)
    bucket = {c: np.flatnonzero(g == c) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            a = oof_auc(X[idx], y[idx], g[idx], name, pca=pca)
        except Exception:
            continue
        if a == a:
            out.append(a)
    return (np.percentile(out, 2.5), np.percentile(out, 97.5), len(out)) if out \
        else (np.nan, np.nan, 0)


# (表征名, 需要 PCA 降到 9 维?, 可用模型)
REPS = [("人工9维", False, ["LR", "RF", "GBM", "MLP16", "MLP64"]),
        ("4mer-PCA9", True, ["LR", "RF", "GBM", "MLP16", "MLP64"]),
        ("4mer-原维", False, ["RF", "GBM"])]


def main():
    v16 = _mod("v16", os.path.join(PIPE, "16_validation.py"))

    print("═" * 78)
    print("29 · 模型容量阶梯（表征与维数固定，只变假设空间）")
    print("═" * 78)
    print("  ⚠️ Scaler/PCA 均在 pipeline 内按折拟合（无监督泄漏已排除）")

    feats = v16.build_features()
    df = v16.load_label("down_loose", feats)
    X9_all, y_all, fcols, idx = v16.usable(df, verbose=True)
    genes_all = df.loc[idx, "gene_id"].values

    dom = set(pd.read_csv(f"{OUT}/domain_genes.tsv")["gene_id"])
    m = np.array([g in dom for g in genes_all])
    if int(m.sum()) != 114:
        raise SystemExit(f"!! 适用域 n={int(m.sum())} ≠ 114 —— 掩码口径变了，停手")

    clust, _ = v16.clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    g_all = clust.reindex(genes_all).fillna(
        pd.Series(np.arange(len(genes_all)) + 10 ** 6, index=genes_all)).astype(int).values

    prom = read_promoters(f"{FEAT}/promoters.fa", set(genes_all))
    miss = [g for g in genes_all if g not in prom]
    if miss:
        raise SystemExit(f"!! {len(miss)} 个入模基因缺启动子序列（如 {miss[:3]}）—— 停手")
    K4_all = kmer_mat([prom[g] for g in genes_all], 4)
    MATS = {"人工9维": X9_all, "4mer-PCA9": K4_all, "4mer-原维": K4_all}
    print(f"  4-mer 矩阵 {K4_all.shape}；启动子 {len(prom)} 条")

    rows = []
    for scope in ("全域", "适用域"):
        sel = np.ones(len(y_all), bool) if scope == "全域" else m
        y, g = y_all[sel], g_all[sel]
        print(f"\n{'═' * 78}\n[{scope}] n={len(y)}（正 {int(y.sum())}）"
              f" 独立单元 {len(np.unique(g))}\n{'═' * 78}")
        print(f"  {'表征':12s} {'模型':7s} {'维':>4s} {'AUC':>7s} {'簇级CI':>17s} {'块数':>5s}")
        for rep, use_pca, models in REPS:
            Xr = MATS[rep][sel]
            for name in models:
                a = oof_auc(Xr, y, g, name, pca=use_pca)
                lo, hi, nb = boot_ci(Xr, y, g, name, pca=use_pca)
                ncol = min(PCA_DIM, Xr.shape[1]) if use_pca else Xr.shape[1]
                print(f"  {rep:12s} {name:7s} {ncol:>4d} {a:>7.3f} "
                      f"[{lo:.3f},{hi:.3f}] {nb:>5d}")
                rows.append(dict(scope=scope, rep=rep, model=name, n_dim=ncol,
                                 AUC=a, CI_lo=lo, CI_hi=hi, n_boot=nb))

    # ── 负对照 ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 78}\n负对照：打乱标签后各模型的零分布（应 ≈0.5）\n{'═' * 78}")
    rng = np.random.default_rng(SEED)
    for name in ["LR", "RF", "GBM", "MLP16", "MLP64"]:
        null = []
        for _ in range(NNULL):
            yp = rng.permutation(y_all)
            a = oof_auc(X9_all, yp, g_all, name)
            if a == a:
                null.append(a)
        med, p95 = float(np.median(null)), float(np.percentile(null, 95))
        flag = "  ← ⚠️ 零分布偏高，该模型 CV 内可能泄漏" if med > 0.60 else ""
        print(f"  {name:7s} 零分布 中位 {med:.3f}  p95 {p95:.3f}  n={len(null)}{flag}")
        rows.append(dict(scope="负对照", rep="人工9维", model=name, n_dim=9,
                         AUC=med, CI_lo=np.nan, CI_hi=p95, n_boot=len(null)))

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/capacity_ladder.tsv", sep="\t", index=False)

    # ── 判据应用 ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 78}\n判据应用（提升 ≥0.03 且 CI 不重叠 → 容量确有增益）\n{'═' * 78}")
    any_open = False
    for scope in ("全域", "适用域"):
        for rep, _, _ in REPS:
            s = r[(r.scope == scope) & (r.rep == rep)]
            base = None
            if (s.model == "LR").any():
                base = s[s.model == "LR"].iloc[0]
            for _, q in s.iterrows():
                if q.model == "LR":
                    print(f"  [{scope}/{rep}] {q.model:7s} AUC={q.AUC:.3f} （基线）")
                    continue
                if base is None:
                    print(f"  [{scope}/{rep}] {q.model:7s} AUC={q.AUC:.3f}"
                          f"（该表征无 LR 基线，不做判定）")
                    continue
                d = q.AUC - base.AUC
                overlap = not (q.CI_hi < base.CI_lo or base.CI_hi < q.CI_lo)
                hit = (d >= 0.03) and (not overlap)
                any_open |= hit
                print(f"  [{scope}/{rep}] {q.model:7s} AUC={q.AUC:.3f} Δ={d:+.3f} "
                      f"CI{'重叠' if overlap else '不重叠'}"
                      + ("   ← ⚠️ 容量确有增益" if hit else ""))
    print(f"\n  ⇒ 判据 **{'B 容量有增益 —— §1.6① 与答辩口径必须改写' if any_open else 'A 容量无增益 —— 原结论成立'}**")
    print(f"  ⇒ 写出 {OUT}/capacity_ladder.tsv（{len(r)} 行）")


if __name__ == "__main__":
    main()
