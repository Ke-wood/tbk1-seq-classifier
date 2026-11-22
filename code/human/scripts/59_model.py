#!/usr/bin/env python3
"""59_model.py —— T-02 第 5 步：九个特征 vs 一个长度

评估协议逐条照抄涡虫侧 16_validation.py（直接 import 它的函数，不另写一份）：
  L2 逻辑回归 C=1.0 / class_weight=balanced / max_iter=3000，StandardScaler 在 pipeline 内
  同源簇 GroupKFold，5 折单次；簇级自助 300 次；置换 1000 次（匹配集内打乱）；SEED=0

必须产出的四个数（§3）：
  1 9 特征 AUC（全域）
  2 长度单变量 AUC（全域）
  3 长度单变量 AUC（共同区间）
  4 9 特征 − 长度（全域差值）  ← 判据就靠这一行

用法: env/bin/python 59_model.py GSE309167 [pooled]
"""
import os, sys, importlib.util
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("IETX_ROOT", R)      # 16_validation.py 模块级就要这个变量
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("V16", os.path.join(S, "16_validation.py"))
V16 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V16)

FEATURES = V16.FEATURES
COV_MIN = V16.COV_MIN
SEED = 0


def usable(df):
    """与 16_validation.usable 同规则；单独写是因为要同时拿到行掩码做丢弃统计。"""
    feats = [c for c in FEATURES if df[c].notna().sum() > COV_MIN * len(df)]
    X = df[feats].astype(float).values
    y = df["label"].values
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], y[ok], feats, ok


def main(gse):
    tag = sys.argv[2] if len(sys.argv) > 2 else "pooled"
    D = f"{R}/人类侧_{gse}"
    out = f"{D}/06_model"; os.makedirs(out, exist_ok=True)

    feats = pd.read_csv(f"{R}/03_features/features.tsv", sep="\t").set_index("gene_id")

    # ⚠️ 先验一次「转录本 ID 能不能映射到基因」。映射不上时 read_avs 会把 qg/sg
    #    全变成 NaN、dropna 后一条边都不剩 —— 得到「每个基因都没有同源」的假结论，
    #    而下游 GroupKFold 照常跑、AUC 照常出，一声不响。
    av = f"{R}/homology/allvall_filt.tsv"
    t2g = pd.read_csv(f"{R}/results/tx2gene.tsv", sep="\t",
                      index_col="transcript_id")["gene_id"]
    probe = pd.read_csv(av, sep="\t", header=None, usecols=[0, 1],
                        names=["q", "s"], nrows=200000)
    miss = float((~probe.q.isin(t2g.index)).mean())
    print(f"DIAMOND 命中前 {len(probe)} 行中，转录本无法映射到基因的比例 {miss:.4f}")
    assert miss < 0.01, "转录本 ID 与 tx2gene 对不上（版本号？），同源簇会全空"

    cache = f"{R}/homology/clusters.pkl"
    if os.path.exists(cache):
        clust = pd.read_pickle(cache)
        print(f"同源簇从缓存读入（{cache}）")
    else:
        clust, _ = V16.clusters_from_diamond(av)
        clust.to_pickle(cache)
    nclust = int(clust.nunique())
    sizes = clust.value_counts()
    print(f"同源簇：基因 {len(clust)}  簇 {nclust}  最大簇 {sizes.iloc[0]}  "
          f"≥2 基因的簇 {int((sizes >= 2).sum())}")

    summary, lenc, abl, shift, drops = [], [], [], [], []
    for kind in ("both", "down", "up"):
        df = pd.read_csv(f"{D}/05_label/label_{kind}_{tag}.tsv", sep="\t")
        # ⚠️ cds_len_longest 不在 9 个模型特征里，但长度单变量要用它 ——
        #    只 merge FEATURES 会在下面 df.loc[idx,"cds_len_longest"] 处 KeyError。
        df = df.merge(feats.reset_index()[["gene_id"] + FEATURES + ["cds_len_longest"]],
                      on="gene_id", how="left", suffixes=("", "_f"))
        for c in FEATURES:
            if c + "_f" in df.columns:
                df[c] = df[c + "_f"]
        X, y, fcols, ok = usable(df)
        idx = df.index[ok]
        print(f"\n【{kind}】入模 {len(y)}（正 {int(y.sum())} 负 {int((y == 0).sum())}）"
              f"  特征 {fcols}")
        if len(fcols) < len(FEATURES):
            print(f"  ⚠️ 覆盖率 <{COV_MIN:.0%} 被丢的特征: "
                  f"{[c for c in FEATURES if c not in fcols]}")

        # ── 丢弃统计（§3 附加产出）────────────────────────────
        dr = df.loc[~ok]
        cause = df.loc[~ok, FEATURES].isna().idxmax(axis=1).value_counts().to_dict()
        drops.append(dict(kind=kind, n_total=len(df), n_used=len(y),
                          dropped_pos=int(dr.label.sum()),
                          dropped_neg=int((dr.label == 0).sum()),
                          cause=";".join(f"{k}:{v}" for k, v in cause.items())))
        if len(dr):
            print(f"  行级丢弃 {len(dr)}（正 {int(dr.label.sum())} / 负 "
                  f"{int((dr.label == 0).sum())}）  原因 {cause}")

        genes = df.loc[idx, "gene_id"].values
        gs = clust.reindex(genes)
        g = gs.fillna(pd.Series(np.arange(len(genes)) + 10 ** 6,
                                index=genes)).astype(int).values   # 单例各自成组

        a9, ap9, ng = V16.cv_auc(X, y, g)
        ci = V16.boot_cluster(X, y, g, kind="group", n=300)
        lo, hi = np.percentile(ci, [2.5, 97.5])
        null = V16.perm_null_group(X, y, g, n=1000)
        p95 = float(np.percentile(null, 95))
        pemp = max(float((null >= a9).mean()), 1 / len(null))

        # ── 长度 ────────────────────────────────────────────
        e = np.log10(df.loc[idx, "cds_len_longest"].values.astype(float) + 1)
        ok2 = np.isfinite(e)
        a_len = V16.cv_auc(e[ok2].reshape(-1, 1), y[ok2], g[ok2])[0]
        pp, qq = e[ok2 & (y == 1)], e[ok2 & (y == 0)]
        lo_, hi_ = max(pp.min(), qq.min()), min(pp.max(), qq.max())
        sel = ok2 & (e >= lo_) & (e <= hi_)
        a9_sub = V16.cv_auc(X[sel], y[sel], g[sel])[0]
        alen_sub = V16.cv_auc(e[sel].reshape(-1, 1), y[sel], g[sel])[0]
        # ⚠️ 第 4 行要的是「同一批行上 9 特征 − 长度」。首版这里把长度又算了一遍
        #    赋给了 a_len_full_same，于是 delta_9_minus_len 恒等于 0.0 —— 而 0.0
        #    看着完全像个正常结果（还正好落在 A 档），不查根本发现不了。
        a9_lenrows = V16.cv_auc(X[ok2], y[ok2], g[ok2])[0]

        summary.append(dict(
            label=kind, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
            groups=len(np.unique(g)),
            auc9=round(a9, 4), auc9_ci_lo=round(lo, 4), auc9_ci_hi=round(hi, 4),
            ap9=round(ap9, 4),
            auc9_vs_perm_p=round(pemp, 4), perm_p95=round(p95, 4),
            auc_len=round(a_len, 4),
            # ⚠️ 第 4 行必须在【同一批行】上相减 —— 长度有值的行
            auc9_on_len_rows=round(a9_lenrows, 4),
            delta_9_minus_len=round(a9_lenrows - a_len, 4),
            auc9_common=round(a9_sub, 4), auc_len_common=round(alen_sub, 4),
            n_common=int(sel.sum()), common_lo=round(10 ** lo_), common_hi=round(10 ** hi_),
            delta_common=round(a9_sub - alen_sub, 4)))
        print(f"  9 特征 {a9:.4f} [{lo:.3f}, {hi:.3f}]  AP {ap9:.4f}  "
              f"置换 p95 {p95:.3f} p={pemp:.4f}")
        print(f"  长度全域 {a_len:.4f}  共同区间 [{10**lo_:.0f}, {10**hi_:.0f}]bp "
              f"n={int(sel.sum())}: 9特征 {a9_sub:.4f} / 长度 {alen_sub:.4f}")
        print(f"  ★ 第 4 行 9特征 − 长度 = {a9_lenrows - a_len:+.4f}")

        # ── 长度偏移 ────────────────────────────────────────
        cp, cn = 10 ** pp, 10 ** qq
        shift.append(dict(kind=kind, pos_median=float(np.median(cp)),
                          neg_median=float(np.median(cn)),
                          ratio=float(np.median(cp) / np.median(cn)),
                          log2_ratio=float(np.log2(np.median(cp) / np.median(cn)))))

        # ── 消融 ────────────────────────────────────────────
        for c in fcols:
            Xa = X[:, [fcols.index(x) for x in fcols if x != c]]
            ag = V16.cv_auc(Xa, y, g)[0]
            abl.append(dict(kind=kind, dropped=c, auc=round(ag, 4),
                            dAUC=round(ag - a9, 4)))

        pd.DataFrame({"perm_index": np.arange(len(null)),
                      "auc": np.round(null, 6)}).to_csv(
            f"{out}/perm_null_1000_{kind}_{tag}.tsv", sep="\t", index=False)

    # ⚠️ 必须带 tag 后缀。首版四个汇总表都用固定文件名，于是每跑一个对照就覆盖一次 ——
    #    最终留在 model_cv.tsv 里的是【最后一个对照】的结果，而它看起来完全正常
    #    （池化 284 个正类 → 文件里却是 306 个），只有拿标注数去对才能发现。
    sfx = "" if tag == "pooled" else f"_{tag}"
    pd.DataFrame(summary).to_csv(f"{out}/model_cv{sfx}.tsv", sep="\t", index=False)
    pd.DataFrame(abl).to_csv(f"{out}/ablation{sfx}.tsv", sep="\t", index=False)
    pd.DataFrame(shift).to_csv(f"{out}/length_shift{sfx}.tsv", sep="\t", index=False)
    pd.DataFrame(drops).to_csv(f"{out}/drop_stats{sfx}.tsv", sep="\t", index=False)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main(sys.argv[1])
