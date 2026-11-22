#!/usr/bin/env python3
"""
19_ceiling.py —— 有界收尾实验：把「换表征/加特征」的天花板钉死

⚠️ 判据已在 logs/outcome14_prereg.md 中【运行前写定】，本脚本只做实现，不改判据。

候选（预注册第二节，不增不减）：
  A  k-mer 组成 k=3 / 5 / 6  —— 组分表征是否已饱和
  B  4-mer 对 log10(cds_len) 正交化 —— 直接检验 §7.4：信号是否来自长度

口径：适用域 n=114，簇级 GroupKFold，StandardScaler→PCA(9)→LR，簇级自助 1000。
判据：提升 ≥0.03 且 CI 不重叠 → 未到天花板；否则 → 已到天花板。

用法: HF_ENDPOINT=... $IETX_ROOT/env/bin/python 19_ceiling.py
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
BASE_D = 9          # 预注册固定，不搜参


def _mod(name, path):
    from importlib.util import spec_from_file_location, module_from_spec
    sp = spec_from_file_location(name, path)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def kmer_mat(seqs, k):
    idx = {"".join(t): i for i, t in enumerate(product("ACGT", repeat=k))}
    M = np.zeros((len(seqs), len(idx)))
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                M[r, j] += 1
    tot = M.sum(1, keepdims=True)
    return M / np.where(tot == 0, 1, tot)


def orthogon_alize(M, z):
    """逐维对 z（含截距）回归取残差 —— 即把 z 能线性解释的部分从每个维度里剥掉。"""
    Z = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(Z, M, rcond=None)
    return M - Z @ beta


def eval_rep(X, y, g, tag, d=BASE_D, nboot=1000, seed=0):
    # random_state 必须给：max(X.shape)>500 且 n_components<0.8*min(X.shape) 时，
    # sklearn 的 svd_solver='auto' 选【randomized SVD】，不给种子结果逐次不同。
    # 实测：k=5(1024 维)/k=6(4096 维) 命中，两次跑 AUC 差 0.001–0.003；
    #       k=3(64 维)/4-mer(256 维) 不命中（走 full SVD），两次完全一致。
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=min(d, X.shape[1], X.shape[0] - 1), random_state=0),
                         LogisticRegression(C=1.0, max_iter=5000,
                                            class_weight="balanced"))

    def oof(Xa, ya, ga):
        oof_, cnt = np.zeros(len(ya)), np.zeros(len(ya))
        for tr, te in GroupKFold(min(5, len(np.unique(ga)))).split(Xa, ya, ga):
            if len(np.unique(ya[tr])) < 2:
                continue
            oof_[te] += pipe.fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
            cnt[te] += 1
        m = cnt > 0
        return roc_auc_score(ya[m], oof_[m] / cnt[m]), average_precision_score(
            ya[m], oof_[m] / cnt[m])

    auc, pr = oof(X, y, g)
    rng = np.random.default_rng(seed)
    uc = np.unique(g)
    bucket = {c: np.flatnonzero(g == c) for c in uc}
    b = []
    for _ in range(nboot):
        i = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
        if len(np.unique(y[i])) < 2:
            continue
        try:
            b.append(oof(X[i], y[i], g[i])[0])
        except ValueError:
            pass
    lo, hi = np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)
    print(f"  {tag:28s} AUC={auc:.3f}  PR-AUC={pr:.3f}  簇级CI [{lo:.3f}, {hi:.3f}]")
    return dict(rep=tag, AUC=auc, PR_AUC=pr, CI_lo=lo, CI_hi=hi)


def main():
    v16 = _mod("v16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "16_validation.py"))
    v18 = _mod("v18", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "18_embed.py"))

    print("═" * 78)
    print("有界收尾实验（判据见 logs/outcome14_prereg.md，运行前写定）")
    print("═" * 78)

    feats = v16.build_features()
    clust, _ = v16.clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    df = v16.load_label("down_loose", feats)
    Xh, y, fcols, idx = v16.usable(df, verbose=False)
    genes = df.loc[idx, "gene_id"].values
    # 补充 scope 用：掩码之前的全域副本
    Xh_all, y_all, genes_all = Xh.copy(), y.copy(), genes.copy()

    dom = set(pd.read_csv(f"{OUT}/domain_genes.tsv")["gene_id"])
    m = np.array([gname in dom for gname in genes])
    y = y[m]
    gm = genes[m]
    # ⚠️ fillna 的 Series 必须带 index —— 否则按位置对齐失败，整列变 NaN
    g = clust.reindex(gm).fillna(
        pd.Series(np.arange(len(gm)) + 10 ** 6, index=gm)).astype(int).values
    Xh = Xh[m]
    genes = genes[m]
    print(f"适用域 n={len(y)}（正 {int(y.sum())} 负 {int((y == 0).sum())}）"
          f" 独立单元 {len(np.unique(g))}")

    prom = v18.read_fa(f"{FEAT}/promoters.fa", set(genes))
    seqs = [prom[gname] for gname in genes]
    L = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")["cds_len_longest"]
    z = np.log10(L.reindex(genes).values.astype(float) + 1)
    print(f"  启动子序列 {len(seqs)}/{len(genes)}\n")

    rows = []
    print("── 基线（同批重算，不引用历史数字）──")
    rows.append(eval_rep(Xh, y, g, "基线 人工特征(9维)"))
    print()

    print("── 候选 A：k-mer 组成 k=3/5/6 ──")
    for k in (3, 5, 6):
        K = kmer_mat(seqs, k)
        print(f"   [k={k}, {K.shape[1]} 维]")
        rows.append(eval_rep(K, y, g, f"A k-mer k={k}"))
    print()

    print("── 候选 B：4-mer 对 log10(cds_len) 正交化 ──")
    K4 = kmer_mat(seqs, 4)
    K4o = orthogon_alize(K4, z)
    rows.append(eval_rep(K4, y, g, "B0 4-mer 未正交化"))
    rows.append(eval_rep(K4o, y, g, "B  4-mer 正交化后"))
    print()

    print("── 参考：长度单变量与人工特征+长度（不参与判定）──")
    rows.append(eval_rep(z.reshape(-1, 1), y, g, "ref log10(cds_len) 单变量"))
    rows.append(eval_rep(np.column_stack([z, Xh]), y, g,
                         "ref 人工特征+长度(10维)", d=10))

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/ceiling.tsv", sep="\t", index=False)
    base = r[r.rep.str.startswith("基线")].iloc[0]      # ⚠️ 是 DataFrame 的行，不是 dict

    print("\n" + "═" * 78)
    print("判据应用（提升 ≥0.03 且 CI 不重叠 → 未到天花板）")
    print("═" * 78)
    print(f"  基线 AUC={base['AUC']:.3f}  簇级 CI [{base['CI_lo']:.3f}, {base['CI_hi']:.3f}]")
    any_open = False
    for _, q in r.iterrows():
        if q.rep.startswith("基线") or q.rep.startswith("ref"):
            continue
        gain = q.AUC - base.AUC
        overlap = not (q.CI_lo > base.CI_hi or q.CI_hi < base.CI_lo)
        verdict = "未到天花板" if (gain >= 0.03 and not overlap) else "已到天花板"
        if verdict == "未到天花板":
            any_open = True
        print(f"  {q.rep:28s} 提升 {gain:+.3f}  CI重叠 {'是' if overlap else '否'}"
              f"  → {verdict}")
        if gain >= 0.03 and overlap:
            print(f"       ⚠️ 点估计提升够（{gain:+.3f}）但 CI 重叠 → 按预注册判据仍算【已到天花板】")
    print(f"\n  ⇒ 整体判定：【{'未到天花板' if any_open else '已到天花板'}】")
    if any_open:
        print(f"     ⚠️ 按预注册第五节第 3 条：不得据此调参追 AUC，"
              f"只如实报告「此处尚有余量，未探」。")
        print(f"     ⚠️ 且按第四节，§5.3 的收敛结论【不成立】，需重写。")
    b = r[r.rep.str.startswith("B ")].iloc[0]
    print(f"\n  候选 B 判读（§7.4 的关键分支）:")
    print(f"    4-mer 未正交化 {r[r.rep.str.startswith('B0')].iloc[0].AUC:.3f}"
          f" → 正交化后 {b.AUC:.3f}（掉 {r[r.rep.str.startswith('B0')].iloc[0].AUC - b.AUC:.3f}）")
    b0 = r[r.rep.str.startswith("B0")].iloc[0].AUC
    if b.AUC <= 0.55:
        print(f"    ⇒ 掉到 ≈0.5：信号主体是 CDS 长度，§7.4 需升级，主结果定性要再改")
    elif b.AUC < b0 - 0.03:
        print(f"    ⇒ 明显下降但未到 0.5：长度是重要来源之一，【不是全部】")
    else:
        print(f"    ⇒ 【未下降甚至上升】：剥掉长度的线性成分后信号仍在 → ")
        print(f"       组成信号【不是】长度的替身。这与预注册第六节的预期相反，如实报告。")
    # ── 补充（不在预注册判据内）：候选 B 在全域 141 上也跑一遍 ──
    #    若全域正交化也在 0.8 上下，适用域就退成纯外推边界声明，主结果不需要它。
    print("\n" + "═" * 78)
    print("补充：候选 B 在【全域 n=141】上重跑（不在预注册判据内）")
    print("═" * 78)
    g_all = clust.reindex(genes_all).fillna(
        pd.Series(np.arange(len(genes_all)) + 10 ** 6, index=genes_all)).astype(int).values
    prom_all = v18.read_fa(f"{FEAT}/promoters.fa", set(genes_all))
    seqs_all = [prom_all[gname] for gname in genes_all]
    z_all = np.log10(L.reindex(genes_all).values.astype(float) + 1)
    K4_all = kmer_mat(seqs_all, 4)
    supp = [eval_rep(Xh_all, y_all, g_all, "全域 基线 人工特征(9维)"),
            eval_rep(K4_all, y_all, g_all, "全域 4-mer 未正交化"),
            eval_rep(orthogon_alize(K4_all, z_all), y_all, g_all, "全域 4-mer 正交化后")]
    pd.DataFrame(supp).to_csv(f"{OUT}/ceiling_global.tsv", sep="\t", index=False)

    print(f"\n  写出 ceiling.tsv / ceiling_global.tsv")


if __name__ == "__main__":
    main()
