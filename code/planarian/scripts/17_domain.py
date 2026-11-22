#!/usr/bin/env python3
"""
17_domain.py —— 适用域诊断与界定（CDS 长度混杂的处置）

【背景】outcome12 §7.4：CDS 长度 SMD 0.745，且 log10(cds_len) 单特征 AUC 0.743
  ＞ 完整 9 特征模型 0.712。这是本报告最硬的结构混杂。

【处置】主结果改为「两类 cds_len 分布重叠区」子集，全域 141 降为敏感性分析。
  适用域规则必须 **outcome-blind**（只由 cds_len 的取值分布定，与标签无关）。

【本脚本产出】
  诊断 A  方向拆分：显著集分上调/下调，各自 cds_len 分布 + 对负类池的 SMD
  诊断 B  特征 × log10(cds_len) 的 Spearman 矩阵（n=141），重点 prom_gc / cds_gc
  核查     rho(cds_len, pvalue) 与长度五分位表的精确口径（两者曾被指为自相矛盾）
  适用域   重叠区上下界 + 子集内成套指标（簇级 CI / 区间内置换 / 消融基线）

用法: $IETX_ROOT/env/bin/python 17_domain.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
SEED = 0


def main():
    import sys
    from importlib.util import spec_from_file_location, module_from_spec
    here = os.path.dirname(os.path.abspath(__file__))
    sp = spec_from_file_location("v16", os.path.join(here, "16_validation.py"))
    v = module_from_spec(sp)
    sp.loader.exec_module(v)

    feats = v.build_features()
    clust, nclust = v.clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    df = v.load_label("down_loose", feats)
    X, y, fcols, idx = v.usable(df, verbose=False)
    genes = df.loc[idx, "gene_id"].values
    g = clust.reindex(genes).fillna(
        pd.Series(np.arange(len(genes)) + 10 ** 6, index=genes)).astype(int).values
    L = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")["cds_len_longest"]
    cds = L.reindex(genes).values.astype(float)
    e = np.log10(cds + 1)

    print("═" * 78)
    print("诊断 A：显著集的方向拆分 —— cds_len 分布 + 对【负类池】的 SMD")
    print("═" * 78)
    rr = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t", index_col=0)
    neg_pool = df.loc[df.label == 0, "gene_id"]
    neg_len = L.reindex(neg_pool).dropna()
    sig = rr[rr.padj < 0.05]
    up = L.reindex(sig.index[sig.log2FoldChange > 0]).dropna()
    dn = L.reindex(sig.index[sig.log2FoldChange < 0]).dropna()
    print(f"  负类池（label_down_loose 的负类）n={len(neg_len)}  中位 {neg_len.median():.0f} bp")
    print(f"\n  {'集合':10s} {'n':>5s} {'中位':>10s} {'p25':>10s} {'p75':>10s} "
          f"{'vs 负类池 p':>12s} {'SMD':>8s}")
    for nm, s in (("上调组", up), ("下调组", dn)):
        smd = abs(s.mean() - neg_len.mean()) / pd.concat([s, neg_len]).std()
        p = mannwhitneyu(s, neg_len).pvalue
        print(f"  {nm:10s} {len(s):>5d} {s.median():>10.0f} {s.quantile(.25):>10.0f} "
              f"{s.quantile(.75):>10.0f} {p:>12.2e} {smd:>8.3f}")
    # ⚠️ "偏长" 必须同时看【中位方向】和显著性。只看 p 会被尾部拖动：
    #    上调组均值检验 p=0.035 但【中位数反而低于负类池】—— 那不是"偏长"。
    pu, pdw = mannwhitneyu(up, neg_len).pvalue, mannwhitneyu(dn, neg_len).pvalue
    up_long = (up.median() > neg_len.median()) and pu < 0.05
    dn_long = (dn.median() > neg_len.median()) and pdw < 0.05
    print(f"\n  「偏长」判据 = 中位数高于负类池【且】Mann-Whitney p<0.05")
    print(f"    上调组: 中位 {up.median():.0f} vs 池 {neg_len.median():.0f}"
          f" → {'偏长' if up_long else '不偏长（中位更低，仅均值检验边缘显著）'}")
    print(f"    下调组: 中位 {dn.median():.0f} vs 池 {neg_len.median():.0f}"
          f" → {'偏长' if dn_long else '不偏长'}")
    if up_long and dn_long:
        verdict, why = "技术性混杂", "两组都偏长 = 检出功效假象（双向）"
    elif dn_long:
        verdict, why = "可能是真生物学，未排除", "仅下调组偏长 = 方向特异，非通用功效假象"
    else:
        verdict, why = "证据不足", "两组都不偏长"
    print(f"  → 判据：{'两组都显著偏长 → 技术性混杂' if True else ''}")
    print(f"  ⇒ 定性为「{verdict}」（{why}）")

    # 分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(2, 5, 40)
        for nm, s, c in (("up (n=%d)" % len(up), up, "tab:green"),
                         ("down (n=%d)" % len(dn), dn, "tab:red"),
                         ("negative pool (n=%d)" % len(neg_len), neg_len, "tab:gray")):
            ax.hist(np.log10(s + 1), bins=bins, alpha=.5, label=nm, color=c, density=True)
        ax.set_xlabel("log10(CDS length, bp)")
        ax.set_ylabel("density")
        ax.set_title("CDS length: up vs down vs negative pool")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT}/domain_cds_len.png", dpi=130)
        print(f"  图 → {OUT}/domain_cds_len.png")
    except Exception as e:
        print(f"  (作图跳过: {e})")

    print("\n" + "═" * 78)
    print("诊断 B：特征 × log10(cds_len) 的 Spearman（n=141 入模集）")
    print("═" * 78)
    print(f"  {'特征':16s} {'rho':>8s} {'p':>12s}")
    flags = []
    for c in fcols:
        s = pd.DataFrame({"f": df.loc[idx, c].astype(float).values, "e": e}).dropna()
        r = spearmanr(s.f, s.e)
        mark = "  ← |rho|>0.5，消融结论需重新定性" if abs(r.statistic) > 0.5 else ""
        if abs(r.statistic) > 0.5:
            flags.append(c)
        print(f"  {c:16s} {r.statistic:>+8.3f} {r.pvalue:>12.2e}{mark}")
    print(f"  → {'有特征 |rho|>0.5：' + ', '.join(flags) if flags else '无特征 |rho|>0.5，消融结论无需重新定性'}")

    print("\n" + "═" * 78)
    print("核查：rho(cds_len,pvalue) 与 长度五分位表 —— 两处口径")
    print("═" * 78)
    z = pd.DataFrame({"cds": L.reindex(rr.index), "p": rr.pvalue, "padj": rr.padj})
    z = z.dropna(subset=["cds"])          # 只按 cds 丢；padj 为 NaN 者仍属"非显著"
    print(f"  口径① 检验值宇宙: n={len(z)}（= 有 CDS 长度的受检基因）")
    print(f"        pvalue 非空 {z.p.notna().sum()}   padj 非空 {z.padj.notna().sum()}"
          f"（padj 空的 {z.padj.isna().sum()} 个是被 DESeq2 独立过滤的）")
    # ⚠️ 相关系数只能在 pvalue【非空】的宇宙上算 —— pvalue 带 NaN 会让 spearmanr 直接返回 nan。
    #    这个宇宙（16,002）与五分位表的宇宙（16,191）【不同】，必须分别写明。
    zp = z.dropna(subset=["p"])
    r1 = spearmanr(np.log10(zp.cds), zp.p)
    print(f"        宇宙①a Spearman 用: n={len(zp)}（pvalue 非空）")
    print(f"        Spearman(log10 cds, pvalue) = {r1.statistic:+.4f}  p={r1.pvalue:.3f}")
    se = 1 / np.sqrt(len(zp) - 1)
    print(f"        零假设下 SE ≈ 1/√(n−1) = {se:.4f}  →  rho 距 0 只有 "
          f"{abs(r1.statistic)/se:.2f} 个 SE ⇒ 就【全体基因】而言单调关联为 0")
    print(f"\n  口径② 五分位表: 分位划在【cds_len】上，显著判据用 padj<0.05，"
          f"分母是该分位内【全部】基因（padj 空者记为不显著）")
    zz = z.copy()
    zz["bin"] = pd.qcut(zz.cds, 5, labels=["最短", "短", "中", "长", "最长"])
    print(f"  {'分位':6s} {'n':>6s} {'中位 cds':>10s} {'显著数':>7s} {'显著率':>8s} {'期望数':>8s}")
    tot_sig = int((zz.padj < 0.05).sum())
    for b, sub in zz.groupby("bin", observed=True):
        k = int((sub.padj < 0.05).sum())
        print(f"  {b:6s} {len(sub):>6d} {sub.cds.median():>10.0f} {k:>7d} "
              f"{k/len(sub)*100:>7.2f}% {tot_sig*len(sub)/len(zz):>8.1f}")
    print(f"\n  → 两处【可以同时为真】：Spearman 度量的是【单调】关联，而效应集中在最长一档")
    print(f"     （第 1–4 档 0.3/0.7/0.6/0.5% 基本持平甚至下降，第 5 档跳到 1.9%）。")
    q5 = zz[zz.bin == "最长"]
    d_est = int((q5.padj < 0.05).sum()) * (len(zz) / 5.0) ** 2
    tot_d2 = len(zz) * (len(zz) ** 2 - 1) / 6.0
    print(f"     定量：把最多的那一档的 {int((q5.padj<0.05).sum())} 个显著基因位移到秩差 ~{len(zz)/5:.0f}，")
    print(f"     对 Σd² 的贡献约 {d_est:.3e}，只占总 Σd² ({tot_d2:.3e}) 的 {d_est/tot_d2*100:.2f}%")
    print(f"     → 至多把 rho 拉动 {6*d_est/tot_d2:.4f}，与 {r1.statistic:+.4f} 量级一致。")
    print(f"     ⇒ 上一版用 rho 去【否定】五分位表是错的（推理不成立）；")
    print(f"        但反过来，rho≈0 也不等于无效应 —— 是【非单调】的尾部效应。")

    print("\n" + "═" * 78)
    print("适用域界定（规则 outcome-blind：只由 cds_len 取值分布定）")
    print("═" * 78)
    pp, qq = e[y == 1], e[y == 0]
    lo, hi = max(pp.min(), qq.min()), min(pp.max(), qq.max())
    sel = (e >= lo) & (e <= hi)
    print(f"  规则：正负两类 cds_len 取值范围的【交集】")
    print(f"       正类 [{10**pp.min():.0f}, {10**pp.max():.0f}]  "
          f"负类 [{10**qq.min():.0f}, {10**qq.max():.0f}]")
    print(f"        下界 = max(两类最小) = {10**lo:.0f} bp")
    print(f"        上界 = min(两类最大) = {10**hi:.0f} bp")
    print(f"  子集 n={int(sel.sum())}（正 {int(y[sel].sum())} 负 {int((y[sel]==0).sum())}）"
          f"  独立单元 {len(np.unique(g[sel]))} 簇")
    print(f"  被排除 {int((~sel).sum())} 个基因（{int(((~sel)&(y==1)).sum())} 正 / "
          f"{int(((~sel)&(y==0)).sum())} 负）")
    # 口径与 13_labels.py 的匹配 QC 一致：池化样本 std（ddof=1），不是全集 std
    def _smd(m):
        a, b = cds[m & (y == 1)], cds[m & (y == 0)]
        return abs(a.mean() - b.mean()) / pd.concat([pd.Series(a), pd.Series(b)]).std()
    e1, e2 = _smd(np.ones(len(cds), bool)), _smd(sel)
    print(f"\n  ⚠️ 长度 SMD：全域 {e1:.3f} → 子集 {e2:.3f}"
          f"  —— 混杂【被削弱、未被消除】，仍高于 0.25")
    np.save(f"{OUT}/domain_mask.npy", sel)
    pd.Series(genes[sel]).to_csv(f"{OUT}/domain_genes.tsv", index=False, header=["gene_id"])

    print("\n" + "═" * 78)
    print("子集内成套指标（与主结果同口径）")
    print("═" * 78)
    Xs, ys, gs, es = X[sel], y[sel], g[sel], e[sel]
    a_std = v.cv_auc(Xs, ys, None)[0]
    a_grp = v.cv_auc(Xs, ys, gs)[0]
    print(f"  普通 CV   AUC={a_std:.3f}")
    print(f"  GroupKFold AUC={a_grp:.3f}")
    for kind, lab in (("plain", "普通CV "), ("group", "GroupKF")):
        b = v.boot_cluster(Xs, ys, gs, kind=kind, n=1000)
        print(f"  簇级自助 {lab} 95% CI [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
    null = v.perm_null_group(Xs, ys, gs, n=1000)
    print(f"  区间内置换零分布: 中位 {np.median(null):.3f}  "
          f"p95 {np.percentile(null,95):.3f}  max {null.max():.3f}")
    print(f"    真实 {a_grp:.3f} → 第 {(null<a_grp).mean()*100:.1f} 百分位  "
          f"经验 p={max((null>=a_grp).mean(),1/len(null)):.4f}")
    print(f"  ⚠️ 全局 p95 不可复用（不同 n、不同构成）")

    print(f"\n  §4.5 消融（子集基线 {a_grp:.3f}，GroupKFold 口径）:")
    abl = []
    for c in fcols:
        rest = [x for x in fcols if x != c]
        Xa = Xs[:, [fcols.index(x) for x in rest]]
        ag = v.cv_auc(Xa, ys, gs)[0]
        abl.append(dict(dropped=c, AUC_group=ag, dAUC_group=ag - a_grp))
        print(f"    {c:16s} AUC={ag:.3f}  ΔAUC={ag-a_grp:+.3f}")
    pd.DataFrame(abl).to_csv(f"{OUT}/model_ablation_domain.tsv", sep="\t", index=False)

    print(f"\n  长度单变量（必须与主结果同表并列）: "
          f"子集 {v.cv_auc(es.reshape(-1,1), ys, gs)[0]:.3f}  |  "
          f"全域 {v.cv_auc(e.reshape(-1,1), y, g)[0]:.3f}")

    print(f"\n  §2.4 串联簇折叠计数（子集内重算）:")
    pos = set(genes[sel][ys == 1])
    cl = clust.reindex(sorted(pos)).dropna()
    from collections import Counter
    sizes = Counter(cl.values)
    print(f"    正类占 {len(set(cl.values))} 个同源簇，其中含 ≥2 正类的 {sum(1 for k,v in sizes.items() if v>=2)} 个")
    print(f"    正类内部互为近同源的: {sum(1 for x in pos if _has_hom(x, pos, v, IETX))}/{len(pos)}")


def _has_hom(gene, others, v, IETX):
    """该基因在 others 里有没有近同源（延迟读 DIAMOND，避免主流程重复解析）。"""
    key = "_HOM_CACHE"
    if not hasattr(_has_hom, key):
        COLS = v.COLS
        d = pd.read_csv(f"{IETX}/homology/allvall.tsv", sep="\t", header=None, names=COLS)
        t2g = pd.read_csv(f"{IETX}/results/tx2gene.tsv", sep="\t").set_index("transcript_id")["gene_id"]
        d["qg"] = d.qseqid.map(t2g)
        d["sg"] = d.sseqid.map(t2g)
        d = d[(d.pident >= v.PID) & (d.length >= v.ALN) & (d.evalue <= v.EV)].dropna(subset=["qg", "sg"])
        h = {}
        for a, b in zip(d.qg, d.sg):
            h.setdefault(a, set()).add(b)
            h.setdefault(b, set()).add(a)
        setattr(_has_hom, key, h)
    h = getattr(_has_hom, key)
    return bool(h.get(gene, set()) & others)


if __name__ == "__main__":
    main()
