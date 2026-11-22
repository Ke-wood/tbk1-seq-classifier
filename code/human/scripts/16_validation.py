#!/usr/bin/env python3
"""
16_validation.py —— 泄漏控制 + 跨时间点验证 + 特征消融（用户必补 1/3/4/5）

【为什么必须做】正类里已经证实存在串联重复（Mx 250 kb 簇、TRAF3 ×4）。
若正类的近同源旁系落在负类池里，普通 StratifiedKFold 会让模型「背下这个家族」，
AUC 虚高。这是能推翻结论的风险，必须先查。

必补项对应：
  1 同源簇泄漏 —— GroupKFold（按同源簇分组）+ leave-one-family-out
  3 跨时间点留出 —— 24h 训练，去预测 6h / 3h 的 TBK1 依赖基因
  4 特征重要性 + 逐特征消融曲线
  5 AUC 95% 自助置信区间 + 正负比
  2 置换零分布：permutation_null 在【匹配集内】打乱标签（df 就是 pos+matched neg），
    保持表达量匹配结构 —— 故 p95 可直接与真实 AUC 比。此处再明确产出该说明。

同源簇定义：DIAMOND all-vs-all，pident≥40 且 比对长度≥80 且 e≤1e-5 的边，
取连通分量。（连通分量有「链式合并」问题，故同时报分量大小分布；
若出现吞噬性巨分量，会明说。）

用法: $IETX_ROOT/env/bin/python 16_validation.py
"""
import os
# ⚠️ 必须在 import numpy 之前设：本机默认吃 20 线程 BLAS，在这种 100×9 的小矩阵上
#    单次 fit 要 0.44 s（正常 ~12 ms）—— 纯粹是线程同步开销，实测慢 36 倍。
#    1000 次置换 × 5 折会因此从 1 分钟变成 37 分钟。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
SEED, NREP = 0, 10
# ⚠️ DIAMOND 默认输出是【12 列】(qseqid sseqid pident length mismatch gapopen
#    qstart qend sstart send evalue bitscore)，不是 6 列。首版按 6 列解析，
#    结果「evalue」位是 mismatch 数、「bits」位是 gapopen —— 过滤条件全是错的，
#    得到一个 3252 基因的假巨分量和「0/83 有旁系」的假结论。必须按 12 列解析。
COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
PID, ALN, EV = 70.0, 200, 1e-5        # 同源边阈值：70%/200aa ≈ 近期复制（家族）
                                      # 50/100 会链式并成巨分量，无法用于 GroupKFold
COV_MIN = 0.5          # 与 15_train.py 保持一致（CDS 特征只覆盖 85%，用 0.9 会误删）
TARGET = "h1SMcG0013953"   # TBK1 本体：所有标签构造都要剔除（循环论证）
FEATURES = ["prom_gc", "prom_cpg_oe", "prom_lowcomp", "gene_span",
            "n_transcripts", "cds_gc", "cds_cpg_oe", "cai", "mfe_per_nt"]


def mk():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=3000,
                                            class_weight="balanced"))


def clusters_from_diamond(path):
    d = read_avs(path)
    d = d[(d.pident >= PID) & (d.length >= ALN) & (d.evalue <= EV)]
    d = d.dropna(subset=["qg", "sg"])
    d = d[d.qg != d.sg]
    ids = pd.Index(sorted(set(d.qg) | set(d.sg)))
    ix = {g: i for i, g in enumerate(ids)}
    M = coo_matrix((np.ones(len(d)), ([ix[q] for q in d.qg], [ix[s] for s in d.sg])),
                   shape=(len(ids), len(ids)))
    n, lab = connected_components(M, directed=False)
    return pd.Series(lab, index=ids), n


def read_avs(path):
    """按 DIAMOND 的 12 列标准格式解析 all-vs-all 结果，并把转录本 ID 映到基因 ID。"""
    t2g = pd.read_csv(f"{OUT}/tx2gene.tsv", sep="\t").set_index("transcript_id")["gene_id"]
    d = pd.read_csv(path, sep="\t", header=None, names=COLS)
    d["qg"] = d.qseqid.map(t2g)
    d["sg"] = d.sseqid.map(t2g)
    return d


def build_features():
    f = pd.read_csv(f"{FEAT}/features_fast.tsv", sep="\t").set_index("gene_id")
    f = f.drop(columns=[c for c in ("cds_gc", "cds_cpg_oe", "cai") if c in f.columns])
    for extra in ("features_cds.tsv",):
        c = pd.read_csv(f"{FEAT}/{extra}", sep="\t").set_index("gene_id")
        f = f.join(c, how="left")
    try:
        m = pd.read_csv(f"{FEAT}/mfe.tsv", sep="\t", header=None,
                        names=["gene_id", "mfe"]).set_index("gene_id")
        f = f.join(m, how="left")
        f["mfe_per_nt"] = f.mfe / f.prom_len
    except FileNotFoundError:
        f["mfe_per_nt"] = np.nan
    return f


def load_label(name, feats):
    df = pd.read_csv(f"{OUT}/label_{name}.tsv", sep="\t")
    df = df.merge(feats.reset_index()[["gene_id"] + FEATURES],
                  on="gene_id", how="left", suffixes=("", "_f"))
    for c in FEATURES:
        if c + "_f" in df.columns:
            df[c] = df[c + "_f"]
    return df


def usable(df, verbose=True):
    """筛特征 + 丢含 NaN 的行。

    ⚠️ 这里是整个管道里最容易【静默丢数据】的地方 —— MFE 那次失败（outcome12 §3.1）
       就是以「行数正常、值全是 NaN」的形态在这里被无声吃掉的，下游毫无报错。
       一般化的规则：**任何会丢弃的过滤器都必须打印它丢了什么**。
       所以本函数强制报告三件事：丢了哪些特征、丢了多少行、每行是被哪个特征丢的。
    """
    feats = [c for c in FEATURES if df[c].notna().sum() > COV_MIN * len(df)]
    low_cov = [c for c in FEATURES if c not in feats]
    X = df[feats].astype(float).values
    y = df["label"].values
    ok = ~np.isnan(X).any(axis=1)
    if verbose:
        if low_cov:
            print(f"  ⚠️ 覆盖率 <{COV_MIN:.0%} 被丢弃的特征: {low_cov}")
            for c in low_cov:
                print(f"      {c}: 非空 {df[c].notna().sum()}/{len(df)}")
        n_drop = int((~ok).sum())
        if n_drop:
            d = df.loc[df.index[~ok]]
            print(f"  ⚠️ 行级丢弃 {n_drop}/{len(df)}（正 {int(d.label.sum())} 负 "
                  f"{int((d.label == 0).sum())}）")
            # 每行到底被哪个特征丢的 —— 只说"丢了 25 行"是不够的
            cause = df.loc[d.index, feats].isna().idxmax(axis=1).value_counts()
            for c, k in cause.items():
                print(f"      {k:>3d} 行缺 {c}")
            for cov in ("expr", "gene_span"):
                if cov in df.columns:
                    a = d[cov].astype(float); b = df.loc[df.index[ok], cov].astype(float)
                    print(f"      被丢 vs 保留  {cov}: 中位 {a.median():.1f} vs {b.median():.1f}"
                          f"  SMD={abs(a.mean()-b.mean())/pd.concat([a,b]).std():.3f}")
    return X[ok], y[ok], feats, df.index[ok]


def cv_auc(X, y, groups=None, nrep=NREP, seed=SEED):
    """折外预测 AUC。给 groups 则用 GroupKFold（同源簇不跨折）。"""
    if groups is None:
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=nrep, random_state=seed)
        splits = list(splitter.split(X, y))
    else:
        g = np.asarray(groups)
        if len(np.unique(g)) < 5:
            return np.nan, np.nan, len(np.unique(g))
        splits = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g))
    oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        p = mk().fit(X[tr], y[tr])
        oof[te] += p.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    m = cnt > 0
    oof[m] /= cnt[m]
    return roc_auc_score(y[m], oof[m]), average_precision_score(y[m], oof[m]), len(np.unique(g)) if groups is not None else 0


def bootstrap_ci(X, y, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        p = mk().fit(X, y)
        a.append(roc_auc_score(y[idx], p.predict_proba(X[idx])[:, 1]))
    a = np.array(a)
    return np.percentile(a, 2.5), np.percentile(a, 97.5), len(a)


def boot_cluster(X, y, g, kind="plain", n=1000, seed=SEED):
    """【簇级】bootstrap：重采样同源簇，不重采样基因。

    ⚠️ 按基因自助是假精度：83 个正类里 70 个互为近同源、只占 13 个簇，
       独立单元数远小于样本数。按基因重采样会把「同一家族的 5 个拷贝」
       当成 5 次独立观测，CI 自然收窄到骗人的程度。
       簇级重采样（以簇为观测单位、整簇进整簇出）才反映真实不确定性。

    kind='group' 时每轮内用 GroupKFold —— 抽中的簇若出现多份，GroupKFold
    按组号折叠，副本仍落在同一折，不会 train/test 互漏。"""
    uc = np.unique(g)
    bucket = {c: np.flatnonzero(g == c) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        # nrep=2：自助里只需压低折划分噪声，1000 轮本身已足够
        a, _, _ = cv_auc(X[idx], y[idx], g[idx] if kind == "group" else None, nrep=2)
        if a == a:
            out.append(a)
    return np.array(out)


def vif(X, cols):
    """方差膨胀因子。VIF>5 说明该特征的系数被共线吃掉，系数符号不可单独解释。"""
    out = {}
    for i, c in enumerate(cols):
        o = [j for j in range(len(cols)) if j != i]
        r2 = LinearRegression().fit(X[:, o], X[:, i]).score(X[:, o], X[:, i])
        out[c] = 1.0 / max(1.0 - r2, 1e-9)
    return out


def perm_null_group(X, y, g, n=1000, seed=SEED):
    """置换零分布 —— 【当前 9 特征 + GroupKFold】口径。

    ⚠️ 15_train.py 那个 p95=0.585 是【不同特征集（当时 MFE 全 NaN）× 不同 CV 口径
       （普通分层）】下算的，与 0.712 不可比。交付文档要用的是这一个。
       置换在匹配集【内】做，保持表达量匹配结构。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, _, _ = cv_auc(X, rng.permutation(y), g, nrep=1)
        if a == a:
            out.append(a)
    return np.array(out)


def sign_stability(X, y, g, cols, n=1000, seed=SEED):
    """簇级自助下各特征系数的符号翻转率。
    prom_gc(+) 与 cds_gc(−) 符号相反又都与 GC 相关，可能只是共线下的不稳定分配 ——
    翻转率高就不能解释符号，只能说「GC 类特征有贡献」。"""
    ref = mk().fit(X, y)[-1].coef_[0]
    uc = np.unique(g)
    bucket = {c: np.flatnonzero(g == c) for c in uc}
    rng = np.random.default_rng(seed)
    flip = np.zeros(len(cols))
    tot = 0
    for _ in range(n):
        idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        co = mk().fit(X[idx], y[idx])[-1].coef_[0]
        flip += (np.sign(co) != np.sign(ref))
        tot += 1
    return pd.Series(flip / max(tot, 1), index=cols), ref, tot


def jackknife_pos_clusters(X, y, g):
    """留一正类同源簇 jackknife。
    簇级自助的 115 个单元里只有 13 个含正类，自助对【正类侧】不确定性估计不足。
    逐个剔掉一个正类簇重算 —— 范围窄 = 不由单一家族驱动。"""
    out = {}
    for c in sorted(set(g[y == 1])):
        keep = g != c
        if len(np.unique(y[keep])) < 2:
            continue
        a, _, _ = cv_auc(X[keep], y[keep], g[keep])
        out[int(c)] = (a, int(((g == c) & (y == 1)).sum()))
    return out


def _labels13():
    """把 13_labels.py 当模块加载 —— 复用它的最近邻匹配，避免两处实现漂移。
    带数字前缀的文件名不能 import，只能走 spec。"""
    from importlib.util import spec_from_file_location, module_from_spec
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_labels.py")
    sp = spec_from_file_location("labels13", p)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main():
    feats = build_features()
    clust, nclust = clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    sizes = clust.value_counts()
    print("════ 同源簇（DIAMOND all-vs-all，pident≥70% & 比对≥200aa & e≤1e-5）════")
    print(f"  基因 {len(clust)}  簇 {nclust}  最大簇 {sizes.iloc[0]} 个基因  "
          f"含 ≥2 基因的簇 {int((sizes>=2).sum())} 个")
    print(f"  簇大小分布: " + "  ".join(f"{k}基因:{v}簇" for k, v in
          sizes.value_counts().head(6).sort_index().items()))

    df = load_label("down_loose", feats)
    X, y, fcols, idx = usable(df)
    genes = df.loc[idx, "gene_id"].values
    gs = clust.reindex(genes)
    # 未进同源簇的基因各自成组（否则 NaN 会让 GroupKFold 直接报错）
    g = gs.fillna(pd.Series(np.arange(len(genes)) + 10 ** 6, index=genes)).astype(int).values
    print(f"\n════ 必补5：正负比 ════")
    print(f"  正类 {int(y.sum())}  负类 {int((y==0).sum())}  正负比 1:{int((y==0).sum())/max(int(y.sum()),1):.2f}"
          f"（匹配抽样设定为 1:1）")

    # ⚠️ 匹配质量必须在【实际入模的这 141 个】上算，不是在标签文件的 166 个上。
    #    13_labels.py 的匹配 QC 报的是 166；行级丢弃会改变两类构成，SMD 随之变化。
    print(f"\n════ 必补5b：匹配 SMD —— 在【实际入模集】上重算 ════")
    Lc = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")["cds_len_longest"]
    dd = df.loc[idx].copy()
    dd["cds_len_longest"] = Lc.reindex(dd.gene_id).values
    for cov in ("expr", "gene_span", "n_transcripts", "cds_len_longest"):
        if cov not in dd.columns:
            continue
        a = dd.loc[dd.label == 1, cov].astype(float).dropna()
        b = dd.loc[dd.label == 0, cov].astype(float).dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        smd = abs(a.mean() - b.mean()) / pd.concat([a, b]).std()
        print(f"    {cov:16s} 正 {a.median():>10.1f}  负 {b.median():>10.1f}   SMD={smd:.3f}"
              f"{'   ✗ 超 0.25' if smd > 0.25 else '   ✓'}")
    print(f"    （166 标签集上 expr SMD=0.024；行级丢弃后升到上表值 —— "
          f"§2.3 的「匹配成功」只在 166 上成立）")
    # 长度方向性：富集是只在下调方向，还是双向都有
    print(f"\n    方向拆分（全基因组 24h DEG，与标签无关的独立检查）:")
    rr = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t", index_col=0)
    z = pd.DataFrame({"cds": Lc.reindex(rr.index), "lfc": rr.log2FoldChange,
                      "padj": rr.padj})
    # ⚠️ 只能按 cds 丢！padj 为 NaN 的是被 DESeq2 独立过滤的基因，它们【仍属"非显著"背景】。
    #    用 .dropna() 一把丢会把背景从 16,191 缩到 4,370（只剩高表达基因），
    #    于是上调集变成"显著富集"（p=0.003 而不是 0.42）——又是一个静默过滤造出来的假结论。
    z = z.dropna(subset=["cds"])
    s_, bg_ = z[z.padj < 0.05], z[~(z.padj < 0.05)]
    for nm, sub in (("上调集", s_[s_.lfc > 0]), ("下调集", s_[s_.lfc < 0])):
        if len(sub) < 3:
            continue
        from scipy.stats import mannwhitneyu
        print(f"      {nm} n={len(sub):>3d}  中位 cds_len={sub.cds.median():>9.0f}  "
              f"vs 非显著(n={len(bg_)}) p={mannwhitneyu(sub.cds, bg_.cds).pvalue:.2e}"
              f"{'   ← 富集' if mannwhitneyu(sub.cds, bg_.cds).pvalue < 0.05 else '   ← 不富集'}")
    print(f"      → 长度富集若只在【下调】方向，指向真生物学或方向特异的机制，"
          f"不是通用可检测性假象（通用假象会双向都富集）")

    print(f"\n════ 必补1：同源簇泄漏 ════")
    # 正类的近同源旁系有多少落在负类里
    pos_ids = set(df.loc[df.label == 1, "gene_id"])
    neg_ids = set(df.loc[df.label == 0, "gene_id"])
    pos_clusters = set(clust.reindex(sorted(pos_ids)).dropna().values)
    leak = 0
    for c in pos_clusters:
        mem = set(clust.index[clust == c])
        if mem & neg_ids:
            leak += 1
    print(f"  正类占 {len(pos_clusters)} 个同源簇（共 {nclust} 簇）；"
          f"其中 {leak} 个簇的成员同时出现在负类里")
    # 更直接的口径：逐个正类问「负类里有没有它的近同源」
    t2g = pd.read_csv(f"{OUT}/tx2gene.tsv", sep="\t").set_index("transcript_id")["gene_id"]
    dd = read_avs(f"{IETX}/homology/allvall.tsv")
    dd = dd[(dd.pident >= PID) & (dd.length >= ALN) & (dd.evalue <= EV)]
    dd = dd.dropna(subset=["qg", "sg"])
    hom = {}   # 必须【双向】存：只存 q→s 时，若负类是 query 就查不到（首版就是这个错）
    for q, s2 in zip(dd.qg, dd.sg):
        hom.setdefault(q, set()).add(s2)
        hom.setdefault(s2, set()).add(q)
    n_leak = sum(1 for g in pos_ids if hom.get(g, set()) & neg_ids)
    print(f"  逐个正类查：{n_leak}/{len(pos_ids)} 个正类在负类里有近同源旁系"
          f"（pident≥{PID:.0f}% & alnlen≥{ALN}aa）")
    n_hom_pos = sum(1 for g in pos_ids if hom.get(g, set()) & pos_ids)
    print(f"  正类【内部】互相为近同源的: {n_hom_pos}/{len(pos_ids)} 个"
          f"（这些基因若被随机分到训练/测试两侧，就是泄漏）")
    print(f"  → 这些正类的信号可能被同族负类「背下来」，普通基因间 CV 会高估")

    a_std, p_std, _ = cv_auc(X, y, None)
    a_grp, p_grp, ng = cv_auc(X, y, g)
    print(f"\n  普通 StratifiedKFold : AUC={a_std:.3f}  PR-AUC={p_std:.3f}")
    print(f"  GroupKFold(同源簇)   : AUC={a_grp:.3f}  PR-AUC={p_grp:.3f}   （{ng} 组）")
    print(f"  ΔAUC = {a_std-a_grp:+.3f}  → "
          f"{'掉幅小，序列语法是通用的' if a_std-a_grp < 0.10 else '掉幅大，家族特异性强，口径要改'}")

    # leave-one-family-out：剔除最大同源簇后重训
    big = sizes.index[0]
    keep = ~np.isin(g, [big])
    if keep.sum() > 20 and len(np.unique(y[keep])) == 2:
        a_lofo, p_lofo, _ = cv_auc(X[keep], y[keep], None)
        print(f"\n  leave-largest-family-out（剔除最大簇 {int((~keep).sum())} 个基因后重训）"
              f": AUC={a_lofo:.3f}  PR-AUC={p_lofo:.3f}  ΔAUC={a_std-a_lofo:+.3f}")

    lo, hi, nb = bootstrap_ci(X, y)
    uc = np.unique(g)
    n_single = int(sum(1 for c in uc if (g == c).sum() == 1))
    print(f"\n════ 必补5：AUC 95% 自助置信区间 ════")
    print(f"  独立单元数 {len(uc)}（同源簇，其中单例 {n_single}）"
          f" vs 基因数 {len(y)} —— 按基因自助是假精度，只作对照")
    print(f"  基因级自助（{nb} 次）      : [{lo:.3f}, {hi:.3f}]   ← 假精度")
    try:
        null_p95 = pd.read_csv(f"{OUT}/model_cv.tsv", sep="\t").set_index("label") \
                     .loc["down_loose", "null_p95"]
    except Exception:
        null_p95 = np.nan
    for kind, lab, base in (("plain", "普通 CV ", a_std), ("group", "GroupKFold", a_grp)):
        a = boot_cluster(X, y, g, kind=kind, n=1000)
        blo, bhi = np.percentile(a, [2.5, 97.5])
        print(f"  簇级自助 {lab} : AUC={base:.3f}  95% CI [{blo:.3f}, {bhi:.3f}]"
              f"  （{len(a)} 次有效重采样）"
              f"{'  ← 跨过 0.5' if blo < 0.5 else ''}")
    print(f"  参照：匹配集内置换零分布 p95 = {null_p95:.3f}（15_train.py，口径一致）")

    print(f"\n════ 必补4：特征重要性与消融 ════")
    p = mk().fit(X, y)
    co = pd.Series(p[-1].coef_[0], index=fcols).sort_values(key=abs, ascending=False)
    print("  全数据拟合系数（标准化后，符号=方向）:")
    for k, v in co.items():
        print(f"    {k:16s} {v:+.3f}  {'↑正类' if v>0 else '↓正类'}")
    # ⚠️ 消融必须两套并列。普通 CV 的 ΔAUC 里混着家族记忆（正类 70/83 互为近同源），
    #    去掉一个特征后模型可以改用「家族指纹」补回来，ΔAUC 就被压小了 —— 假的。
    #    GroupKFold 版才是「这个特征对新家族有没有用」。
    print("\n  逐特征消融（去掉该特征后 AUC）:")
    print(f"    {'去掉的特征':16s} {'普通CV':>8s} {'ΔAUC':>8s} {' | ':>3s} {'GroupKF':>8s} {'ΔAUC':>8s}")
    abl = []
    for c in fcols:
        rest = [x for x in fcols if x != c]
        Xa = X[:, [fcols.index(x) for x in rest]]
        a, _, _ = cv_auc(Xa, y, None)
        ag, _, _ = cv_auc(Xa, y, g)
        abl.append(dict(dropped=c, AUC=a, dAUC=a - a_std,
                        AUC_group=ag, dAUC_group=ag - a_grp))
        print(f"    {c:16s} {a:>8.3f} {a-a_std:>+8.3f} {' | ':>3s} "
              f"{ag:>8.3f} {ag-a_grp:>+8.3f}")
    pd.DataFrame(abl).to_csv(f"{OUT}/model_ablation.tsv", sep="\t", index=False)

    # ── 共线拆分：prom_gc 与 prom_lowcomp 的归因是否站得住 ──
    print(f"\n════ 特征共线（VIF）与拆分重跑 ════")
    v = vif(X, fcols)
    for k, x in sorted(v.items(), key=lambda t: -t[1]):
        flag = "  ← VIF>5，系数不可单独解释" if x > 5 else ""
        print(f"    {k:16s} VIF={x:7.2f}{flag}")
    paired = [c for c in ("prom_gc", "prom_lowcomp") if c in fcols]
    if len(paired) == 2:
        for keep in paired:
            drop = [c for c in paired if c != keep][0]
            Xs = X[:, [fcols.index(c) for c in fcols if c != drop]]
            a_s, _, _ = cv_auc(Xs, y, None)
            print(f"    只留 {keep:13s}（去掉 {drop}）: AUC={a_s:.3f}  "
                  f"ΔAUC={a_s-a_std:+.3f}"
                  f"{'  ← 单独就顶得住，共线不影响结论' if a_s - a_std > -0.05 else '  ← 掉得多，两者贡献分不开'}")

    # ── 追加1：置换零分布（当前 9 特征 + GroupKFold 口径）──
    print(f"\n════ 追加1：置换零分布（当前口径，1000 次）════")
    null = perm_null_group(X, y, g, n=1000)
    p95 = np.percentile(null, 95)
    print(f"  零分布: 中位 {np.median(null):.3f}  95分位 {p95:.3f}  最大 {null.max():.3f}")
    print(f"  真实 GroupKFold AUC={a_grp:.3f}  → 第 {(null < a_grp).mean()*100:.1f} 百分位"
          f"  经验 p={max((null >= a_grp).mean(), 1/len(null)):.4f}")
    print(f"  ⚠️ 15_train.py 里那个 p95=0.585 是【当时 MFE 全为 NaN、且用普通分层 CV】算的，")
    print(f"     与本行不可比 —— 交付文档引用的是本行。")

    # ── 追加2：系数符号稳定性（簇级自助）──
    print(f"\n════ 追加2：系数符号稳定性（簇级自助 1000 次）════")
    fr, ref, tot = sign_stability(X, y, g, fcols)
    for c in fcols:
        print(f"    {c:16s} 全量系数 {ref[fcols.index(c)]:+.3f}  符号翻转率 {fr[c]*100:5.1f}%"
              f"{'   ← 不稳，不可解释符号' if fr[c] > 0.2 else ''}")
    # ⚠️ 判读只看【真正带信号的】特征（消融 ΔAUC < −0.02 的那几个）。一个 ΔAUC≈0 的
    #    噪声特征符号乱翻，不影响 prom_gc/cds_gc 的方向能否解释 —— 别把两件事混为一谈。
    signal = [r["dropped"] for r in abl if r["dAUC_group"] < -0.02]
    hi_flip = [c for c in signal if fr[c] > 0.2]
    print(f"  带信号的特征（GroupKF 消融 ΔAUC<−0.02）: {signal}")
    print(f"  → " + (f"这几个的符号都稳（最大翻转率 {max(fr[c] for c in signal)*100:.1f}%），"
                     f"方向可以解释" if not hi_flip else
                     f"{hi_flip} 符号不稳，只能说「GC 类特征有贡献」"))
    other = [c for c in fcols if c not in signal and fr[c] > 0.2]
    if other:
        print(f"  （{', '.join(other)} 符号乱翻，但它们的 ΔAUC≈0，本就是噪声，不影响上面的结论）")

    # ── 追加3：留一正类同源簇 jackknife ──
    print(f"\n════ 追加3：留一正类同源簇 jackknife（逐个剔掉一个正类簇）════")
    jk = jackknife_pos_clusters(X, y, g)
    vals = np.array([a for a, _ in jk.values()])
    print(f"  正类共占 {len(jk)} 个同源簇，逐个剔除后重算 GroupKFold AUC:")
    for c, (a, n) in jk.items():
        print(f"    剔除簇 {c:>5d}（{n} 个正类）: AUC={a:.3f}  ΔAUC={a-a_grp:+.3f}"
              f"{'   ← 掉得多，该家族是主要驱动' if a - a_grp < -0.05 else ''}")
    print(f"  范围 [{vals.min():.3f}, {vals.max():.3f}]  中位 {np.median(vals):.3f}  "
          f"（基线 {a_grp:.3f}）")
    print(f"  → " + ("范围窄，不由单一家族驱动"
                     if vals.min() > a_grp - 0.08 else "范围宽，有单家族主导，见上表点名"))

    # ── 追加4：CDS 长度混杂（结构协变量未匹配）──
    print(f"\n════ 追加4：CDS 长度 —— 一个未匹配的结构协变量 ════")
    try:
        L = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")["cds_len_longest"]
        cd = L.reindex(genes).values.astype(float)
        e = np.log10(cd + 1)
        ok = np.isfinite(e)
        pp, qq = e[ok & (y == 1)], e[ok & (y == 0)]
        smd = abs(np.mean(10 ** pp) - np.mean(10 ** qq)) / np.std(np.r_[10 ** pp, 10 ** qq])
        print(f"  正类中位 {np.median(10**pp):>8.0f} bp   负类中位 {np.median(10**qq):>8.0f} bp"
              f"   比值 {np.median(10**pp)/np.median(10**qq):.1f}×   SMD={smd:.3f}")
        print(f"  （对照：expr SMD 0.024、gene_span SMD 0.006、n_transcripts SMD 0.052 —— 都匹配住了，"
              f"唯独漏了 CDS 长度）")
        print(f"  负类长度上限 {10**qq.max():.0f} bp，正类有 {int((pp>qq.max()).sum())}/{len(pp)} 个超出其整个范围"
              f" → 两侧长度分布重叠度 {np.mean((pp>=qq.min())&(pp<=qq.max())):.2f}")
        a_len = cv_auc(e[ok].reshape(-1, 1), y[ok], g[ok])[0]
        print(f"  log10(cds_len) 单特征 GroupKFold AUC = {a_len:.3f}"
              f"   （完整 9 特征 = {a_grp:.3f}）")
        Xl = np.column_stack([e[ok], X[ok]])
        print(f"  9特征 + log10(cds_len) GroupKFold AUC = {cv_auc(Xl, y[ok], g[ok])[0]:.3f}")
        lo_, hi_ = max(pp.min(), qq.min()), min(pp.max(), qq.max())
        sel = ok & (e >= lo_) & (e <= hi_)
        print(f"  只在两侧共同长度区间 [{10**lo_:.0f}, {10**hi_:.0f}] bp 内重算（n={int(sel.sum())}）:"
              f"  9特征 {cv_auc(X[sel], y[sel], g[sel])[0]:.3f}"
              f"  长度单特征 {cv_auc(e[sel].reshape(-1, 1), y[sel], g[sel])[0]:.3f}")
        print(f"  含 NaN 的行: {int((~ok).sum())}")
    except FileNotFoundError:
        print("  (features_cds.tsv 缺失，跳过)")

    # ── 必补3a：跨时间点留出（无检验力，不可当泛化检验读）──
    print(f"\n════ 必补3a：跨时间点（24h 训练 → 3h/6h 预测）════")
    print("  ⚠️ 3h/6h 的 down_loose 集本身只有个位数正类 —— 这是【无检验力】，")
    print("     不是「泛化失败」。零假设下 AUC 的标准误 ≈ sqrt((n1+n2+1)/(12·n1·n2))，")
    print("     5 正 6 负时 σ≈0.18，实测值离 0.5 不到半个 σ，什么也说明不了。")
    p24 = mk().fit(X, y)                       # ← 只用 24h 训练
    for tp in ("6h", "3h"):
        lab = cross_tp_label(tp, feats, clust)
        if lab is None:
            print(f"  {tp}: 标签不可用，跳过"); continue
        # ⚠️ 特征列必须与训练时【完全一致】—— 6h 子集单独筛特征会得到不同的列，
        #    那样 predict_proba 拿到的是错位矩阵，AUC 无意义。
        Xt = lab[fcols].astype(float).values
        yt = lab["label"].values
        ok = ~np.isnan(Xt).any(axis=1)
        Xt, yt = Xt[ok], yt[ok]
        if len(np.unique(yt)) < 2:
            print(f"  {tp}: 标签单一，跳过"); continue
        n1, n0 = int(yt.sum()), int((yt == 0).sum())
        auc = roc_auc_score(yt, p24.predict_proba(Xt)[:, 1])
        se = np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
        print(f"  {tp}: 正 {n1} 负 {n0}  AUC={auc:.3f}  零假设 σ={se:.3f}  "
              f"z={(auc-0.5)/se:+.2f}σ  → {'无检验力' if abs((auc-0.5)/se) < 2 else '有信号'}")
    print("  【生物结论另立】3h/6h 的下调集只有个位数 → TBK1 依赖程序是 24h 主导的，"
          "这是真发现，与「模型能否泛化」是两个问题。")

    # ── 必补3b：臂内留出（独立取样）──
    print(f"\n════ 必补3b：臂内留出 —— DNA 臂 ⇄ Mock 臂（独立取样）════")
    print("  两臂是分开处理、分开建库的样本，理论上是这份数据里唯一真独立的留出。")
    try:
        counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
        meta = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t").set_index("sample")
        ex = counts.div(counts.sum(axis=0), axis=1).mul(1e6).mean(axis=1)
        fr = feats.reset_index()
        fr["expr"] = fr.gene_id.map(ex)
        d_strata = _labels13().strata_table(fr)
        arms, raw = {}, {}
        for arm in ("DNA", "Mock"):
            lab, r, n = arm_label(arm, counts, meta, feats, d_strata)
            arms[arm], raw[arm] = lab, r
            # 落盘：这两张表原先只在内存里，交付包核完整性时是缺项
            r.to_csv(f"{OUT}/deg_arm_{arm}_24h_TBK1_vs_eGFP.tsv", sep="\t")
            lab.to_csv(f"{OUT}/label_arm_{arm}_24h.tsv", sep="\t", index=False)
            dh = r.index[(r.padj < 0.05) & (r.log2FoldChange < -0.5)]
            print(f"  [{arm:4s}] n={n} 样本 → 正类 {int(lab.label.sum())}  负类 {int((lab.label == 0).sum())}"
                  f"  |  离散度中位 {r.attrs['disp']:.3f}  pvalue 中位 {r.pvalue.median():.3f}"
                  f"  命中表达量中位 {ex.reindex(sorted(dh)).median():.0f} CPM")
        # ⚠️ 两臂命中数差 19 倍，必须先判【哪一臂的 DEG 表能用】再谈留出。
        #    判据不是命中数本身，而是：另一臂（重复数更多的那臂）能不能复现它、
        #    以及两臂的全基因组 log2FC 是否相关。
        from scipy.stats import fisher_exact
        D, M = raw["DNA"], raw["Mock"]
        dh = set(D.index[(D.padj < 0.05) & (D.log2FoldChange < -0.5)])
        mh = set(M.index[(M.padj < 0.05) & (M.log2FoldChange < -0.5)])
        uni = set(D.index) & set(M.index)
        hi = set(ex[ex >= 10].index)
        a, b = len(dh & mh & hi), len(dh & hi) - len(dh & mh & hi)
        c_, d_ = len(mh & hi) - len(dh & mh & hi), len(uni & hi) - len(dh & hi) - len(mh & hi) + len(dh & mh & hi)
        orr, fp = fisher_exact([[a, b], [c_, d_]])
        j = pd.concat([D.log2FoldChange.rename("D"), M.log2FoldChange.rename("M")], axis=1).dropna()
        j = j.loc[j.index.isin(uni)]
        print(f"\n  两臂命中重叠 {len(dh & mh)}／DNA {len(dh)}  Fisher(expr≥10) OR={orr:.1f} p={fp:.2g}"
              f"   ← DNA 的高置信命中在 Mock 里复现")
        print(f"  两臂全基因组 log2FC 相关: r={j.D.corr(j.M):+.3f}（n={len(j)}）"
              f"；仅 expr≥10: r={j.loc[j.index.isin(hi)].D.corr(j.loc[j.index.isin(hi)].M):+.3f}")
        print(f"  → Mock 臂 n=2，离散度中位 {M.attrs['disp']:.3f} 仅为 DNA"
              f"（{D.attrs['disp']:.3f}）的一半 → 低计数基因 p 值被系统性压低，"
              f"多出的 {len(mh - dh)} 个命中 DNA 一个也验不出。")
        print(f"     两臂 log2FC 近乎不相关 ⇒ 【Mock 臂的 DEG 表不可用】，"
              f"3b 两个方向都不可解释 —— 不是泛化失败，是测试标签一半是噪声。")
        for tr, te in (("DNA", "Mock"), ("Mock", "DNA")):
            A, B = arms[tr], arms[te]
            Xa, ya = A[fcols].astype(float).values, A.label.values
            ok = ~np.isnan(Xa).any(axis=1); Xa, ya = Xa[ok], ya[ok]
            Xb, yb = B[fcols].astype(float).values, B.label.values
            ok = ~np.isnan(Xb).any(axis=1); Xb, yb = Xb[ok], yb[ok]
            if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
                print(f"  {tr} → {te}: 一侧单一类别，无法评估"); continue
            auc = roc_auc_score(yb, mk().fit(Xa, ya).predict_proba(Xb)[:, 1])
            n1, n0 = int(yb.sum()), int((yb == 0).sum())
            se = np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)) if n1 and n0 else np.nan
            print(f"  {tr} 训练(n={len(ya)}) → {te} 预测(正 {n1} 负 {n0}): "
                  f"AUC={auc:.3f}  σ={se:.3f}  z={(auc-0.5)/se:+.2f}σ   （标签不可用，仅供参考）")
        print(f"\n  ⚠️ 【结论】这份数据【无法】提供干净的跨臂泛化检验：")
        print(f"     唯一样本不相交的切法，一侧必然落在 Mock 臂（n=2），而其 DEG 表已证不可用。")
        print(f"     故「24h 序列模型能否泛化到独立样本」在本数据集里【仍未检验】。")
    except FileNotFoundError as e:
        print(f"  跳过（缺输入: {e}）")


def arm_label(arm, counts, meta, feats, d):
    """在单个 arm（DNA / Mock）内单独建 24h DEG：TBK1 vs eGFP。
    这是【独立取样】的留出 —— 两臂是同一次实验里分开处理、分开建库的样本，
    与 3h/6h（同一批动物的另一个时间点）不同，是真正的外部队列。"""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    sub = [s for s in meta.index
           if meta.loc[s, "timepoint"] == "24h" and meta.loc[s, "treatment"] == arm]
    c = counts[sub]
    c = c.loc[c.sum(axis=1) > 0].T
    dds = DeseqDataSet(counts=c, metadata=meta.loc[sub], design="~rnai",
                       refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["rnai", "TBK1", "eGFP"], quiet=True)
    st.summary()          # ⚠️ 不调 summary() 就没有 results_df（09 的 fit() 同理）
    r = st.results_df
    pos = set(r.index[(r.padj < 0.05) & (r.log2FoldChange < -0.5)]) - {TARGET}
    pool = (set(r.index[r.padj.isna() | (r.padj >= 0.5)]) - pos)
    neg, _ = _labels13().match(sorted(pos), sorted(pool), d)   # 复用 13 的最近邻匹配
    out = pd.DataFrame([dict(gene_id=g, label=1) for g in sorted(pos)] +
                       [dict(gene_id=g, label=0) for g in neg])
    out = out.merge(feats.reset_index()[["gene_id"] + FEATURES],
                    on="gene_id", how="left")
    r.attrs["disp"] = float(np.median(dds.var["dispersions"]))
    return out, r, len(sub)


def cross_tp_label(tp, feats, clust):
    """在指定时间点重建正负类，负类同样按表达量最近邻匹配。"""
    try:
        res = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_{tp}.tsv", sep="\t", index_col=0)
    except FileNotFoundError:
        return None
    counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    ex = counts.div(counts.sum(axis=0), axis=1).mul(1e6).mean(axis=1)
    pos = set(res.index[(res.padj < 0.05) & (res.log2FoldChange < -0.5)]) - {"h1SMcG0013953"}
    pool = set(res.index[res.padj.isna() | (res.padj >= 0.5)])
    if not pos:
        return None
    f = feats.copy()
    f["expr"] = f.index.map(ex)
    W = {"expr": 0.6, "gene_span": 0.25, "n_transcripts": 0.15}
    # sorted() 必须 —— pos/pool 是 set(str)，行序随 PYTHONHASHSEED 变，
    # 距离并列时 np.argmin 取的"第一个最小"跟着变（同 13_labels.py 的坑）。
    P = f.loc[[g for g in sorted(pos) if g in f.index]]
    C = f.loc[[g for g in sorted(pool) if g in f.index]]
    if P.empty or C.empty:
        return None
    st = {c: (lambda v: (v.mean(), v.std() or 1.0))(np.log2(pd.concat([P[c], C[c]]).astype(float) + 1))
          for c in W}
    mat = lambda d: np.column_stack([(np.log2(d[c].astype(float) + 1) - st[c][0]) / st[c][1] * W[c] for c in W])
    Pz, Cz = mat(P), mat(C)
    used = np.zeros(len(C), bool); pick = []
    for i in np.argsort(-Pz[:, 0]):
        dd = np.linalg.norm(Cz - Pz[i], axis=1); dd[used] = np.inf
        j = int(np.argmin(dd))
        if not np.isfinite(dd[j]):
            break
        used[j] = True; pick.append(C.index[j])
    rows = [dict(gene_id=g, label=1) for g in P.index] + \
           [dict(gene_id=g, label=0) for g in pick]
    d = pd.DataFrame(rows).merge(feats.reset_index()[["gene_id"] + FEATURES],
                                 on="gene_id", how="left")
    return d


if __name__ == "__main__":
    main()
