#!/usr/bin/env python3
"""65_utr_model.py —— M9 / M13 / M14 × 各自行集 / 共同行集

协议与 T-02 逐条相同（任务书 §2.6），直接复用 16_validation.py 的函数。

两组关键约束（任务书 §2.7 / §3.1b）：
  1. 行级丢弃逐模型独立 —— 若统一用 M14 的行集，M9 的样本量会被砍掉，
     就复现不出已发表的 0.6789 / 0.5401。
  2. 分组变量必须在【每次入模的行集上】现算。全基因集上算好再取子集，
     单例组编号不同，GroupKFold 划分随之改变（实测 iMacs 会得 0.5384 而非 0.5401）。
  3. ΔAUC 的 CI 用【配对】簇级自助：同一批簇重采样，两个模型各算一次再相减。

用法: OMP_NUM_THREADS=1 python 65_utr_model.py
"""
import os, sys, importlib.util
# IETX_HUMAN = 人类侧 T-02 目录，IETX_T03 = 本包（T-03）目录
if "IETX_HUMAN" not in os.environ:
    sys.exit("请先设置 IETX_HUMAN（T-02 目录）与 IETX_T03（T-03 目录）")
os.environ.setdefault("IETX_ROOT", os.environ["IETX_HUMAN"])
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

T2 = os.environ["IETX_HUMAN"]
R = os.environ["IETX_T03"]
_s = importlib.util.spec_from_file_location("V16", f"{T2}/scripts/16_validation.py")
V16 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V16)

M9 = V16.FEATURES
UTR5 = ["utr5_len", "utr5_gc", "utr5_cpg_oe", "utr5_mfe_per_nt"]
M13 = M9 + UTR5
M14 = M13 + ["utr3_len"]
MODELS = {"M9": M9, "M13": M13, "M14": M14}
COV_MIN, SEED, NBOOT, NPERM = V16.COV_MIN, 0, 300, 1000
EXPECT = {"GSE351189": 0.6789, "GSE309167": 0.5401}      # T-02 基线，必须复现


def prep(df, cols):
    """覆盖率过滤 + 行掩码（与 16_validation.usable 同规则）。"""
    keep = [c for c in cols if df[c].notna().sum() > COV_MIN * len(df)]
    X = df[keep].astype(float).values
    ok = ~np.isnan(X).any(axis=1)
    return keep, ok, X


def groups_for(df, idx, clust):
    """⚠️ 必须在【当前行集】上现算，不能预先在全基因集上算好再切。"""
    genes = df.loc[idx, "gene_id"].values
    gs = clust.reindex(genes)
    return gs.fillna(pd.Series(np.arange(len(genes)) + 10 ** 6,
                               index=genes)).astype(int).values


def boot_delta(Xa, Xb, y, g, n=NBOOT, seed=SEED):
    """配对簇级自助：同一批簇、同一批行，两个模型各算一次 AUC 再相减。"""
    uc = np.unique(g)
    bucket = {c: np.flatnonzero(g == c) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a9, _, _ = V16.cv_auc(Xa[idx], y[idx], g[idx], nrep=2)
        ab, _, _ = V16.cv_auc(Xb[idx], y[idx], g[idx], nrep=2)
        if a9 == a9 and ab == ab:
            out.append(ab - a9)
    return np.array(out)


def smd(a, b):
    return abs(a.mean() - b.mean()) / pd.concat([a, b]).std()


def main():
    feats = pd.read_csv(f"{R}/03_features/features_utr.tsv", sep="\t").set_index("gene_id")
    clust = pd.read_pickle(f"{T2}/homology/clusters.pkl")

    rows, abl, desc, shift, corrs = [], [], [], [], []
    for G in ("GSE351189", "GSE309167"):
        D = f"{T2}/人类侧_{G}"
        lab = pd.read_csv(f"{D}/05_label/label_both_pooled.tsv", sep="\t")
        df = lab.merge(feats.reset_index(), on="gene_id", how="left")
        expr = pd.read_csv(f"{D}/01_qc/cpm_mean.tsv", sep="\t", index_col=0)["expr"]
        df["expr"] = df.gene_id.map(expr)
        y_all = df["label"].values
        print(f"\n════ {G} ════ 标注 {len(df)}（正 {int(y_all.sum())}）")

        masks, keeps = {}, {}
        for m, cols in MODELS.items():
            keeps[m], masks[m], _ = prep(df, cols)
            print(f"  {m}: 特征 {len(keeps[m])}"
                  f"{'  丢弃 ' + str([c for c in cols if c not in keeps[m]]) if len(keeps[m]) < len(cols) else ''}"
                  f"  行 {int(masks[m].sum())}（正 {int(y_all[masks[m]].sum())}）")

        common = masks["M14"]
        res = {}
        for rmode in ("own", "common"):
            for m, cols in MODELS.items():
                mk = masks[m] if rmode == "own" else common
                idx = df.index[mk]
                X = df.loc[idx, keeps[m]].astype(float).values
                y = df.loc[idx, "label"].values
                g = groups_for(df, idx, clust)
                a, ap, ng = V16.cv_auc(X, y, g)
                ci = V16.boot_cluster(X, y, g, kind="group", n=NBOOT)
                lo, hi = np.percentile(ci, [2.5, 97.5]) if len(ci) else (np.nan, np.nan)
                null = V16.perm_null_group(X, y, g, n=NPERM)
                p5, p95 = np.percentile(null, [5, 95])
                pemp = max(float((null >= a).mean()), 1 / len(null))
                res[(rmode, m)] = dict(X=X, y=y, g=g, auc=a)
                rows.append(dict(dataset=G, model=m, rows=rmode,
                                 n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                                 n_genes=len(y), n_groups=len(np.unique(g)),
                                 auc=round(a, 4), auc_ci_lo=round(lo, 4),
                                 auc_ci_hi=round(hi, 4),
                                 auc_p5=round(p5, 4), auc_p95=round(p95, 4),
                                 perm_p=round(pemp, 4),
                                 delta_vs_M9=np.nan,
                                 delta_ci_lo=np.nan, delta_ci_hi=np.nan))
                print(f"  [{rmode:6s}] {m:4s} n={len(y):>5d}  AUC={a:.4f} "
                      f"[{lo:.3f}, {hi:.3f}]  零分布 p95={p95:.3f}  p={pemp:.4f}")

        # ── ΔAUC ──────────────────────────────────────────
        base = {(r, m): v["auc"] for (r, m), v in res.items()}
        for rec in rows[-6:]:
            rec["delta_vs_M9"] = round(rec["auc"] - base[(rec["rows"], "M9")], 4)
            if rec["rows"] == "common":      # 只有共同行集能配对
                d = boot_delta(res[("common", "M9")]["X"], res[("common", rec["model"])]["X"],
                               res[("common", "M9")]["y"], res[("common", "M9")]["g"])
                if len(d):
                    lo_, hi_ = np.percentile(d, [2.5, 97.5])
                    rec["delta_ci_lo"], rec["delta_ci_hi"] = round(lo_, 4), round(hi_, 4)
                    print(f"  配对 ΔAUC({rec['model']}−M9) = {rec['delta_vs_M9']:+.4f} "
                          f"[{lo_:+.4f}, {hi_:+.4f}]")

        # ── 消融：逐个去掉 UTR 特征（各自行集）─────────────
        for m in ("M13", "M14"):
            idx = df.index[masks[m]]
            y = df.loc[idx, "label"].values
            g = groups_for(df, idx, clust)
            a_full = base[("own", m)]
            for c in MODELS[m]:
                if c in M9:
                    continue
                cols = [x for x in keeps[m] if x != c]
                Xa = df.loc[idx, cols].astype(float).values
                ag = V16.cv_auc(Xa, y, g)[0]
                abl.append(dict(dataset=G, kind=m, dropped=c,
                                auc=round(ag, 4), dAUC=round(ag - a_full, 4)))

        # ── 描述统计 + 分层 SMD ─────────────────────────────
        idx = df.index[common]
        q90 = df.loc[idx, "expr"].quantile(0.90)
        sub = df.loc[idx]
        for c in M14:
            v = sub[c]
            a = v[sub.label == 1].dropna()
            b = v[sub.label == 0].dropna()
            rec = dict(dataset=G, feature=c, n_notna=int(v.notna().sum()),
                       median=round(v.median(), 4),
                       iqr=round(v.quantile(0.75) - v.quantile(0.25), 4),
                       min=round(v.min(), 4), max=round(v.max(), 4))
            for tag, s in (("full", sub),
                           ("low", sub[sub.expr < q90]),
                           ("high", sub[sub.expr >= q90])):
                x_, z_ = s[s.label == 1][c].dropna(), s[s.label == 0][c].dropna()
                rec[f"SMD_{tag}"] = round(smd(x_, z_), 3) if len(x_) > 3 and len(z_) > 3 else np.nan
                rec[f"p_{tag}"] = (f"{mannwhitneyu(x_, z_).pvalue:.2e}"
                                   if len(x_) > 3 and len(z_) > 3 else "")
                rec[f"npos_{tag}"], rec[f"nneg_{tag}"] = len(x_), len(z_)
            desc.append(rec)

        # ── 相关矩阵（Spearman）+ VIF ───────────────────────
        Xc = sub[M14].astype(float).dropna()
        sp = pd.DataFrame(spearmanr(Xc).correlation, index=M14, columns=M14)
        sp.insert(0, "feature", M14)
        sp.insert(0, "dataset", G)
        corrs.append(sp)
        v = V16.vif(Xc.values, M14)      # 与相关矩阵同一批行（已 dropna）
        print("  VIF: " + "  ".join(f"{k}={x:.2f}" for k, x in
                                    sorted(v.items(), key=lambda t: -t[1])[:5]))

        # ── UTR 长度偏移 ────────────────────────────────────
        for kind in ("both", "down", "up"):
            lk = pd.read_csv(f"{D}/05_label/label_{kind}_pooled.tsv", sep="\t")
            lk = lk.merge(feats[["utr5_len", "utr3_len"]].reset_index(),
                          on="gene_id", how="left")
            for c in ("utr5_len", "utr3_len"):
                p_, n_ = lk[lk.label == 1][c].dropna(), lk[lk.label == 0][c].dropna()
                if len(p_) < 5 or len(n_) < 5:
                    continue
                shift.append(dict(dataset=G, kind=kind, feature=c,
                                  pos_median=round(float(p_.median()), 1),
                                  neg_median=round(float(n_.median()), 1),
                                  ratio=round(float(p_.median() / n_.median()), 3),
                                  log2_ratio=round(float(np.log2(p_.median() / n_.median())), 3),
                                  smd=round(smd(p_, n_), 3)))

    out = f"{R}/06_utr"
    os.makedirs(out, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(f"{out}/model_utr.tsv", sep="\t", index=False)
    pd.DataFrame(abl).to_csv(f"{out}/ablation_utr.tsv", sep="\t", index=False)
    pd.DataFrame(desc).to_csv(f"{out}/utr_describe.tsv", sep="\t", index=False)
    pd.concat(corrs).to_csv(f"{out}/utr_corr.tsv", sep="\t", index=False)
    pd.DataFrame(shift).to_csv(f"{out}/length_shift_utr.tsv", sep="\t", index=False)

    # ── 任务书 §2.7 的硬断言：M9 必须精确复现 ──────────────
    for gse, exp in EXPECT.items():
        got = m[(m.dataset == gse) & (m.model == "M9") & (m.rows == "own")]["auc"].iloc[0]
        assert abs(got - exp) < 0.0002, (
            f"{gse} M9 复现失败：得 {got:.4f}，应为 {exp:.4f}。"
            f"检查分组变量是否在每次入模的行集上现算。")
        print(f"✓ {gse} M9 复现 {got:.4f}（基线 {exp}）")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
