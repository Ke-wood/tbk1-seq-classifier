#!/usr/bin/env python3
"""
28_cluster_threshold_sweep.py —— 同源簇定义的敏感性扫描

【为什么要做】`PID≥70 / ALN≥200 / EV≤1e-5` 是**一个冻结值，全程没有敏感性分析**。
  而全部核心数字都架在它上面：
      · 有效独立单元 115（全域）/ 93（适用域）
      · §1.4 的 ΔAUC = +0.047（普通 CV → GroupKFold）
      · 所有簇级自助 CI 的宽度
  评委一句"这个阈值是不是挑出来的"，目前**没有实验答案**。

【口径】网格写死：PID ∈ {50,60,70,80,90} × ALN ∈ {100,200}，EV 固定 1e-5。
  **70/200/1e-5 仍是唯一主口径**；本脚本报的是它周围的变化带，不是替代品。

【判据】见 判据预注册_outcome15_prereg.md §28。要点：
  A 稳健   ：AUC 极差 ≤0.03 且 CI 全重叠 → 115/93 改写为区间
  B 不稳健 ：极差 >0.03 或独立单元变化 >50% → **必须上报并限定 §1.4 / §五限制 4**
  C 退化   ：出现吞掉 >20% 基因的巨分量 → 标注"定义失效"，照报但不参与极差
  **明令禁止：不许从 10 格里挑 AUC 最高的一格当新主口径。**

【自检门】快速路径在 (70,200,1e-5) 的簇划分必须与
  `16_validation.clusters_from_diamond()` **逐基因完全一致**；不一致即中止。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 28_cluster_threshold_sweep.py
"""
import os
# ⚠️ 必须在 import numpy 之前设 —— 理由见 16_validation.py 头部（小矩阵上快 36 倍）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
AVS = f"{IETX}/homology/allvall.tsv"
# 主流程脚本所在目录（本文件夹会被拉到 $IETX_ROOT/posthoc/，而 16_validation.py 在 scripts/）
PIPE = os.environ.get("IETX_SCRIPTS", f"{IETX}/scripts")

GRID_PID = [50, 60, 70, 80, 90]
GRID_ALN = [100, 200]
EV = 1e-5
MAIN = (70.0, 200, 1e-5)
NBOOT = 300          # 够定 CI 形状；正式结论仍引用主口径的 1000 次
SEED = 0


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


def clusters_fast(avs, pid, aln, ev):
    """与 16_validation.clusters_from_diamond() **同构**的快速路径 ——
    唯一的差别是把已经读进内存的 avs 复用，避免在 10 格里重复解析 2.89M 行。
    （自检门会逐基因比对两者，保证同构不是嘴说的。）"""
    d = avs[(avs.pident >= pid) & (avs.length >= aln) & (avs.evalue <= ev)]
    d = d.dropna(subset=["qg", "sg"])
    d = d[d.qg != d.sg]
    ids = pd.Index(sorted(set(d.qg) | set(d.sg)))
    ix = {g: i for i, g in enumerate(ids)}
    M = coo_matrix((np.ones(len(d)), ([ix[q] for q in d.qg], [ix[s] for s in d.sg])),
                   shape=(len(ids), len(ids)))
    n, lab = connected_components(M, directed=False)
    return pd.Series(lab, index=ids), n


def partition(s):
    """把 Series(基因→簇号) 变成与编号无关的划分，用于比对两个实现是否同构。"""
    return frozenset(frozenset(g for g in s.index if s[g] == c)
                     for c in np.unique(s.values))


def groups_for(clust, genes):
    """把簇号对齐到入模基因；不在簇表里的基因各自给一个独立编号（= 单例）。"""
    return clust.reindex(genes).fillna(
        pd.Series(np.arange(len(genes)) + 10 ** 6, index=genes)).astype(int).values


def main():
    v16 = _mod("v16", os.path.join(PIPE, "16_validation.py"))

    print("═" * 78)
    print("28 · 同源簇阈值敏感性扫描")
    print("═" * 78)
    print(f"  网格 PID ∈ {GRID_PID} × ALN ∈ {GRID_ALN}；EV 固定 {EV}；"
          f"主口径 {MAIN[0]:.0f}/{MAIN[1]}/{MAIN[2]}")

    if not os.path.exists(AVS):
        raise SystemExit(f"找不到 {AVS} —— 需先解压随包 homology_allvall.tsv.gz 到 "
                         f"$IETX_ROOT/homology/allvall.tsv")
    avs = v16.read_avs(AVS)
    print(f"  all-vs-all 读入 {len(avs):,} 行")

    # ── 自检门 ────────────────────────────────────────────────────────────
    ref, _ = v16.clusters_from_diamond(AVS)
    got, _ = clusters_fast(avs, *MAIN[:2], MAIN[2])
    if partition(ref) != partition(got):
        raise SystemExit("!! 自检失败：快速路径与 clusters_from_diamond 的划分不一致 —— 停手")
    print(f"  ✓ 自检通过：快速路径在主口径上与 clusters_from_diamond 逐基因一致"
          f"（{len(ref)} 个基因有过阈同源）")

    # ── 分析集（与 17_domain / 19_ceiling 同口径）────────────────────────
    feats = v16.build_features()
    df = v16.load_label("down_loose", feats)
    X_all, y_all, fcols, idx = v16.usable(df, verbose=True)
    genes_all = df.loc[idx, "gene_id"].values
    dom = set(pd.read_csv(f"{OUT}/domain_genes.tsv")["gene_id"])
    m = np.array([g in dom for g in genes_all])
    print(f"\n  全域 n={len(y_all)}（正 {int(y_all.sum())}）")
    print(f"  适用域 n={int(m.sum())}（正 {int(y_all[m].sum())}）"
          f"  ← 掩码取自 results/domain_genes.tsv，不再自行推导")
    if int(m.sum()) != 114:
        raise SystemExit(f"!! 适用域 n={int(m.sum())} ≠ 114 —— 掩码口径变了，停手并核对 17_domain.py")

    # ── 扫描 ──────────────────────────────────────────────────────────────
    rows = []
    print("\n" + "═" * 78)
    print(f"{'PID':>5s} {'ALN':>5s} | {'过阈基因':>8s} {'簇数':>6s} {'最大簇':>7s} "
          f"{'占基因%':>8s} | {'全域AUC':>8s} {'CI':>15s} {'单元':>5s} "
          f"| {'适用域AUC':>9s} {'CI':>15s} {'单元':>5s}")
    print("═" * 78)
    for pid in GRID_PID:
        for aln in GRID_ALN:
            clust, ncomp = clusters_fast(avs, pid, aln, EV)
            sizes = pd.Series(clust.values).value_counts()
            biggest, biggest_frac = int(sizes.iloc[0]), sizes.iloc[0] / len(clust)

            g_all = groups_for(clust, genes_all)
            a_all, _, _ = v16.cv_auc(X_all, y_all, g_all)
            b_all = v16.boot_cluster(X_all, y_all, g_all, n=NBOOT)
            lo_all, hi_all = np.percentile(b_all, [2.5, 97.5])

            Xd, yd, gd = X_all[m], y_all[m], g_all[m]
            a_dom, _, _ = v16.cv_auc(Xd, yd, gd)
            b_dom = v16.boot_cluster(Xd, yd, gd, n=NBOOT)
            lo_dom, hi_dom = np.percentile(b_dom, [2.5, 97.5])

            degenerate = biggest_frac > 0.20
            print(f"{pid:>5d} {aln:>5d} | {len(clust):>8d} {ncomp:>6d} {biggest:>7d} "
                  f"{biggest_frac:>7.1%} | {a_all:>8.3f} [{lo_all:.3f},{hi_all:.3f}] "
                  f"{len(np.unique(g_all)):>5d} "
                  f"| {a_dom:>9.3f} [{lo_dom:.3f},{hi_dom:.3f}] "
                  f"{len(np.unique(gd)):>5d}"
                  + ("   ← 巨分量，定义失效" if degenerate else ""))
            rows.append(dict(PID=pid, ALN=aln, n_genes_over=len(clust), n_clusters=ncomp,
                             max_cluster=biggest, max_cluster_frac=round(biggest_frac, 4),
                             degenerate=degenerate,
                             AUC_global=a_all, CI_global_lo=lo_all, CI_global_hi=hi_all,
                             units_global=len(np.unique(g_all)),
                             AUC_domain=a_dom, CI_domain_lo=lo_dom, CI_domain_hi=hi_dom,
                             units_domain=len(np.unique(gd))))

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/cluster_threshold_sweep.tsv", sep="\t", index=False)

    # ── 判据应用 ──────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("判据应用（详见 判据预注册_outcome15_prereg.md §28）")
    print("═" * 78)
    ok = r[~r.degenerate]
    if len(ok) < len(r):
        print(f"  ⚠️ {len(r) - len(ok)} 格因巨分量被标为『定义失效』，不参与极差计算")
    for scope, ac, uc in (("全域", "AUC_global", "units_global"),
                          ("适用域", "AUC_domain", "units_domain")):
        rng_auc = ok[ac].max() - ok[ac].min()
        u0, u1 = ok[uc].min(), ok[uc].max()
        change = (u1 - u0) / u0
        main_row = ok[(ok.PID == 70) & (ok.ALN == 200)].iloc[0]
        verdict = "A 稳健" if (rng_auc <= 0.03 and change <= 0.50) else "B 不稳健"
        print(f"\n  [{scope}] AUC 极差 = {rng_auc:.3f}   独立单元 {u0}–{u1}（变化 {change:.0%}）")
        print(f"         主口径 AUC={main_row[ac]:.3f} CI[{main_row[ac.replace('AUC', 'CI') + '_lo']:.3f},"
              f"{main_row[ac.replace('AUC', 'CI') + '_hi']:.3f}]"
              f" 单元 {main_row[uc]}")
        print(f"         ⇒ 判据 **{verdict}**")
        if verdict.startswith("B"):
            print(f"         ⇒ ⚠️ 必须上报：§1.4 ΔAUC 与 §五限制 4 需加限定；"
                  f"**不得**只报主口径那一格")

    print(f"\n  写出 {OUT}/cluster_threshold_sweep.tsv（{len(r)} 行）")
    print("  ⚠️ 严禁从本表挑 AUC 最高的一格当新主口径 —— 那是选择泄漏。")
    print("  ⇒ 回传本表全文 + 这段终端输出。")


if __name__ == "__main__":
    main()
