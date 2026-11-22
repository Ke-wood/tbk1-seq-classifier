#!/usr/bin/env python3
"""
13_labels.py —— IRS 标签构造（符号翻转版）

【符号】正类 = TBK1(RNAi) 中显著【下调】的基因（用户 2026-09-25 确认翻转）。
  理由：TBK1 是激活型激酶（磷酸化 IRF3/IRF7），敲降 → 下游降；TRAF 家族
  5/54 同向 p=0.0013、TBK1/IKK 轴 p=0.015 的通路级检验支持这一方向；
  工作文档-3 §2.1「在 TBK1(RNAi) 中被阻断」本就读作 log2FC 为负。

【标签定义】三套，供符号稳健性检验（用户要求「UP 集当正类再训一次」）：
  down_strict  lfc < -1.0  ┐ 主定义
  down_loose   lfc < -0.5  ┘
  up_loose     lfc > +0.5    对照定义（翻转前的方向）

【硬规则】
  · 剔除敲降靶基因 TBK1 自身（循环论证）
  · 负类 = 「完全无响应」（三时间点 pvalue 全 ≥ 0.5），按
    基础表达量 × 基因长度 × 转录本数 分层 1:1 匹配抽样
  · 排除计数全部记录
  · 三时间点同向【不作硬性要求】—— 只作为每基因的属性列输出。
    实测：宽松集 83 个正类里只有 9 个满足三时间点同向，严格集 15 个里只有 4 个。
    硬卡会把标签砍到个位数，反而毁掉模型。故降级为属性，由训练阶段自行决定是否加权。

产出
  results/label_{down_strict,down_loose,up_loose}.tsv   正/负类 + 属性
  results/label_summary.tsv                              各集计数 + 排除计数
  results/label_matching_qc.tsv                          协变量匹配质量（特征泄漏三查）

用法: $IETX_ROOT/env/bin/python 13_labels.py
"""
import os
import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
FEAT = f"{IETX}/features/features_fast.tsv"
TARGET = "h1SMcG0013953"      # TBK1 本体，必须剔除
PADJ, NS_PADJ = 0.05, 0.5
TOL = {"expr": 0.25, "gene_span": 0.25, "n_transcripts": 0.25}   # 匹配后允许的标准化偏差

LABELS = {
    "down_strict": dict(sign=-1, lfc=1.0),
    "down_loose":  dict(sign=-1, lfc=0.5),
    "up_loose":    dict(sign=+1, lfc=0.5),
}


def strata_table(feats):
    d = feats.copy()
    e = np.unique(np.quantile(d.expr.dropna(), np.linspace(0, 1, 11)))
    d["e_bin"] = pd.cut(d.expr, e, labels=False, include_lowest=True)
    d["l_bin"] = pd.qcut(d.gene_span.rank(method="first"), 3, labels=False)
    d["n_bin"] = pd.cut(d.n_transcripts, [0, 1, 2, 10 ** 6], labels=False)
    d["stratum"] = list(zip(d.e_bin, d.l_bin, d.n_bin))
    return d.set_index("gene_id")


def match(pos, pool, d, ratio=1):
    """最近邻匹配（三个协变量一起对齐）。
    ⚠️ 早先用「表达量十分位 × 长度三分位」硬分层 —— 失败了：表达量分布极度右偏，
       第 8/9 十分位横跨 10~335 CPM。正类恰好都落在这两个宽层的【高位】，
       而负类是从整个层里随机抽的，于是层分对了、SMD 仍有 0.55。
       分层太粗等于没分。最近邻才真的对齐。
    距离 = 三个协变量各自 log 后标准化、加权（表达量权重最高）的欧氏距离。
    从表达量最高的正类开始贪心挑（最难匹配的先挑），确定性、可复现。
    ⚠️ sorted() 不是装饰：pos/pool 是 set(str)，不排序则行序随 PYTHONHASHSEED 变，
       距离并列时 np.argmin 取的"第一个最小"就跟着变 —— 本函数曾因此自称确定性却并非。"""
    P = d.loc[[g for g in sorted(pos) if g in d.index]]
    C = d.loc[[g for g in sorted(pool) if g in d.index]]
    if P.empty or C.empty:
        return [], len(P)
    W = {"expr": 0.60, "gene_span": 0.25, "n_transcripts": 0.15}
    st = {}
    for c in W:
        v = np.log2(pd.concat([P[c], C[c]]).astype(float) + 1)
        st[c] = (v.mean(), v.std() or 1.0)

    def mat(df):
        return np.column_stack([
            (np.log2(df[c].astype(float) + 1) - st[c][0]) / st[c][1] * W[c] for c in W])

    Pz, Cz = mat(P), mat(C)
    used = np.zeros(len(C), bool)
    pick = []
    for i in np.argsort(-Pz[:, 0]):                 # 从高表达正类开始
        dd = np.linalg.norm(Cz - Pz[i], axis=1)
        dd[used] = np.inf
        for _ in range(ratio):
            j = int(np.argmin(dd))
            if not np.isfinite(dd[j]):
                break
            used[j] = True
            pick.append(C.index[j])
            dd[j] = np.inf
    return pick, 0


def main():
    res = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t", index_col=0)
    per_tp = {tp: pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_{tp}.tsv", sep="\t", index_col=0)
              for tp in ("3h", "6h", "24h")}
    feats = pd.read_csv(FEAT, sep="\t")
    counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    feats["expr"] = feats.gene_id.map(
        counts.div(counts.sum(axis=0), axis=1).mul(1e6).mean(axis=1))
    d = strata_table(feats)

    print(f"universe = {len(res)}（24h 实际检验基因数）")
    print(f"剔除靶基因 TBK1 = {TARGET}")

    # ── 负类池：两个版本，因为「完全无响应」与「可匹配表达量」在这份数据里冲突 ──
    # 实测：严格池(pvalue≥0.5 三点)的 p90 只有 23.7 CPM，而正类【全部】在 25 CPM 以上，
    # 6 个正类甚至超过严格池的最大值。原因很实在 —— 高表达基因几乎总会在某个时间点
    # 有一点响应(pvalue<0.5)，于是「完全无响应」这个定义恰好把高表达端整个删掉了。
    # 而表达量不匹配 = 模型学到「高表达=正类」的技术假信号（致命，且 AUC 看不出来）；
    # 负类纯度略降 = 标签噪声（会平滑地损失性能）。两者取前者更安全。
    # 故：主用宽松池（三时间点均不显著），严格池留作敏感性对照。
    ns_strict = set(res.index[res.pvalue >= NS_PADJ])
    for t in per_tp:
        ns_strict &= set(per_tp[t].index[per_tp[t].pvalue >= NS_PADJ])
    ns_loose = set(res.index[res.padj.isna() | (res.padj >= NS_PADJ)])
    for t in per_tp:
        ns_loose &= set(per_tp[t].index[per_tp[t].padj.isna()
                                      | (per_tp[t].padj >= NS_PADJ)])
    ns = ns_loose
    e = d["expr"]
    print(f"负类池(严格, pvalue≥{NS_PADJ} 三点): {len(ns_strict)}  "
          f"expr p90={e.reindex(sorted(ns_strict)).quantile(.9):.1f} max={e.reindex(sorted(ns_strict)).max():.0f}")
    print(f"负类池(宽松, 三点均不显著)        : {len(ns_loose)}  "
          f"expr p90={e.reindex(sorted(ns_loose)).quantile(.9):.1f} max={e.reindex(sorted(ns_loose)).max():.0f}"
          f"   ← 用作主负类池")
    print(f"  （正类 expr min={e.reindex(sorted(set(res.index[(res.padj<0.05)&(res.log2FoldChange<-0.5)]))).min():.0f}；"
          f"严格池够不到这个高度，故无法用）")

    rng = np.random.default_rng(0)
    summary, qc, sets = [], [], {}
    for name, cfg in LABELS.items():
        s, lfc = cfg["sign"], cfg["lfc"]
        sig = set(res.index[(res.padj < PADJ) & (s * res.log2FoldChange > lfc)])
        sig -= {TARGET}
        # 同向（非硬性）：该基因在另外两个时间点是否同号
        cons = {g for g in sig
                if all(g in per_tp[t].index and s * per_tp[t].loc[g, "log2FoldChange"] > 0
                       for t in per_tp)}
        # 相反方向显著者 = 排除项
        opp = set(res.index[(res.padj < PADJ) & (s * res.log2FoldChange < -lfc)]) - {TARGET}
        partial = sig - cons

        pool = sorted(ns - sig - opp)
        neg, short = match(sig, pool, d)
        nst = len(set(d.loc[[g for g in sig if g in d.index], "stratum"]))
        print(f"\n【{name}】方向={'下调' if s<0 else '上调'} lfc>{lfc}")
        print(f"  正类 {len(sig)}（三时间点同向 {len(cons)}；仅部分时间点 {len(partial)}）")
        print(f"  排除-反向显著 {len(opp)}   负类池 {len(pool)}")
        print(f"  负类匹配抽样 {len(neg)}（欠填层 {short}/{nst}）")

        rows = []
        # ⚠️ 【必须 sorted】sig 是 set(str)，而 CPython 的字符串哈希逐进程随机化
        #    （PYTHONHASHSEED）。直接 `for g in sig` 时**每次运行行序都不同**，
        #    集合内容却完全一样 —— 于是 label_*.tsv 内容等价、md5 不同。
        #    下游 15_train.py 的 RepeatedStratifiedKFold 是**按行号切折**的，
        #    permutation_null 也是按行号打乱标签 ⇒ 行序一变，AUC 就变。
        #    实测：同一份数据、同一脚本，up_loose 的 AUC 在 0.436~0.474 间游走。
        #    这不是"数据变了"，是**结果不可复现**。故此处及下面的 neg 一律排序。
        for g in sorted(sig):
            rows.append(dict(gene_id=g, label=1,
                             log2FC=res.loc[g, "log2FoldChange"], padj=res.loc[g, "padj"],
                             consistent=bool(g in cons),
                             log2FC_3h=per_tp["3h"].loc[g, "log2FoldChange"] if g in per_tp["3h"].index else np.nan,
                             log2FC_6h=per_tp["6h"].loc[g, "log2FoldChange"] if g in per_tp["6h"].index else np.nan))
        for g in sorted(neg):
            rows.append(dict(gene_id=g, label=0,
                             log2FC=res.loc[g, "log2FoldChange"], padj=res.loc[g, "padj"],
                             consistent=False, log2FC_3h=np.nan, log2FC_6h=np.nan))
        df = pd.DataFrame(rows).merge(feats, on="gene_id", how="left")
        df.to_csv(f"{OUT}/label_{name}.tsv", sep="\t", index=False)
        sets[name] = df

        # 协变量匹配质量（特征泄漏三查）
        row = {"label": name, "n_pos": len(sig), "n_neg": len(neg)}
        for cov in ("expr", "gene_span", "n_transcripts"):
            a = df.loc[df.label == 1, cov].astype(float)
            b = df.loc[df.label == 0, cov].astype(float)
            mus, sd = pd.concat([a, b]).mean(), pd.concat([a, b]).std()
            smd = abs(a.mean() - b.mean()) / (sd if sd else 1)     # 标准化均值差
            row[f"{cov}_pos"] = a.mean(); row[f"{cov}_neg"] = b.mean()
            row[f"{cov}_SMD"] = smd
            row[f"{cov}_pass"] = smd < TOL["expr"]
        qc.append(row)
        summary.append(dict(set=name, 方向="下调" if s < 0 else "上调", lfc=lfc,
                            正类=len(sig), 三点同向=len(cons), 仅部分时间点=len(partial),
                            反向显著_排除=len(opp), 负类=len(neg),
                            负类池=len(pool), 欠填层=short, 总层数=nst))

    pd.DataFrame(summary).to_csv(f"{OUT}/label_summary.tsv", sep="\t", index=False)
    q = pd.DataFrame(qc)
    q.to_csv(f"{OUT}/label_matching_qc.tsv", sep="\t", index=False)
    print("\n════ 协变量匹配质量（标准化均值差 SMD，<0.25 为通过）════")
    for _, r in q.iterrows():
        print(f"  {r['label']:12s} " + "  ".join(
            f"{c}: SMD={r[c+'_SMD']:.3f}{'✓' if r[c+'_pass'] else '✗'}"
            for c in ("expr", "gene_span", "n_transcripts")))
    print(f"\n  写出 label_{{down_strict,down_loose,up_loose}}.tsv / "
          f"label_summary.tsv / label_matching_qc.tsv")


if __name__ == "__main__":
    main()
