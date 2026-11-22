#!/usr/bin/env python3
"""
09_tbk1_deseq.py —— TBK1 vs eGFP 主标签分析 + DESeq2 标准质控 + 标签构造

【定性】回归工作文档-3 §2.1 的原始设计，不是路线被迫调整。
  原始设计：正类 = 「TBK1 依赖的免疫响应基因」。任务书 §5.3 把原始的两步条件
  （对照中上调 + TBK1 中被阻断）压缩成一步，丢掉了 TBK1 这个核心变量。现改回。

【模型设计订正 2026-09-25】—— 同一阈值（lfc>1）在池化 36 样本给出 0 个正类、
  在 24h 单独建模给出 13 个。差异来源是**池化把时变效应摊平**：
  池化模型估的是「恒定 rnai 效应」，且残差离散度被跨时间点变异灌大 → 功效反而更低。
  故：
    主分析   24h 单独建模，两臂并入（DNA+Mock，n=11），设计 ~rnai + treatment
             —— DNA vs Mock 已证零效应（outcome11 四重证据），并入不引入混杂，
                且 `treatment` 项仍把残余注射效应吸收掉，白捡功效
    一致性   3h / 6h 用同一设计各跑一次 → 用于「仅部分时间点显著」的排除判据
    敏感性   池化 36 样本，设计 ~rnai + timepoint + rnai:timepoint + treatment
             —— timepoint 参考水平设为 24h，故 rnai 主效应对应 24h，
                与主分析**同对比**、仅离散度估计来源不同。这才是有意义的敏感性对照。
             两个模型的输出都留档，不覆盖。

【标签规则】（用户 2026-09-25 定）
  正类：主对比中【显著上调】（TBK1 缺失后被放出）= 方向不得反
  负类：【完全无响应】——主对比与三个时间点 padj 全部 ≥ 0.5，
        且按基础表达量分层抽样匹配正类（工作文档-3 §2.1）
  排除：① 显著下调的 ② 仅部分时间点同向的 —— 均记录数量

产出
  results/deg_TBK1_vs_eGFP_24h.tsv          主分析（24h，n=11）
  results/deg_TBK1_vs_eGFP_{3h,6h}.tsv      一致性用
  results/deg_TBK1_vs_eGFP_pooledX.tsv      敏感性（池化 + 交互项）
  results/threshold_landscape.tsv           阈值景观
  results/label_positive_{strict,loose}.tsv 正类
  results/label_negative_{strict,loose}.tsv 负类（分层匹配抽样）
  results/label_excluded.tsv                排除基因 + 原因
  results/label_summary.tsv                 各集计数（含排除计数）
  results/qc_*.png  results/size_factors.tsv

用法: $IETX_ROOT/env/bin/python 09_tbk1_deseq.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
FEAT = f"{IETX}/features/features_fast.tsv"

# dataviz：散点/分组图类别色上限 3
C_UP, C_DN, C_NS = "#2a78d6", "#eb6834", "#c9c7c0"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
# ⚠️ DejaVu Sans 放第一位：当初 Droid Sans Fallback 在前，它没有 ASCII 数字字形，
#    matplotlib 报 "does not have a glyph for '1' [U+31]"，刻度标签全是豆腐块。
#    mpl ≥3.6 支持逐字形回退，故 DejaVu 出数字、Droid 出中文，两者互补。
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

PADJ = 0.05
LFC_STRICT, LFC_LOOSE = 1.0, 0.5
NS_PADJ = 0.5          # 「完全无响应」判据
BASE = ["eGFP", "TBK1"]
TIMEPOINTS = ["3h", "6h", "24h"]


def load():
    counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t").set_index("sample")
    # 不把 timepoint 转成 Categorical —— 早先那样做会把 intact 的 "intact" 变成 NaN，
    # pydeseq2 随即静默丢掉那 5 个样本(38→33)，再和外部 38 行的元数据对不上而报错。
    # 参考水平本来就对：formulaic 默认 treatment 编码按层名字典序，'24h'<'3h'<'6h'，
    # 故 24h 天然是参考水平，交互模型里的 rnai 主效应正好落在 24h。
    assert sorted(meta.timepoint.unique())[0] == "24h", "24h 不再是参考水平，交互模型解释要改"
    return counts, meta


def fit(counts, meta, sub, design, contrast, label, quiet=True):
    s = meta.loc[sub].copy()
    c = counts[list(s.index)]
    c = c.loc[c.sum(axis=1) > 0].T
    dds = DeseqDataSet(counts=c, metadata=s, design=design,
                       refit_cooks=True, quiet=quiet)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=quiet)
    st.summary()
    # ⚠️ 不 dropna(padj)：DESeq2 的独立过滤(informative covariates)会按表达量把
    #    低计数基因的 padj 置空，只留下「有希望显著」的高表达基因。若在此处丢掉它们，
    #    负类池就被限制在高表达基因里 —— 正是「高表达=正类」技术假信号的来源。
    #    负类判据改用 pvalue（对所有被检基因都有定义），显著性判据仍用 padj。
    r = st.results_df
    print(f"  [{label}] n={len(sub):>2d}  基因 {len(r):>6d}  "
          f"padj 非空 {r.padj.notna().sum():>6d}（独立过滤后）")
    return dds, r


def qc_plots(counts, meta, dds, tag):
    sf = pd.Series(dds.obs["size_factors"], name="size_factor")
    sf.to_frame().to_csv(f"{OUT}/size_factors.tsv", sep="\t")
    order = [s for s in meta.sort_values(["rnai", "treatment", "timepoint", "rep"]).index
             if s in sf.index]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.42), 3.6))
    ax.bar(x, sf.loc[order].values, color=C_UP, width=0.72)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=8, color=INK2)
    ax.set_ylabel("size factor", fontsize=9, color=INK2)
    ax.set_title(f"DESeq2 size factor（中位数比值法，{tag}）", fontsize=12, color=INK)
    ax.grid(axis="y", color="#e8e6e0", lw=0.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/qc_sizefactor.png", dpi=170); plt.close(fig)

    # RLE = log2(count/sizeFactor) - median_i(log2(count/sizeFactor))
    sub = counts[order]
    nz = sub[(sub > 0).all(axis=1)]
    norm = np.log2(nz.div(sf.loc[order], axis=1))
    rle = norm.sub(norm.median(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.42), 4.2))
    bp = ax.boxplot([rle[c].values for c in order], positions=x, widths=0.62,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2),
                    whiskerprops=dict(color=MUTED, lw=0.8),
                    capprops=dict(color=MUTED, lw=0.8))
    for b in bp["boxes"]:
        b.set(facecolor=C_UP, edgecolor=SURFACE, alpha=0.85, lw=1.0)
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=8, color=INK2)
    ax.set_ylabel("RLE", fontsize=9, color=INK2)
    ax.set_title("RLE 图 —— 各样本中位应≈0（未归一化残留会整体偏移）", fontsize=12, color=INK)
    ax.grid(axis="y", color="#e8e6e0", lw=0.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/qc_rle.png", dpi=170); plt.close(fig)
    return sf, rle


def ma_plot(res, title, fname):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    sig = (res.padj < PADJ) & (res.log2FoldChange.abs() > LFC_STRICT)
    up = sig & (res.log2FoldChange > 0); dn = sig & (res.log2FoldChange < 0)
    ax.scatter(res.baseMean[~sig], res.log2FoldChange[~sig], s=5, c=C_NS,
               alpha=0.45, lw=0, label="不显著", zorder=2)
    ax.scatter(res.baseMean[dn], res.log2FoldChange[dn], s=8, c=C_DN, alpha=0.8,
               lw=0, label=f"下调 ({dn.sum()})", zorder=3)
    ax.scatter(res.baseMean[up], res.log2FoldChange[up], s=8, c=C_UP, alpha=0.8,
               lw=0, label=f"上调 ({up.sum()})", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.9)
    for y in (LFC_STRICT, -LFC_STRICT): ax.axhline(y, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("baseMean（log 刻度）", fontsize=9, color=INK2)
    ax.set_ylabel("log2 fold change", fontsize=9, color=INK2)
    ax.set_title(title, fontsize=12, color=INK)
    ax.grid(color="#e8e6e0", lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, markerscale=2.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=170); plt.close(fig)


def strata_table(feats):
    """分层键 = 基础表达量十分位 × 基因长度三分位 × 转录本数档。
    三者都是 padj 的已知依赖项（DESeq2 独立过滤按表达量；长度/转录本数影响
    离散度与多重映射），不对齐的话模型会学到「高表达=正类」这个技术假信号。"""
    d = feats.copy()
    e = np.unique(np.quantile(d.expr.dropna(), np.linspace(0, 1, 11)))
    d["e_bin"] = pd.cut(d.expr, e, labels=False, include_lowest=True)
    d["l_bin"] = pd.qcut(d.gene_span.rank(method="first"), 3, labels=False)
    d["n_bin"] = pd.cut(d.n_transcripts, [0, 1, 2, 10 ** 6], labels=False)
    d["stratum"] = list(zip(d.e_bin, d.l_bin, d.n_bin))
    return d.set_index("gene_id")


def matched_negatives(pos, pool, d, rng):
    """在每个分层内按正类的需求量 1:1 抽样。返回 (基因, 欠填层数, 总层数)。
    欠填如实报告，不掩盖 —— 某些层负类候选不足本身就是一条结论。"""
    want = d.loc[[g for g in pos if g in d.index], "stratum"].value_counts()
    cand = d.loc[[g for g in pool if g in d.index]]
    pick, short = [], 0
    for st, need in want.items():
        c = cand.index[cand.stratum == st].tolist()
        if len(c) >= need:
            pick += list(rng.choice(c, need, replace=False))
        else:
            pick += c
            short += 1
    return pick, short, len(want)


def main():
    counts, meta = load()
    feats = pd.read_csv(FEAT, sep="\t")
    feats["expr"] = feats.gene_id.map(
        counts.div(counts.sum(axis=0), axis=1).mul(1e6).mean(axis=1))
    print("════ TBK1 vs eGFP 主标签分析（模型设计订正版）════")

    # ── 主分析：24h 单独建模，两臂并入 ──
    print("\n── 主分析：24h，~rnai + treatment，两臂并入 ──")
    s24 = meta.index[meta.timepoint == "24h"]
    print(f"  样本({len(s24)}): {sorted(s24)}")
    dds, res = fit(counts, meta, s24, "~rnai + treatment", ["rnai", "TBK1", "eGFP"], "24h")
    res.to_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t")

    sf, rle = qc_plots(counts, meta.loc[s24], dds, "24h 主分析")
    rle_med = rle.median()
    print(f"\n  size factor 中位 {sf.median():.4f}  范围 {sf.min():.3f}–{sf.max():.3f}")
    print(f"  RLE 中位范围 {rle_med.min():+.3f} ~ {rle_med.max():+.3f}  "
          f"(|max| = {rle_med.abs().max():.3f})")
    print(f"  → 判据 |RLE 中位| < 0.5：{'PASS' if rle_med.abs().max()<0.5 else 'FAIL'}")
    # 中位数恒≈0 是本构造的必然（每行减了行中位），真正该看的是【离散度】：
    # 盒体过高说明该样本的残差结构异常。故额外报 IQR。
    iqr = (rle.quantile(0.75) - rle.quantile(0.25))
    print(f"  RLE IQR 范围 {iqr.min():.3f} ~ {iqr.max():.3f}  中位 {iqr.median():.3f}"
          f"   → 判据 max/min < 3：{'PASS' if iqr.max()/iqr.min() < 3 else 'FAIL'}")
    ma_plot(res, "MA 图 —— TBK1(RNAi) vs eGFP(RNAi)，24h", "qc_ma.png")

    # ── 逐时间点（同设计）→ 一致性判据 ──
    print("\n── 逐时间点同设计（用于「仅部分时间点显著」排除）──")
    per_tp = {}
    for tp in TIMEPOINTS:
        sub = meta.index[meta.timepoint == tp]
        _, r = fit(counts, meta, sub, "~rnai + treatment", ["rnai", "TBK1", "eGFP"], tp)
        per_tp[tp] = r
        r.to_csv(f"{OUT}/deg_TBK1_vs_eGFP_{tp}.tsv", sep="\t")

    # ── 敏感性：池化 + rnai:timepoint 交互（timepoint 参考=24h）──
    print("\n── 敏感性对照：池化 36 样本 + rnai:timepoint 交互 ──")
    print("   （timepoint 参考水平 = 24h → 主效应即 24h 效应，与主分析同对比，仅离散度来源不同）")
    try:
        s_pool = [s for s in meta.index if meta.loc[s, "timepoint"] in TIMEPOINTS]
        _, res_pool = fit(counts, meta, s_pool,
                          "~rnai + timepoint + rnai:timepoint + treatment",
                          ["rnai", "TBK1", "eGFP"], "pooledX")
        res_pool.to_csv(f"{OUT}/deg_TBK1_vs_eGFP_pooledX.tsv", sep="\t")
        for padj in (0.01, 0.05):
            for lfc in (0.5, 1.0):
                a = set(res.index[(res.padj < padj) & (res.log2FoldChange > lfc)])
                b = set(res_pool.index[(res_pool.padj < padj) & (res_pool.log2FoldChange > lfc)])
                j = len(a & b) / max(len(a | b), 1)
                print(f"    padj<{padj} lfc>{lfc}: 主分析 {len(a):>3d}  池化+交互 {len(b):>3d}  "
                      f"交集 {len(a&b):>3d}  Jaccard={j:.3f}")
    except Exception as e:
        print(f"  ⚠️ 交互模型不可用（{type(e).__name__}: {e}）—— 留档于此，主分析不受影响")

    # ── 阈值景观 ──
    rows = []
    for padj in [0.01, 0.05, 0.10]:
        for lfc in [0.0, 0.5, 1.0, 1.5]:
            up = res.index[(res.padj < padj) & (res.log2FoldChange > lfc)]
            dn = res.index[(res.padj < padj) & (res.log2FoldChange < -lfc)]
            cons = [g for g in up if all(g in per_tp[t].index
                                         and per_tp[t].loc[g, "log2FoldChange"] > 0
                                         for t in per_tp)]
            rows.append({"padj": padj, "abs_log2FC": lfc, "上调": len(up),
                         "下调": len(dn), "上调且三时间点同向": len(cons)})
    tl = pd.DataFrame(rows)
    tl.to_csv(f"{OUT}/threshold_landscape.tsv", sep="\t", index=False)
    print("\n── 阈值景观 ──")
    print(tl.to_string(index=False))

    # ── 标签构造已移出本脚本 ──
    # 符号翻转 + 剔除靶基因 + 串联簇折叠 + 负类分层匹配都在 13_labels.py，
    # 理由：标签规则还会随阈值决策变动，不该每次都重跑 2.5 分钟的 DESeq2。
    print("\n── 标签构造见 13_labels.py（本脚本只负责 DE + 质控 + 阈值景观）──")


if __name__ == "__main__":
    main()
