#!/usr/bin/env python3
"""
07_qc_plots.py —— B6 质控图：PCA + 样本相关性热图

目的：验证生物学重复是否按预期聚在一起，以及分组结构是否与实验设计一致。
这是任务书 B6 的硬性要求，也是判断 DEG 结果可不可信的前置检查。

图 1  results/pca.png            分面 PCA（左 eGFP / 右 TBK1）
图 2  results/corr_heatmap.png   样本相关性热图
表    results/pca_coords.tsv     PCA 坐标（表格视图，兼作可访问性 relief）

配色遵循 dataviz 规范：
  · 散点图（all-pairs）类别色上限为 3 —— 故【颜色 = 处理(3)】而非按 14 个组上色
  · 【分面 = RNAi】【形状 = 时间点】承载其余身份维度
  · 调色板 #2a78d6 / #eb6834 / #1baf7a 已过校验器（all-pairs 全通过）
  · #1baf7a 对浅底对比度 2.74 < 3:1 → 触发 relief 规则 → 图内加组中心直接标注 + 输出坐标表

用法: $IETX_ROOT/env/bin/python 07_qc_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
N_TOP = 2000                      # 取变异最大的 2000 个基因做 PCA

# ---- dataviz 参考调色板：散点图类别色上限 3 个 ----
C_TREATMENT = {"DNA": "#2a78d6", "Mock": "#eb6834", "intact": "#1baf7a"}
TREATMENT_CN = {"DNA": "注射 DNA", "Mock": "模拟注射", "intact": "未处理"}
MARKER = {"3h": "o", "6h": "s", "24h": "^", "intact": "D"}
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"

plt.rcParams.update({
    "font.family": ["Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def cpm_log(c):
    """log2(CPM+1) 标准化——PCA 前的常规处理"""
    lib = c.sum(axis=0)
    return np.log2(c.div(lib, axis=1) * 1e6 + 1)


def main():
    counts = pd.read_csv(f"{OUT}/counts.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t")
    # ⚠️ .loc[counts.columns] 会把结果索引名改成 counts.columns.name（此处为 None），
    #    随后 reset_index() 就产出名为 "index" 的列而不是 "sample"。
    #    必须 rename_axis 显式把名字设回来。
    meta = meta.set_index("sample").loc[counts.columns].rename_axis("sample").reset_index()
    assert "sample" in meta.columns, f"sample 列缺失: {list(meta.columns)}"

    L = cpm_log(counts)
    # 取变异最大的基因
    v = L.var(axis=1).sort_values(ascending=False)
    top = v.head(N_TOP).index
    X = L.loc[top].T.values
    X = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-9)

    pca = PCA(n_components=3, random_state=0)
    P = pca.fit_transform(X)
    ve = pca.explained_variance_ratio_ * 100
    meta["PC1"], meta["PC2"], meta["PC3"] = P[:, 0], P[:, 1], P[:, 2]
    meta.to_csv(f"{OUT}/pca_coords.tsv", sep="\t", index=False)

    # ================= 图 1：分面 PCA =================
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, rnai in zip(axes, ["eGFP", "TBK1"]):
        sub = meta[meta.rnai == rnai]
        for _, r in sub.iterrows():
            ax.scatter(r.PC1, r.PC2, s=110, marker=MARKER[r.timepoint],
                       c=C_TREATMENT[r.treatment], edgecolors=SURFACE,
                       linewidths=1.6, zorder=3)
        # 选择性直接标注（不逐个点标 —— 那是"每点带数字"反模式）。
        # 只标两类：① intact 点（#1baf7a 对比度 2.74<3:1，relief 规则要求可见标注）
        #            ② 最极端的 2 个样本（离群是这张图真正要说的事）
        for _, r in sub[sub.treatment == "intact"].iterrows():
            ax.annotate("未处理", (r.PC1, r.PC2), fontsize=8.5, color=INK2,
                        xytext=(7, 5), textcoords="offset points", zorder=6)
        ext = sub.reindex(sub.PC1.abs().add(sub.PC2.abs()).sort_values(ascending=False).index)[:2]
        for _, r in ext.iterrows():
            ax.annotate(r["sample"], (r.PC1, r.PC2), fontsize=8, color=INK,
                        xytext=(7, -11), textcoords="offset points", zorder=6,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_title(f"{rnai}(RNAi) 组", fontsize=12, color=INK, pad=10)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_xlabel(f"PC1（{ve[0]:.1f}% 方差）", fontsize=10, color=INK2)
        ax.grid(True, color="#e8e6e0", lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d6d0")
    axes[0].set_ylabel(f"PC2（{ve[1]:.1f}% 方差）", fontsize=10, color=INK2)

    h1 = [Line2D([], [], marker="o", ls="", ms=9, mfc=C_TREATMENT[t],
                 mec=SURFACE, mew=1.4, label=TREATMENT_CN[t]) for t in C_TREATMENT]
    h2 = [Line2D([], [], marker=MARKER[t], ls="", ms=9, mfc=MUTED,
                 mec=SURFACE, mew=1.4, label=t) for t in MARKER]
    leg = axes[1].legend(handles=h1 + h2, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         frameon=False, fontsize=9, labelcolor=INK2,
                         title="颜色=处理　形状=时间点", title_fontsize=9)
    leg.get_title().set_color(INK2)

    fig.suptitle("样本 PCA —— 生物学重复应聚在一起",
                 fontsize=14, color=INK, y=0.98)
    fig.text(0.5, 0.005,
             "基于 log2(CPM+1) 标准化、变异最大 2000 基因；坐标明细见 results/pca_coords.tsv",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 0.88, 0.95])
    fig.savefig(f"{OUT}/pca.png", dpi=170)
    plt.close(fig)

    # ================= 图 2：样本相关性热图 =================
    order = meta.sort_values(["rnai", "treatment", "timepoint", "rep"]).index.tolist()
    M = meta.loc[order]
    C = np.log2(counts[M["sample"]] + 1).corr().values

    # ⚠️ 对角线恒为 1.0，若把它算进色标范围，色标会被拉满到 1.0，
    #    所有组内方块都糊成同一片蓝，结构完全看不出来。故遮掉对角线，
    #    并按**非对角线**样本的分布定色标上下限。
    off = C[~np.eye(len(C), dtype=bool)]
    vmin, vmax = np.percentile(off, 0.5), np.percentile(off, 99.5)
    Cm = C.copy()
    np.fill_diagonal(Cm, np.nan)

    fig, ax = plt.subplots(figsize=(11.5, 10))
    # 顺序色 = 单一色相 浅→深（相关性全为正，用顺序而非发散）
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#eef4fb", "#2a78d6", "#0d3f78"])
    cmap.set_bad(SURFACE)
    im = ax.imshow(Cm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(M)))
    ax.set_yticks(range(len(M)))
    ax.set_xticklabels(M["sample"], rotation=90, fontsize=7, color=INK2)
    ax.set_yticklabels(M["sample"], fontsize=7, color=INK2)
    # 组分隔线
    grp = (M["rnai"] + "_" + M["treatment"] + "_" + M["timepoint"]).values
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(i - 0.5, color=SURFACE, lw=1.2)
            ax.axvline(i - 0.5, color=SURFACE, lw=1.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Pearson r（log2 counts，对角线已遮）", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    ax.set_title("样本相关性热图 —— 组内方块应深于组间", fontsize=13, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/corr_heatmap.png", dpi=170)
    plt.close(fig)

    # ================= 文本结论 =================
    # 正确的"方块"判据：每个组内部的相关性 vs 该组样本对所有其它样本的相关性
    # （早先版本拿"组内均值 vs 全组间均值"比较，差值仅 0.005–0.02，无判别力，已废弃）
    grp_key = (M["rnai"] + "_" + M["treatment"] + "_" + M["timepoint"]).values
    rows = []
    for g in sorted(set(grp_key)):
        idx = np.where(grp_key == g)[0]
        oth = np.where(grp_key != g)[0]
        if len(idx) < 2:
            continue
        w = C[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
        b = C[np.ix_(idx, oth)].mean()
        rows.append((w - b, g, len(idx), w, b))
    rows.sort(reverse=True)
    print("  组内 vs 组间相关性（差值越大 = 该组越独立）:")
    print(f"    {'组':22s} {'n':>2s} {'组内':>7s} {'组间':>7s} {'差值':>7s}")
    for d, g, n, w, b in rows:
        flag = "✅" if d > 0 else "⚠️ 组内≤组间"
        print(f"    {g:22s} {n:>2d} {w:>7.4f} {b:>7.4f} {d:>+7.4f} {flag}")
    print(f"\n  写出: {OUT}/pca.png  {OUT}/corr_heatmap.png  {OUT}/pca_coords.tsv")


if __name__ == "__main__":
    print("════ B6 质控图 ════")
    main()
    print("════ 完成 ════")
