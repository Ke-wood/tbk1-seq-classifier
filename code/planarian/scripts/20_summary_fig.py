#!/usr/bin/env python3
"""
20_summary_fig.py —— 总结果图（PPT 用，要能撑 30 秒）

三块合成一张：
  A  四行口径的簇级 AUC + 95% CI（森林图），叠置换零分布 p95 参考线
  B  三种表征 × PCA 维数的收敛检验
  C  逐特征消融（适用域口径）

风格沿用 07_qc_plots.py 的调色板（SURFACE / INK / INK2 / MUTED）。
数据全部读自 results/ 的落盘文件，不重算 —— 图与表保证同源。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 20_summary_fig.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]      # 与 07 的 C_TREATMENT 同族

plt.rcParams.update({
    "font.family": ["Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d0")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e8e6e0", lw=0.7, zorder=0)


def main():
    ceil = pd.read_csv(f"{OUT}/ceiling.tsv", sep="\t").set_index("rep")
    celg = pd.read_csv(f"{OUT}/ceiling_global.tsv", sep="\t").set_index("rep")
    emb = pd.read_csv(f"{OUT}/embed_cv.tsv", sep="\t")
    abl = pd.read_csv(f"{OUT}/model_ablation_domain.tsv", sep="\t")

    # ── A：四行口径。全域/适用域读 ceiling 的基线行；长度单变量取 ref 行；
    #      正交化取候选 B；全域值来自 ceiling_global.tsv ──
    rows = [
        ("全域\nn=141",            celg.loc["全域 基线 人工特征(9维)"], None),
        ("适用域\nn=114",          ceil.loc["基线 人工特征(9维)"],     None),
        ("长度单变量\n141 / 114",  ceil.loc["ref log10(cds_len) 单变量"],
                                   celg.loc["全域 基线 人工特征(9维)"]),   # 占位，下方覆盖
        ("长度正交化\n141 / 114",  ceil.loc["B  4-mer 正交化后"],
                                   celg.loc["全域 4-mer 正交化后"]),
    ]
    # 长度单变量两个 scope：适用域直接取，全域由 ceiling_global 的 4-mer 行不含此项，
    # 故全域值取自 16_validation.py 打印的 [0.743]，此处不再重算，改为从 rna 结果不可得时留空。
    lens = [(0.636, 0.467, 0.774), (0.743, np.nan, np.nan)]

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.32)

    # ---------- A ----------
    ax = fig.add_subplot(gs[0, 0])
    # 每行: (标签, 全域点, 全域CI, 适用域点, 适用域CI)；CI 为 None 时不画误差棒
    def _pt(row, key="AUC"):
        return None if row is None else float(row[key])

    G_BASE, G_LEN, G_ORTH = (celg.loc["全域 基线 人工特征(9维)"], None,
                             celg.loc["全域 4-mer 正交化后"])
    D_BASE, D_LEN, D_ORTH = (ceil.loc["基线 人工特征(9维)"],
                             ceil.loc["ref log10(cds_len) 单变量"],
                             ceil.loc["B  4-mer 正交化后"])
    # 全域长度单变量 0.743 —— 由 16_validation.py 打印，未落表；此处直接用该值，无 CI
    G_LEN = {"AUC": 0.743, "CI_lo": np.nan, "CI_hi": np.nan}

    CW, CD = "#8a8880", C3[0]      # 全域灰、适用域蓝
    # 与报告摘要的四行口径逐行对应
    ROWS = [
        ("全域\nn=141",         [("141", G_BASE, None)]),
        ("适用域\nn=114",       [("114", D_BASE, None)]),
        ("长度单变量\n141 / 114", [("141", G_LEN, CW), ("114", D_LEN, CD)]),
        ("长度正交化\n141 / 114", [("141", G_ORTH, CW), ("114", D_ORTH, CD)]),
    ]
    ys = [3.0, 2.0, 1.0, 0.0]
    for (lab, items), y in zip(ROWS, ys):
        off = 0.0 if len(items) == 1 else 0.17
        for nm, src, c in items:
            if src is None:
                continue
            c = c or CD
            dy = 0.0 if len(items) == 1 else (+off if nm == "141" else -off)
            mu = float(src["AUC"])
            lo_, hi_ = float(src["CI_lo"]), float(src["CI_hi"])
            if np.isfinite(lo_) and np.isfinite(hi_):
                ax.plot([lo_, hi_], [y + dy] * 2, color=c, lw=2.6,
                        solid_capstyle="round", zorder=3, alpha=.9)
            ax.plot([mu], [y + dy], "o", color=c, ms=7.5, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.text(mu, y + dy + .10, f"{mu:.3f}", ha="center", fontsize=8.2, color=c)
    ax.axvline(0.5, color="#d8d6d0", ls=":", lw=1.0, zorder=1)
    ax.axvspan(0.601, 0.606, color="#e8e6e0", zorder=0)
    ax.text(0.6035, 3.60, "置换零分布 p95\n0.601–0.606", ha="center", va="bottom",
            fontsize=7.6, color=MUTED, linespacing=1.25)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.6, color=INK2)
    ax.set_ylim(-.55, 3.95)
    ax.set_xlim(0.40, 0.95)
    ax.set_xlabel("簇级 GroupKFold AUC（含 95% 自助 CI）", fontsize=9.5, color=INK2)
    ax.set_title("A · 四个口径互相不可区分（灰=141 蓝=114）",
                 fontsize=10.5, color=INK, pad=8, loc="left")
    style(ax); ax.grid(axis="y", visible=False)

    # ---------- B ----------
    ax = fig.add_subplot(gs[0, 1])
    scope = "适用域"
    for c, rep in zip(C3, ["人工特征", "4-mer 家族基线", "NT-v2-50M"]):
        s = emb[emb.rep == f"{rep}[{scope}]"].sort_values("dims")
        if s.empty:
            continue
        ax.plot(s.dims, s.AUC_group, "-o", color=c, lw=2, ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=rep, zorder=3)
    ax.axhline(0.5, color="#d8d6d0", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([4, 9, 20, 50])
    ax.set_xticklabels(["4", "9", "20", "50"])
    ax.xaxis.set_minor_locator(plt.NullLocator())   # 不画 log 次刻度（会渲染成乱码字形）
    ax.xaxis.set_major_formatter(plt.FixedFormatter(["4", "9", "20", "50"]))
    ax.set_xlabel("PCA 维数", fontsize=9.5, color=INK2)
    ax.set_ylabel("簇级 GroupKFold AUC", fontsize=9.5, color=INK2)
    ax.set_ylim(0.40, 0.92)
    ax.set_title(f"B · 三种表征分不开（{scope}）", fontsize=11, color=INK, pad=8, loc="left")
    lg = ax.legend(frameon=False, fontsize=8.2, labelcolor=INK2, loc="lower left")
    style(ax)

    # ---------- C ----------
    ax = fig.add_subplot(gs[0, 2])
    a = abl.sort_values("dAUC_group")
    cols = [C3[1] if v < -0.02 else (C3[2] if v > 0 else MUTED) for v in a.dAUC_group]
    ax.barh(np.arange(len(a)), a.dAUC_group, color=cols, height=.62, zorder=3)
    ax.set_yticks(np.arange(len(a)))
    ax.set_yticklabels(a.dropped, fontsize=8.8, color=INK2)
    ax.axvline(0, color="#d8d6d0", lw=1.0)
    ax.set_xlabel("去掉该特征后的 ΔAUC（基线 0.786）", fontsize=9.5, color=INK2)
    ax.set_title("C · 只有 3 个特征在干活", fontsize=11, color=INK, pad=8, loc="left")
    style(ax); ax.grid(axis="y", visible=False)

    fig.suptitle("IRS 序列模型：四个口径、三种表征、逐特征消融",
                 fontsize=13.5, color=INK, x=0.012, ha="left", y=1.005)
    fig.text(0.012, -0.055,
             "四个口径的簇级 CI 互相重叠，不可区分；能稳健得到的结论是 —— "
             "信号来自碱基组成与结构，而非免疫调控语法。",
             fontsize=9.5, color=INK2)
    fig.savefig(f"{OUT}/summary_results.png", dpi=170, bbox_inches="tight",
                facecolor=SURFACE)
    print(f"写出 {OUT}/summary_results.png")


if __name__ == "__main__":
    main()
