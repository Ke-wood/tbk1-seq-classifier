#!/usr/bin/env python3
"""
31_power_analysis.py —— 附件3 的功效与样本量（补一个空白）

【要补的缺口】附件3 §四 给了判定阈值（≥1.5×、≥2×），但**全文没有样本量/功效依据**。
  判定阈值 ≠ 可执行的实验设计：不知道「要几只虫」，阈值就落不了地。

【输入】Drees 2026 报告的去阻遏倍数（平均 ≈3.2×、个体最高 ≈10×）。
  ⚠️ **组内变异系数 CV 在文献里没有给** ⇒ 本脚本把 CV **当假定扫一遍**，
     一个字都不许把某个 CV 写成事实。

【模型】发光读数近似对数正态。组内 CV → 对数尺度 SD: σ = sqrt(ln(1+CV²))。
  **检验定死：对 log(RLU) 做两样本 Welch t 检验**（这类读数的标准做法；
  直接对原始 RLU 做 t 检验会因右偏而失真，本项目不用那一套）。
  两样本、等样本量、α 双侧、power=1−β：
      n/组 = 2 (t_{1−α/2, df} + t_{1−β, df})² σ² / (ln f)²,  df = 2n−2
  **df 依赖 n，故迭代求解**（用 z 分位数会给小样本下偏乐观的 n ——
  实测 n=3 时 z 式声称 power 0.80，蒙特卡洛只有 0.39）。
  每个 n 都用 **蒙特卡洛**（对数正态抽样 + log 尺度 Welch t 检验）复核，
  偏离 >0.05 即打印告警 —— 解析式不许在没被验证的情况下交付。

【判据】见 判据预注册_outcome15_prereg.md §31。要点：
  · 只报「在 CV=x 的假定下…」，不报任何单一数字
  · 禁止把某个 CV 当成事实写进附件3 而不标注"假定"
  · 若所需 n 超出可行规模 → 明写「该终点在本方案规模下无功效」

用法: $IETX_ROOT/env/bin/python 31_power_analysis.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist, ttest_ind

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"

ALPHA = 0.05          # 双侧
POWER = 0.80
N_PRED = 6            # 附件3 的预期条目数 P1–P6，用于 Bonferroni 备选口径
FOLDS = [1.5, 2.0, 3.2, 10.0]          # 1.5/2.0 = 附件3 的判定阈值；3.2 = Drees 平均值；10 = 个体最高
CVS = [0.3, 0.5, 0.8, 1.0, 1.5]        # ⚠️ 全部是【假定】，文献未给
N_SWEEP = [6, 8, 10, 12, 16, 20, 24]   # 反向问：给定 n，最小可检出倍数是多少
N_MC = 4000
SEED = 0


def n_per_group(fold, cv, alpha=ALPHA, power=POWER, iters=200):
    """两样本等样本量的解析解（对数尺度，**t 分位数、df 迭代**）。"""
    if fold <= 1:
        return np.inf
    sigma = np.sqrt(np.log(1 + cv ** 2))
    d = np.log(fold)
    # 初值：z 近似（小样本下偏乐观，故必须迭代到 t 解）
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n = 2 * (z ** 2) * (sigma ** 2) / (d ** 2)
    for _ in range(iters):
        df = max(2 * n - 2, 1.0)
        tt = tdist.ppf(1 - alpha / 2, df) + tdist.ppf(power, df)
        n_new = 2 * (tt ** 2) * (sigma ** 2) / (d ** 2)
        if abs(n_new - n) < 1e-10:
            n = n_new
            break
        n = n_new
    return n


def mc_power(n, fold, cv, alpha=ALPHA, n_mc=N_MC, seed=SEED):
    """蒙特卡洛复核：**对 log(RLU) 做 Welch t 检验**（与解析式同一个检验）。"""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = -0.5 * sigma ** 2                      # 使中位数 = 1
    a = rng.lognormal(mu, sigma, size=(n_mc, n))
    b = rng.lognormal(mu + np.log(fold), sigma, size=(n_mc, n))
    p = ttest_ind(np.log(a), np.log(b), axis=1, equal_var=False).pvalue
    return float((p < alpha).mean())


def mde(n, cv, alpha=ALPHA, power=POWER):
    """反向：给定 n/组，可检出的最小倍数。"""
    sigma = np.sqrt(np.log(1 + cv ** 2))
    tt = tdist.ppf(1 - alpha / 2, max(2 * n - 2, 1)) + tdist.ppf(power, max(2 * n - 2, 1))
    return float(np.exp(tt * sigma * np.sqrt(2 / n)))


def main():
    print("═" * 78)
    print("31 · 功效与样本量（附件3 §四之补）")
    print("═" * 78)
    print(f"  α={ALPHA}（双侧）  power={POWER}  两样本等样本量  "
          f"Bonferroni 备选 α/{N_PRED}={ALPHA / N_PRED:.4f}")
    print("  检验定死：**对 log(RLU) 做 Welch t 检验**（不是对原始 RLU）")
    print("  ⚠️ **组内 CV 文献未给** —— 下表全部是【假定】下的结果，不得写成事实")

    rows = []
    print(f"\n{'═' * 78}\n正向：检出给定倍数所需的 n/组\n{'═' * 78}")
    head = f"  {'倍数':>6s}" + "".join(f"{'CV=' + str(c):>10s}" for c in CVS) \
           + f"{'（α/' + str(N_PRED) + '）':>12s}"
    print(head)
    for f in FOLDS:
        line = f"  {f:>6.1f}"
        for cv in CVS:
            n = n_per_group(f, cv)
            line += f"{int(np.ceil(n)):>10d}"
        n_bonf = n_per_group(f, CVS[1], alpha=ALPHA / N_PRED)   # 仅对中间档 CV 给一行备选
        line += f"{int(np.ceil(n_bonf)):>12d}"
        print(line)
        for cv in CVS:
            n = int(np.ceil(n_per_group(f, cv)))
            rows.append(dict(方向="正向", 倍数=f, CV假定=cv,
                             sigma_log=round(np.sqrt(np.log(1 + cv ** 2)), 4),
                             n_per_group=n, 解析功效=POWER,
                             MC实测功效=round(mc_power(n, f, cv), 3)))

    print(f"\n{'═' * 78}\n反向：给定 n/组，最小可检出倍数（实验者实际要用的那张）\n{'═' * 78}")
    print(f"  {'n/组':>6s}" + "".join(f"{'CV=' + str(c):>10s}" for c in CVS))
    for n in N_SWEEP:
        line = f"  {n:>6d}"
        for cv in CVS:
            m = mde(n, cv)
            line += f"{m:>10.2f}"
            rows.append(dict(方向="反向", n_per_group=n, CV假定=cv,
                             sigma_log=round(np.sqrt(np.log(1 + cv ** 2)), 4),
                             最小可检出倍数=round(m, 3),
                             MC实测功效=round(mc_power(n, m, cv), 3)))
        print(line)

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/power_analysis.tsv", sep="\t", index=False)

    # ── 解析式 vs 蒙特卡洛 ────────────────────────────────────────────────
    print(f"\n{'═' * 78}\n解析式自检（解析式声称 power={POWER}，MC 实测是否吻合）\n{'═' * 78}")
    chk = r.dropna(subset=["MC实测功效"])
    chk = chk[chk.方向 == "正向"]          # 反向那半张是"给定 n 求倍数"，不该期望 power=0.80
    chk["偏离"] = (chk["MC实测功效"] - POWER).abs()
    # ⚠️ n<5 时 t 近似本身失效（n=1 时 Welch t 无定义），不能算"解析式错"。
    #    分开报，避免用一堆 n=1 的格子淹没真正需要看的偏离。
    small, big = chk[chk.n_per_group < 5], chk[chk.n_per_group >= 5]
    bad = big[big.偏离 > 0.05]
    print(f"  n/组 ≥5 的 {len(big)} 个格中，MC 实测功效与 {POWER} 偏离 >0.05 的有 {len(bad)} 个")
    if len(bad):
        print(bad[["倍数", "CV假定", "n_per_group", "MC实测功效"]].to_string(index=False))
    print(f"  n/组 <5 的 {len(small)} 个格（t 近似失效区，**不据此判定解析式对错**）："
          f"MC 实测功效 "
          + ", ".join(f"n={int(q.n_per_group)}:{q['MC实测功效']:.2f}" for _, q in small.iterrows()))
    if not len(bad):
        print("  ✓ 可用区解析式与蒙特卡洛一致，可交付")
    else:
        print("  ⚠️ 可用区仍有偏离 —— **报告时以 MC 实测功效为准**，并注明。")

    # ── 判据应用 ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 78}\n判据应用\n{'═' * 78}")
    b = r[(r.方向 == "正向") & (r.倍数 == 2.0)]
    print("  附件3 判据 P2（TBK1(RNAi) 相对对照 ≥2 倍）所需 n/组：")
    for _, q in b.iterrows():
        print(f"    假定 CV={q.CV假定:<4} → n/组 = {q.n_per_group:>3d}"
              f"（MC 实测功效 {q.MC实测功效:.2f}）")
    print("\n  ⇒ 附件3 §四须补一列「所需 n/组（假定 CV=x）」；"
          "**必须写明 CV 是假定**，并给出这张敏感性表。")
    print("  ⇒ 若某一档所需 n 超出涡虫实验可行规模，明写「该终点在本方案规模下无功效」。")
    print(f"\n  写出 {OUT}/power_analysis.tsv（{len(r)} 行）")


if __name__ == "__main__":
    main()
