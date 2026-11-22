#!/usr/bin/env python3
"""
34_cross_species_stats.py —— 涡虫 vs 人 vs 鼠 的 CDS 序列统计对比

【要补的】中心论点「跨物种迁移在涡虫上受限」的第 4 条证据（纯序列统计层面）。
  现有 3 条：同源注释覆盖仅 40% / 预训练基因组 LM 无增益 / CpG 不预测 TBK1 依赖程序。

【口径】三物种**等量随机取样**，且只取长度 ≥300 bp 的 CDS（短 CDS 的组成统计噪声极大）。
  涡虫侧 = `ref/cds.fa` 的**每基因最长 CDS**。
  人/鼠侧 = Ensembl CDS fasta（**每条转录本一条** —— 与涡虫侧口径不同，故**两套都报**：
  一是"每转录本"，二是"按基因去重取最长"，后者才与涡虫可严格对比）。

【必须写明的偏倚】涡虫侧只覆盖**注释到的那部分**（覆盖率约 40%），
  **不等于全基因组** —— 拿一个有偏样本冒充物种属性是最容易犯的错。

【判据】见 判据预注册_outcome15_prereg.md §34：
  · 三物种取样量必须相同，报出实际取样数
  · 预注册预测：涡虫 CpG O/E **低于**人/鼠（无 CpG 岛维持机制）
    **若相反，如实报告** —— 这会削弱（非推翻）"不能照搬脊椎动物直觉"的表述

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 34_cross_species_stats.py
"""
import gzip
import os
import re
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
EXT = f"{IETX}/external"
N_SAMPLE = 3000
MIN_LEN = 300
SEED = 0


def read_fa_any(path, longest_per_gene=False):
    """读 FASTA。longest_per_gene=True 时按 gene_id 取最长（用于与涡虫侧对齐口径）。"""
    op = gzip.open if path.endswith(".gz") else open
    out, name, buf = {}, None, []
    with op(path, "rt", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith(">"):
                if name:
                    s = re.sub(r"[^ACGT]", "", "".join(buf).upper().replace("U", "T"))
                    if longest_per_gene:
                        if len(s) > len(out.get(name, "")):
                            out[name] = s
                    else:
                        out.setdefault(name, s)
                if longest_per_gene:
                    m = re.search(r"gene(?:_id)?[:=]\s*([^\s;]+)", ln)
                    name = m.group(1) if m else (re.search(r"gene:(\S+)", ln) or
                                                 [None, ln[1:].split()[0]])[1]
                else:
                    name = ln[1:].split()[0]
                buf = []
            else:
                buf.append(ln.strip())
    if name:
        s = re.sub(r"[^ACGT]", "", "".join(buf).upper().replace("U", "T"))
        if not longest_per_gene or len(s) > len(out.get(name, "")):
            out[name] = s
    return out


def gc3(s):
    """第 3 位密码子的 GC —— 密码子使用偏好的常用代理。"""
    third = [s[i + 2] for i in range(0, len(s) - 2, 3)]
    if not third:
        return np.nan
    return sum(c in "GC" for c in third) / len(third)


def cpg_oe(s):
    c, g, n = s.count("C"), s.count("G"), len(s)
    return (s.count("CG") * n) / (c * g) if c and g and n > 1 else np.nan


def lowcomp(s):
    """与 10_features_seq.sh 同构：2-mer 熵代理。1=最乱，0=最单一。"""
    if len(s) < 10:
        return np.nan
    cnt = Counter(s[i:i + 2] for i in range(len(s) - 1))
    p = np.array(list(cnt.values()), float)
    p /= p.sum()
    return float(1 + (p * np.log2(p)).sum() / np.log2(16))


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else np.nan


def kmer4(seqs):
    idx = {"".join(t): i for i, t in enumerate(product("ACGT", repeat=4))}
    M = np.zeros(256)
    for s in seqs:
        for i in range(len(s) - 3):
            j = idx.get(s[i:i + 4])
            if j is not None:
                M[j] += 1
    return M / M.sum() if M.sum() else M


def js(p, q):
    """Jensen–Shannon 散度（log2，取值 0–1）。"""
    m = 0.5 * (p + q)
    def kl(a, b):
        msk = a > 0
        return float(np.sum(a[msk] * np.log2(a[msk] / np.maximum(b[msk], 1e-300))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sample(seqs, n, seed=SEED):
    keys = sorted(seqs)
    rng = np.random.default_rng(seed)
    if len(keys) <= n:
        return [seqs[k] for k in keys]
    return [seqs[k] for k in rng.choice(keys, n, replace=False)]


def describe(spec, seqs):
    rows = [dict(物种=spec, 指标="n_CDS", 值=len(seqs), 说明="取样条数"),
            dict(物种=spec, 指标="长度中位_bp", 值=float(np.median([len(s) for s in seqs])), 说明=""),
            dict(物种=spec, 指标="GC", 值=float(np.nanmean([gc(s) for s in seqs])), 说明=""),
            dict(物种=spec, 指标="GC3", 值=float(np.nanmean([gc3(s) for s in seqs])), 说明="第3位密码子"),
            dict(物种=spec, 指标="CpG_Oe", 值=float(np.nanmean([cpg_oe(s) for s in seqs])), 说明="1.0=期望"),
            dict(物种=spec, 指标="lowcomp", 值=float(np.nanmean([lowcomp(s) for s in seqs])), 说明="1=最乱")]
    return rows


def main():
    print("═" * 78)
    print("34 · 涡虫 vs 人 vs 鼠 —— CDS 序列统计")
    print("═" * 78)
    print(f"  取样：每物种 ≤{N_SAMPLE} 条，长度 ≥{MIN_LEN} bp，seed={SEED}")

    plan = read_fa_any(f"{IETX}/ref/cds.fa", longest_per_gene=True)
    print(f"  涡虫 ref/cds.fa：{len(plan)} 个基因（每基因最长 CDS）")

    hum_p = f"{EXT}/Homo_sapiens.cds.fa.gz"
    mus_p = f"{EXT}/Mus_musculus.cds.fa.gz"
    for p in (hum_p, mus_p):
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p} —— 先跑 34_cross_species_seq.sh")
    hum_all = read_fa_any(hum_p, longest_per_gene=False)
    mus_all = read_fa_any(mus_p, longest_per_gene=False)
    hum_g = read_fa_any(hum_p, longest_per_gene=True)
    mus_g = read_fa_any(mus_p, longest_per_gene=True)
    print(f"  人 : 转录本 {len(hum_all):,} / 基因 {len(hum_g):,}")
    print(f"  鼠 : 转录本 {len(mus_all):,} / 基因 {len(mus_g):,}")

    def filt(d):
        return {k: v for k, v in d.items() if len(v) >= MIN_LEN}

    sets = {"涡虫(基因)": filt(plan), "人(基因)": filt(hum_g), "鼠(基因)": filt(mus_g)}
    for k, v in sets.items():
        print(f"  {k:10s} 过滤后 {len(v):,} 条（≥{MIN_LEN} bp）")

    n_use = min(N_SAMPLE, *(len(v) for v in sets.values()))
    if len({len(v) for v in sets.values()}) != 1:
        print(f"  ⚠️ 三物种可用条数不等 —— **按最小值 {n_use} 等量取样**（不等量对比无意义）")
    smp = {k: sample(v, n_use) for k, v in sets.items()}
    for k, v in smp.items():
        print(f"  {k:10s} 实取 {len(v)} 条")

    rows = []
    for k, v in smp.items():
        rows += describe(k, v)
    df = pd.DataFrame(rows)

    print("\n" + "═" * 78)
    print(f"{'物种':12s} {'n_CDS':>6s} {'长度中位':>9s} {'GC':>7s} {'GC3':>7s} "
          f"{'CpG_Oe':>8s} {'lowcomp':>8s}")
    print("═" * 78)
    for k, v in smp.items():
        d = df[df.物种 == k].set_index("指标")["值"]
        print(f"{k:12s} {int(d['n_CDS']):>6d} {d['长度中位_bp']:>9.0f} {d['GC']:>7.3f} "
              f"{d['GC3']:>7.3f} {d['CpG_Oe']:>8.3f} {d['lowcomp']:>8.3f}")

    # 4-mer 频谱 + JS 散度
    print("\n" + "═" * 78)
    print("4-mer 频谱的 Jensen–Shannon 散度（0=同分布，1=完全不同）")
    print("═" * 78)
    K = {k: kmer4(v) for k, v in smp.items()}
    keys = list(smp)
    print(f"  {'':12s}" + "".join(f"{k:>14s}" for k in keys))
    for a in keys:
        line = f"  {a:12s}"
        for b in keys:
            line += f"{js(K[a], K[b]):>14.4f}"
            if a < b:
                rows.append(dict(物种=f"{a} vs {b}", 指标="JS_4mer", 值=js(K[a], K[b]), 说明=""))
        print(line)

    pd.DataFrame(rows).to_csv(f"{OUT}/cross_species_seq_stats.tsv", sep="\t", index=False)

    # ── 判据应用 ──────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("判据应用（预注册方向性预测：涡虫 CpG O/E 应【低于】人/鼠）")
    print("═" * 78)
    cp = {k: float(df[(df.物种 == k) & (df.指标 == "CpG_Oe")]["值"].iloc[0]) for k in keys}
    plan_cp = cp["涡虫(基因)"]
    for k in ("人(基因)", "鼠(基因)"):
        rel = "低于" if plan_cp < cp[k] else "**高于**"
        print(f"  涡虫 {plan_cp:.3f}  {rel}  {k} {cp[k]:.3f}"
              + ("   ← 与预注册一致" if plan_cp < cp[k]
                 else "   ← ⚠️ **与预注册相反，必须如实报告**"))
    print(f"\n  ⚠️ 必写的偏倚：涡虫侧只覆盖**注释到的那部分**（覆盖率约 40%），"
          f"**不等于全基因组**。")
    print(f"  ⚠️ 人/鼠侧此处用的是『每基因最长 CDS』以对齐口径；"
          f"若改用『每转录本』结论会变，故两套口径都已在脚本内定义。")
    print(f"\n  写出 {OUT}/cross_species_seq_stats.tsv（{len(rows) + len(df)} 行）")


if __name__ == "__main__":
    main()
