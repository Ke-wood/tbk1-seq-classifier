#!/usr/bin/env python3
"""
11_label_validation.py —— 标签富集检验（符号翻转版）

【符号已翻转，用户 2026-09-25 确认】正类 = TBK1(RNAi) 中显著【下调】的基因。
  理由（写进交付文档）：
    ① TBK1 是激活型激酶（磷酸化 IRF3/IRF7），敲降 → 下游降，规范生物学
    ② TRAF 家族级 5/54 全部同向 p=0.0013；TBK1/IKK 轴 p=0.015 —— 通路级假设驱动检验
    ③ 工作文档-3 §2.1 原措辞「在 TBK1(RNAi) 中被阻断」本就读作 log2FC 为负
  判据是【通路级检验】，不是 cherry-picked 基因 —— PGRP 落在上调侧(+0.446)、
  AMP3632 两侧都无，点名基因不作判据。

本脚本落实用户要求的六件事：
  1. 从正类剔除 TBK1 自身（敲降靶基因，循环论证）与 eGFP
  2. ISG / 黑色化的反向信号如实报告 + best-hit E-value
  3. 下调集是否富集核糖体/翻译/管家基因（84 vs 61 不对称 → 排除「全局病态」）
  4. 富集前折叠串联重复簇（同 scaffold <100 kb 同家族 → 一个位点）
  5. Fisher universe = 实际检验基因数
  6. 列出 TRAF 命中的 best-hit 名称 + E-value（TRAF3 在 TBK1 上游，需排除注释错配）

用法: $IETX_ROOT/env/bin/python 11_label_validation.py
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT = f"{IETX}/results"
CLUSTER_KB = 100          # 串联重复簇折叠阈值
TARGET = "h1SMcG0013953"  # TBK1 本体（最佳命中 IKKE_MOUSE），必须从正类剔除

# ── 免疫面板 ──
PANEL = {
    "TBK1/IKK 轴": [r"\bTBK1", r"I-kappa-B kinase", r"\bIKKE\b"],
    "IRF 家族": [r"interferon regulatory factor", r"\bIRF[0-9]"],
    "STING/cGAS": [r"stimulator of interferon", r"cyclic GMP-AMP", r"\bcGAS\b"],
    "NF-kB/Rel": [r"NF-kappa-B", r"nuclear factor NF-kappa", r"I-kappa-B"],
    "RIG-I/MAVS/MDA5": [r"RIG-I", r"\bMAVS", r"MDA5", r"interferon-induced helicase"],
    "JAK/STAT": [r"\bSTAT[0-9]", r"janus kinase", r"\bJAK[0-9]"],
    "Toll/MyD88": [r"toll-like", r"\bMyD88\b", r"interleukin-1 receptor"],
    "干扰素效应(ISG)": [r"interferon-induced", r"\bviperin\b", r"\bISG15",
                        r"2'-5'-oligoadenylate", r"protein kinase R",
                        r"interferon-induced gtp-binding"],
    "TRAF": [r"TNF receptor-associated factor"],
    "PGRP/肽聚糖": [r"peptidoglycan", r"n-acetylmuramoyl"],
    "抗菌肽": [r"antimicrobial", r"antibacterial", r"defensin", r"cecropin",
               r"attacin", r"diptericin", r"moricin", r"lysozyme"],
    "黑色化/酚氧化酶": [r"phenoloxidase", r"tyrosinase"],
    "补体/凝血": [r"\bcomplement\b", r"coagulation"],
    "清道夫受体": [r"scavenger receptor"],
    "凝集素": [r"\blectin"],
    "凋亡效应": [r"\bcaspase"],
}
# ── 管家/翻译面板：用于排除「全局病态」假信号（用户第 3 条）──
HOUSEKEEPING = {
    "核糖体蛋白": [r"ribosomal protein", r"\bRPS[0-9]", r"\bRPL[0-9]",
                   r"40S ribosomal", r"60S ribosomal"],
    "翻译因子": [r"elongation factor", r"initiation factor", r"peptide chain"],
    "能量代谢": [r"cytochrome c oxidase", r"NADH dehydrogenase", r"\bATP synthase",
                 r"glyceraldehyde-3-phosphate"],
    "细胞骨架": [r"\bactin\b", r"\btubulin\b", r"myosin"],
    "蛋白酶体/泛素": [r"proteasome", r"\bubiquitin\b"],
    "组蛋白/核酸包装": [r"histone", r"nucleosome"],
    "热激/应激": [r"heat shock", r"\bHSP[0-9]"],
}


def load():
    res = pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_24h.tsv", sep="\t", index_col=0)
    ann = pd.read_csv(f"{OUT}/gene_annotation.tsv", sep="\t")
    ann = ann.drop_duplicates("gene_id").set_index("gene_id")
    coords = pd.read_csv(f"{IETX}/features/genes.bed", sep="\t", header=None,
                         names=["chrom", "start", "end", "gene_id", "score", "strand"])
    coords = coords.set_index("gene_id")
    per_tp = {tp: pd.read_csv(f"{OUT}/deg_TBK1_vs_eGFP_{tp}.tsv", sep="\t",
                              index_col=0) for tp in ("3h", "6h", "24h")}
    return res, ann, coords, per_tp


def collapse_clusters(genes, coords, kb=CLUSTER_KB):
    """串联重复簇折叠：同 scaffold 上间隔 < kb 的同家族基因算【一个位点】。
    代表取 padj 最小者。返回 (折叠后集合, 簇数, 被折叠掉的基因数)。
    教训来自 Mx：h1SMcG0008629/30/31/32/33 在 chr1_h1 上跨 250 kb，
    是一个串联重复簇，当成 3 个独立命中会把 OR 抬到 45 —— 必须制度化。"""
    g = sorted((x for x in genes if x in coords.index),
               key=lambda x: (coords.loc[x, "chrom"], coords.loc[x, "start"]))
    reps, clusters, dropped = [], 0, 0
    i = 0
    while i < len(g):
        j = i + 1
        while j < len(g) and coords.loc[g[j], "chrom"] == coords.loc[g[i], "chrom"] \
                and coords.loc[g[j], "start"] - coords.loc[g[j - 1], "end"] < kb * 1000:
            j += 1
        grp = g[i:j]
        if len(grp) > 1:
            clusters += 1
            dropped += len(grp) - 1
        reps.append(min(grp, key=lambda x: res_padj.get(x, 1.0)))
        i = j
    return set(reps), clusters, dropped


def fisher(S, G, universe):
    a = len(S & G); b = len(S) - a; c = len(G) - a
    d = len(universe) - len(S) - c
    if a == 0 and c == 0:
        return a, np.nan, np.nan
    od, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, od, p


def main():
    global res_padj
    res, ann, coords, per_tp = load()
    res_padj = res.padj.fillna(1.0).to_dict()
    desc = ann["stitle"].fillna("")
    ev = ann["evalue"] if "evalue" in ann.columns else pd.Series(dtype=float)

    # 用户第 5 条：universe = 实际检验基因数
    universe = set(res.index)
    print(f"universe = 实际检验基因数 = {len(universe)}"
          f"（矩阵 26,695；24h 子集里 8 个基因全零被弃）")

    # ── 正类：显著【下调】──
    def pos_set(lfc):
        s = set(res.index[(res.padj < 0.05) & (res.log2FoldChange < -lfc)])
        cons = {g for g in s
                if all(g in per_tp[t].index and per_tp[t].loc[g, "log2FoldChange"] < 0
                       for t in per_tp)}
        return s, cons

    dn10, cons10 = pos_set(1.0)
    dn05, cons05 = pos_set(0.5)
    UP = set(res.index[(res.padj < 0.05) & (res.log2FoldChange > 0.5)])

    # 用户第 1 条：剔除 TBK1 自身
    print(f"\n【第1条】从正类剔除敲降靶基因 TBK1 = {TARGET}"
          f"（{desc.get(TARGET,'')[:60]}）")
    for nm, s in (("严格(lfc<-1.0)", dn10), ("宽松(lfc<-0.5)", dn05),
                  ("下滑全集(lfc<0,padj<0.05)", set(res.index[(res.padj<0.05)&(res.log2FoldChange<0)]))):
        print(f"  {nm:26s} 剔除前 {len(s):>3d} → 剔除后 {len(s-{TARGET}):>3d}"
              f"   （TBK1 {'在内' if TARGET in s else '不在内'}）")
    print(f"  注：eGFP 不在参考注释里（GFF3 中 0 条，SwissProt 唯一 gfp 命中是"
          f" GFPT2 谷氨酰胺果糖氨基转移酶，误匹配），无需剔除。")

    POS = {"strict(lfc<-1.0)": dn10 - {TARGET}, "loose(lfc<-0.5)": dn05 - {TARGET}}
    CONS = {"strict(lfc<-1.0)": cons10 - {TARGET}, "loose(lfc<-0.5)": cons05 - {TARGET}}
    for k in POS:
        print(f"  正类[{k}] {len(POS[k])} 个；其中三时间点同向 {len(CONS[k])} 个")
    pd.DataFrame([{"set": k, "n": len(v)} for k, v in POS.items()]).to_csv(
        f"{OUT}/label_pos_counts.tsv", sep="\t", index=False)

    # ── 决定性：折叠前后富集对比 ──
    print("\n════ 富集检验（用户第 4 条：折叠串联重复簇后重跑）════")
    rows = []
    for pname, PS in POS.items():
        print(f"\n  ── 正类 = {pname}（n={len(PS)}）──")
        print(f"    {'家族':18s} {'原始n':>5s} {'折叠n':>5s} {'命中':>4s} "
              f"{'簇':>3s} {'OR原始':>7s} {'p原始':>9s} {'OR折叠':>7s} {'p折叠':>9s}")
        for fam, pats in {**PANEL, **HOUSEKEEPING}.items():
            rx = re.compile("|".join(pats), re.I)
            G = set(desc.index[desc.str.contains(rx, regex=True, na=False)]) & universe
            if not G:
                continue
            Gc, ncl, ndrop = collapse_clusters(G, coords)
            a, o1, p1 = fisher(PS, G, universe)
            ac, o2, p2 = fisher(PS, Gc, universe)
            mark = " ★" if (p2 == p2 and p2 < 0.05) else ""
            print(f"    {fam:18s} {len(G):>5d} {len(Gc):>5d} {ac:>4d} {ncl:>3d} "
                  f"{o1:>7.2f} {p1:>9.3g} {o2:>7.2f} {p2:>9.3g}{mark}")
            rows.append(dict(pos_set=pname, family=fam, n_raw=len(G), n_collapsed=len(Gc),
                             hits=ac, clusters=ncl, dropped=ndrop,
                             OR_raw=o1, p_raw=p1, OR_collapsed=o2, p_collapsed=p2))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/label_enrichment_collapsed.tsv", sep="\t", index=False)

    # ── 方向检验 ──
    print("\n════ 方向检验：上调集 vs 下调集 ════")
    print(f"  下调集(padj<0.05,lfc<-0.5) n={len(dn05)}   上调集(padj<0.05,lfc>0.5) n={len(UP)}")
    drows = []
    for fam, pats in PANEL.items():
        rx = re.compile("|".join(pats), re.I)
        G = set(desc.index[desc.str.contains(rx, regex=True, na=False)]) & universe
        if not G:
            continue
        Gc, _, _ = collapse_clusters(G, coords)
        ad, od_, pd_ = fisher(dn05, Gc, universe)
        au, ou, pu = fisher(UP, Gc, universe)
        print(f"  {fam:18s} 折叠n={len(Gc):>3d}  下调 {ad:>2d} OR={od_:>6.2f} p={pd_:>8.3g}"
              f"   上调 {au:>2d} OR={ou:>6.2f} p={pu:>8.3g}")
        drows.append(dict(family=fam, n_collapsed=len(Gc), down=ad, OR_down=od_,
                          p_down=pd_, up=au, OR_up=ou, p_up=pu))
    pd.DataFrame(drows).to_csv(f"{OUT}/label_direction_test.tsv", sep="\t", index=False)

    # ── 用户第 6 条 + 第 2 条：逐命中细节 + E-value ──
    print("\n════ 用户第6条：TRAF 命中明细（TRAF3 在 TBK1 上游，需排除注释错配）════")
    rxt = re.compile("|".join(PANEL["TRAF"]), re.I)
    GT = set(desc.index[desc.str.contains(rxt, regex=True, na=False)]) & universe
    hit = sorted(GT & dn05, key=lambda g: res_padj.get(g, 1))
    print(f"  TRAF 家族折叠后 {collapse_clusters(GT, coords)[1]} 簇 / {len(collapse_clusters(GT, coords)[0])} 个代表；"
          f"下调节点 {len(hit)} 个")
    for g in hit:
        print(f"    {g}  lfc={res.loc[g,'log2FoldChange']:+.3f} padj={res.loc[g,'padj']:.3g} "
              f"E={ev.get(g, float('nan')):.3g}")
        print(f"        {desc.get(g,'')[:105]}")

    print("\n════ 用户第2条：反向信号（上调集）逐命中 + E-value ════")
    for fam in ("干扰素效应(ISG)", "黑色化/酚氧化酶"):
        rx = re.compile("|".join(PANEL[fam]), re.I)
        G = set(desc.index[desc.str.contains(rx, regex=True, na=False)]) & universe
        Gc, ncl, _ = collapse_clusters(G, coords)
        h = sorted(G & UP, key=lambda g: res_padj.get(g, 1))
        print(f"\n  {fam}: 折叠后 {len(Gc)} 个位点（{ncl} 簇），上调集中 {len(h)} 个命中")
        for g in h:
            print(f"    {g}  lfc={res.loc[g,'log2FoldChange']:+.3f} padj={res.loc[g,'padj']:.3g} "
                  f"E={ev.get(g, float('nan')):.3g}   {desc.get(g,'')[:80]}")


if __name__ == "__main__":
    res_padj = {}
    main()
