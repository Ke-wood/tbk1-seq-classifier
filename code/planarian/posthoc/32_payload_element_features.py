#!/usr/bin/env python3
"""
32_payload_element_features.py —— 候选载荷的元件级特征 + 在基因组分布中的分位

【为什么要做】附件1 §〇之二 说「IRS 在载荷上 6/9 个特征退化为常数」——
  但那句话现在是定性的。本脚本把它**具体化到元件级**：
  哪些特征恒定、为什么恒定（所有候选共用同一个 eF1α 启动子）、哪些真的在变。

【要补的第二件】现有表述里有「GC = 0.41」这类**没有参照系**的数字。
  本脚本给每个特征配上**参照分布**（prom / cds），把裸数字变成**分位**。

【口径】全部复用主流程的同一套函数与参数 ——
  特征定义来自 `10_features_seq.sh`（GC / CpG o/e / lowcomp 用 2-mer 熵代理）、
  CAI 来自 `22_payload.cai`、MFE 用 `RNAfold --noPS`（与 `10_features_seq.sh` 同参）。

【自检】载荷的启动子元件就是 h1SMcG0010823 上游 1 kb，
  故其四个启动子特征**必须与 features_fast.tsv 里该基因那一行相同**
  （GC / CpG o/e / lowcomp / MFE 全部是链不变的，反向互补不改变它们）。
  不符 → **显著告警并记入 flag 列，但不中止**（可能是上游窗口约定差 1 bp，
  属于"须核对"而非"必须停手"）。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 32_payload_element_features.py
"""
import os
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
PAY = f"{IETX}/payload"
PIPE = os.environ.get("IETX_SCRIPTS", f"{IETX}/scripts")
PROM_GENE = "h1SMcG0010823"          # eF1α 启动子来源基因


def _mod(name, path):
    """载入主流程脚本为模块。找不到时给一句人话，而不是抛原始 traceback。"""
    if not os.path.exists(path):
        raise SystemExit(f"!! 找不到主流程脚本 {path} —— 用 IETX_SCRIPTS 指定它的目录"
                         f"（默认 $IETX_ROOT/scripts）")
    sp = spec_from_file_location(name, path)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def lowcomp(s):
    """与 10_features_seq.sh 同构：2-mer 熵的粗糙代理。**1 = 最乱，0 = 最单一。**"""
    if len(s) < 10:
        return np.nan
    cnt = Counter(s[i:i + 2] for i in range(len(s) - 1))
    p = np.array(list(cnt.values()), float)
    p /= p.sum()
    return float(1 + (p * np.log2(p)).sum() / np.log2(16))


def read_plain_fa(path):
    seq, name, buf = {}, None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name:
                    seq[name] = "".join(buf).upper()
                name, buf = ln[1:].split()[0], []
            else:
                buf.append(ln.strip())
    if name:
        seq[name] = "".join(buf).upper()
    return seq


def pct(value, dist):
    d = np.asarray([x for x in dist if x == x], float)
    if not len(d) or value != value:
        return np.nan
    return float((d < value).mean() * 100)


def main():
    v22 = _mod("v22", os.path.join(PIPE, "22_payload.py"))
    v25 = _mod("v25", os.path.join(PIPE, "25_accessibility.py"))

    print("═" * 78)
    print("32 · 候选载荷元件级特征 + 分位")
    print("═" * 78)

    fast = pd.read_csv(f"{FEAT}/features_fast.tsv", sep="\t").set_index("gene_id")
    cdsf = pd.read_csv(f"{FEAT}/features_cds.tsv", sep="\t").set_index("gene_id")
    try:
        mfe = pd.read_csv(f"{FEAT}/mfe.tsv", sep="\t", header=None,
                          names=["gene_id", "mfe"]).set_index("gene_id")
        fast = fast.join(mfe, how="left")
        fast["mfe_per_nt"] = fast.mfe / fast.prom_len
        print(f"  MFE 参照分布：{fast.mfe_per_nt.notna().sum()} 条启动子"
              f"（来自 features/mfe.tsv）")
    except FileNotFoundError:
        fast["mfe_per_nt"] = np.nan
        print("  ⚠️ 无 features/mfe.tsv —— MFE 分位将缺失（须先跑 10_features_seq.sh rnafold）")

    cds = v22.longest_cds()
    _, freq, _, _ = v22.codon_table(cds)
    print(f"  密码子频率取自 {len(cds)} 个基因的 CDS（与 G4 优化同一张表）")

    cands = read_plain_fa(f"{PAY}/sequences/candidates_v1.fa")
    parts = pd.read_csv(f"{PAY}/parts_table.tsv", sep="\t")
    print(f"  载荷 {len(cands)} 条；元件行 {len(parts)} 行")

    rows = []
    for _, p in parts.iterrows():
        full = cands.get(p.候选)
        if full is None:
            print(f"  ✗ {p.候选} 不在 candidates_v1.fa 里 —— 跳过")
            continue
        s = full[int(p.起) - 1:int(p.止)]
        if len(s) != int(p.长度):
            print(f"  ⚠️ {p.候选}/{p.元件}: 取到 {len(s)} nt，表中写 {p.长度}"
                  f"（按表中坐标截取，口径以此为准）")
        is_prom = "promoter" in str(p.元件)
        dist_gc = fast.prom_gc if is_prom else cdsf.cds_gc
        dist_cp = fast.prom_cpg_oe if is_prom else cdsf.cds_cpg_oe
        dist_lc = fast.prom_lowcomp if is_prom else None
        dist_cai = None if is_prom else cdsf.cai
        gc_v = v22.gc(s)
        cp_v = v22.cpg_oe(s)
        lc_v = lowcomp(s)
        cai_v = np.nan if is_prom else v22.cai(s, freq)
        rows.append(dict(
            _seq=s,
            候选=p.候选, 元件=p.元件, 来源=p.来源, 长度=len(s),
            参照分布="prom(1 kb 上游)" if is_prom else "cds(最长 CDS)",
            GC=round(gc_v, 4), GC_分位=round(pct(gc_v, dist_gc), 1),
            CpG_oe=round(cp_v, 4), CpG_oe_分位=round(pct(cp_v, dist_cp), 1),
            lowcomp=round(lc_v, 4),
            lowcomp_分位=round(pct(lc_v, dist_lc), 1) if dist_lc is not None else np.nan,
            CAI=round(cai_v, 4) if cai_v == cai_v else np.nan,
            CAI_分位=round(pct(cai_v, dist_cai), 1) if dist_cai is not None else np.nan,
            flag=""))

    # ── 自检：启动子元件应等于 features_fast 里 h1SMcG0010823 那一行 ─────────
    if PROM_GENE in fast.index:
        f0 = fast.loc[PROM_GENE]
        for r in rows:
            if "promoter" not in str(r["元件"]):
                continue
            d = [abs(r["GC"] - f0.prom_gc), abs(r["CpG_oe"] - f0.prom_cpg_oe),
                 abs(r["lowcomp"] - f0.prom_lowcomp)]
            if max(d) > 1e-6:
                r["flag"] = "⚠️ 与 features_fast 的启动子行不符"
                print(f"\n  ⚠️ 自检不符：{r['候选']} 的启动子特征 ≠ {PROM_GENE} 在 "
                      f"features_fast.tsv 的行")
                print(f"     载荷 GC={r['GC']:.4f} CpG={r['CpG_oe']:.4f} lc={r['lowcomp']:.4f}")
                print(f"     参照 GC={f0.prom_gc:.4f} CpG={f0.prom_cpg_oe:.4f} "
                      f"lc={f0.prom_lowcomp:.4f}")
                print("     ⇒ 多半是上游窗口约定差 1 bp —— 记入 flag，**须核对**，不中止")
            else:
                r["flag"] = "✓ 与 features_fast 启动子行一致"
        print(f"  ✓ 启动子自检完成（结果见 flag 列）")
    else:
        print(f"  ⚠️ features_fast.tsv 里没有 {PROM_GENE} —— 启动子自检跳过")

    # ── MFE（RNAfold，与 10_features_seq.sh 同参：--noPS，默认参数）─────────
    try:
        recs = [(f"E{k}", r["_seq"].replace("T", "U")) for k, r in enumerate(rows)]
        folds = v25.run_rnafold(recs)
        for k, r in enumerate(rows):
            m, _ = folds.get(f"E{k}", (None, None))
            r["MFE_kcal"] = round(m, 2) if m is not None else np.nan
            r["MFE_per_nt"] = round(m / r["长度"], 4) if m is not None and r["长度"] else np.nan
            r["MFE_分位"] = round(pct(m / r["长度"], fast.mfe_per_nt), 1) \
                if m is not None and "promoter" in str(r["元件"]) else np.nan
        print(f"  ✓ RNAfold 解析 {sum(1 for k in range(len(rows)) if folds.get(f'E{k}', (None, None))[0] is not None)}"
              f"/{len(rows)} 条元件")
    except Exception as e:
        print(f"  ⚠️ RNAfold 跳过（{e}）—— MFE 列留空，不补零")
        for r in rows:
            r.setdefault("MFE_kcal", np.nan)
            r.setdefault("MFE_per_nt", np.nan)
            r.setdefault("MFE_分位", np.nan)

    out = pd.DataFrame(rows).drop(columns=["_seq"], errors="ignore")
    out.to_csv(f"{PAY}/element_features.tsv", sep="\t", index=False)

    print("\n" + "═" * 78)
    print("元件级特征表")
    print("═" * 78)
    show = ["候选", "元件", "长度", "GC", "GC_分位", "CpG_oe", "CpG_oe_分位",
            "lowcomp", "lowcomp_分位", "CAI", "CAI_分位", "MFE_kcal", "flag"]
    print(out[[c for c in show if c in out.columns]].to_string(index=False))

    print(f"\n  写出 {PAY}/element_features.tsv（{len(out)} 行）")
    print("  ⚠️ 每个分位都必须带『参照分布是哪一套』——不得出现无参照系的『偏低/偏高』。")


if __name__ == "__main__":
    main()
