#!/usr/bin/env python3
"""
30_offtarget_genomewide.py —— 全基因组脱靶扫描（含 PAM）

【为什么要做】`results/sgrna/README.md:151` 自己写着：
    「本口径只在**其它基因的 CDS** 里查重，不是全基因组 … **不额外声称全基因组唯一性**」
  而 CRISPR 领域的**标准口径就是全基因组 + PAM**。补上这一步，脱靶分析从「近似」变「标准」。

【口径】（写死，不开放式调参）
  SpCas9，**必须带 PAM**：
    正链：guide(20 nt) 后紧跟 GG
    负链：CC 后紧跟 revcomp(guide)
  **与现有 CDS 口径并列报，不替换** —— 现有 0 命中结论原样保留，另开一列。

【判据】见 判据预注册_outcome15_prereg.md §30。要点：
  A：除去靶位点后全基因组仍 0 命中 → 可写"含 PAM 的全基因组检索：0"
  B：出现靶基因外命中 → **如实报**（scaffold/坐标/是否重复区/是否内含子），
     表述改为"CDS 内 0、全基因组 N"，**不得**含糊成"基本唯一"

【自检门，不过即停】
  ① 正对照：扫描必须**重新找到**每条候选在 sgrna_candidates.tsv 里自己报的坐标
  ② 计数器自证：随机 12 nt 均值 与 L/4^12 同量级

【必须写明的免责】这是**序列层面的命中上界**。落在外显子外的位点在染色质层面
  通常不可及，**不等于可编辑脱靶**。不写这一句就是过度声称。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 30_offtarget_genomewide.py
"""
import os

import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
PKG = f"{IETX}/results/sgrna"
CANDS = f"{PKG}/sgrna_candidates.tsv"
GEN = f"{IETX}/genome/schMedS3_h1.fa"
CDS_FA = f"{IETX}/ref/cds.fa"
SEED = 0
N_RAND = 200
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def read_fai(path):
    fai = {}
    for ln in open(path + ".fai"):
        c, length, off, lb, lw = ln.split("\t")[:5]
        fai[c] = (int(length), int(off), int(lb), int(lw))
    return fai


def iter_scaffolds(path, fai):
    """逐 scaffold 读序列（814 MB 不整份进内存）。"""
    with open(path) as fh:
        for chrom, (L, off, lb, lw) in fai.items():
            fh.seek(off)
            need, buf, got = L, [], 0
            while got < need:
                ln = fh.readline()
                if not ln:
                    break
                s = ln.strip()
                buf.append(s)
                got += len(s)
            yield chrom, "".join(buf)[:need].upper()


def cds_blob():
    """CDS 串（用于判定命中是否落在外显子 —— 与 24_sgrna.py 同构，取每基因最长 CDS）。"""
    import re
    best, gid, buf = {}, None, []
    for ln in open(CDS_FA):
        if ln[0] == ">":
            if gid and len("".join(buf)) > len(best.get(gid, "")):
                best[gid] = "".join(buf).upper()
            m = re.search(r"gene_id=([^;]+)", ln)
            gid, buf = (m.group(1) if m else None), []
        elif gid:
            buf.append(ln.strip())
    if gid and len("".join(buf)) > len(best.get(gid, "")):
        best[gid] = "".join(buf).upper()
    return "|".join(re.sub(r"[^ACGT]", "", s) for s in best.values())


def scan(seq, query, pam, strand):
    """返回命中位置（1-based，指 protospacer 的首个碱基）。strand='+' → PAM 在后；'-' → PAM 在前。"""
    hits = []
    i = seq.find(query)
    while i != -1:
        if strand == "+":
            if seq[i + 20:i + 22] == pam:
                hits.append(i + 1)
        else:
            if i >= 2 and seq[i - 2:i] == pam:
                hits.append(i + 1)
        i = seq.find(query, i + 1)
    return hits


def main():
    print("═" * 78)
    print("30 · 全基因组脱靶扫描（SpCas9，必须带 PAM）")
    print("═" * 78)

    for p in (CANDS, GEN, GEN + ".fai"):
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}")
    d = pd.read_csv(CANDS, sep="\t", comment="#")
    print(f"  候选 {len(d)} 条；基因组 {GEN}")

    fai = read_fai(GEN)
    total_bp = sum(v[0] for v in fai.values())
    print(f"  scaffold {len(fai)} 条，合计 {total_bp / 1e6:.1f} Mb")

    cds = cds_blob()
    print(f"  CDS 串 {len(cds) / 1e6:.1f} Mb（用于判定命中是否落在外显子）")

    # 解析候选
    guides = []
    for _, r in d.iterrows():
        chrom, rng, strand = r.位置.split(":")
        a, b = (int(x) for x in rng.split("-"))
        assert b - a + 1 == 20, f"位置区间非 20 nt: {r.位置}"
        guides.append(dict(靶点=r.靶点, gene_id=r.gene_id, 序列=r.序列, 位置=r.位置,
                           guide=r.序列[:20], PAM=r.PAM[:2],
                           chrom=chrom, start=a, end=b, strand=strand))
    print(f"  解析 {len(guides)} 条 guide（20 nt 口径）")

    # ── 扫描 ──────────────────────────────────────────────────────────────
    res = {i: [] for i in range(len(guides))}
    found_own = {i: False for i in range(len(guides))}
    rng = np.random.default_rng(SEED)
    rands = ["".join(rng.choice(list("ACGT"), 12)) for _ in range(N_RAND)]
    rand_hits = np.zeros(len(rands), dtype=np.int64)

    for chrom, seq in iter_scaffolds(GEN, fai):
        for i, G in enumerate(guides):
            if G["chrom"] != chrom:
                continue
            for strand, q, pam in (("+", G["guide"], "GG"),
                                   ("-", revcomp(G["guide"]), "CC")):
                for pos in scan(seq, q, pam, strand):
                    res[i].append((chrom, pos, strand))
                    if strand == G["strand"] and pos == G["start"]:
                        found_own[i] = True
        for j, k in enumerate(rands):
            rand_hits[j] += seq.count(k)

    # ── 自检门 ①：正对照 ──────────────────────────────────────────────────
    bad = [i for i in range(len(guides)) if not found_own[i]]
    if bad:
        for i in bad:
            G = guides[i]
            print(f"  ✗ {G['序列']} 报告位置 {G['chrom']}:{G['start']}-{G['end']}:{G['strand']}"
                  f" **未在扫描中重现**（该 scaffold 命中 {len(res[i])} 处）")
        raise SystemExit("!! 正对照失败：扫描重现不出候选自己报的坐标 —— 读取器/链向有问题，停手")

    # ── 自检门 ②：计数器自证 ──────────────────────────────────────────────
    obs = float(rand_hits.mean())
    exp = total_bp / 4 ** 12
    print(f"\n  计数器自证：{N_RAND} 条随机 12 nt 均值 {obs:.3f}（理论 {exp:.3f}）；"
          f"20 nt 理论 {total_bp / 4 ** 20:.6f}")
    if not (0.5 * exp < obs < 2 * exp):
        raise SystemExit("!! 计数器自证未过 —— 全基因组命中数不可信，停手")
    print("  ✓ 正对照与计数器自证均通过")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(f"{'靶点':10s} {'序列':24s} {'报告位置':34s} {'命中':>5s} {'脱靶':>5s} {'inCDS':>6s} "
          f"{'scaffold数':>10s}")
    print("═" * 78)
    rows, any_off = [], False
    for i, G in enumerate(guides):
        hits = res[i]
        own = (G["chrom"], G["start"], G["strand"])
        off = [h for h in hits if h != own]
        # 【内部一致性】CDS 口径的命中数应与 sgrna_candidates.tsv 的『脱靶_全长命中』一致
        in_cds = cds.count(G["guide"]) + cds.count(revcomp(G["guide"]))
        nscaf = len(set(h[0] for h in off))
        if off:
            any_off = True
        print(f"{G['靶点']:10s} {G['序列']:24s} "
              f"{G['chrom'] + ':' + str(G['start']) + '-' + str(G['end']) + ':' + G['strand']:34s} "
              f"{len(hits):>5d} {len(off):>5d} {in_cds:>6d} {nscaf:>10d}")
        rows.append(dict(靶点=G["靶点"], gene_id=G["gene_id"], 序列=G["序列"],
                         报告位置=G["位置"] if "位置" in G else
                         f"{G['chrom']}:{G['start']}-{G['end']}:{G['strand']}",
                         全基因组命中=len(hits), 靶位点=1, 脱靶命中=len(off),
                         命中所在scaffold数=nscaf,
                         脱靶位置=";".join(f"{c}:{p}:{s}" for c, p, s in off[:20]) or "—"))

    out = pd.DataFrame(rows)
    with open(f"{PKG}/offtarget_genomewide.tsv", "w") as fh:
        fh.write("# 口径：SpCas9，必须带 PAM（正链 guide+GG；负链 CC+revcomp(guide)）；"
                 "全基因组（非仅 CDS），正负链均扫。\n"
                 "# 与 sgrna_candidates.tsv 的『脱靶_全长命中』（CDS 口径）**并列，不替换**。\n"
                 "# ⚠️ 这是序列层面的命中上界，不等于可编辑脱靶（染色质可及性未计入）。\n")
        out.to_csv(fh, sep="\t", index=False)

    print("\n" + "═" * 78)
    print("判据应用")
    print("═" * 78)
    if not any_off:
        print("  ⇒ 判据 **A**：含 PAM 的全基因组检索，除去靶位点后 **0 命中**。")
        print("     交付文本可写「含 PAM 的全基因组检索：0」——强于现有 CDS 口径。")
    else:
        print("  ⇒ 判据 **B**：出现靶基因外命中 —— **必须如实报**，给出 scaffold 与坐标。")
        print("     交付文本改为「CDS 内 0、全基因组 N」，**不得**含糊成『基本唯一』。")
        print("     并需说明：这些位点是否落在重复区 / 内含子；序列命中 ≠ 可编辑脱靶。")
    print(f"\n  写出 {PKG}/offtarget_genomewide.tsv（{len(out)} 行）")
    print("  ⚠️ 免责句必须进交付文本：序列层面命中上界，不等于可编辑脱靶。")


if __name__ == "__main__":
    main()
