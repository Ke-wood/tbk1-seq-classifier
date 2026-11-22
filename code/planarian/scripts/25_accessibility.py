#!/usr/bin/env python3
"""
25_accessibility.py —— sgRNA 结构可及性补算（赛道子任务1 的「结构」一项）

【为什么要做】子任务1 要求「结构、序列和功能机制分析」。序列与功能机制已在
  `24_sgrna.py`（规则打分）与脱靶检索里做了；**结构**一项此前是空的。
  本脚本只补这一项。

【本脚本报什么、不报什么 —— 先说清楚】
  报：两条**相对**指标，以及由此得到的**相对排序**。
    (a) guide 单链 MFE（kcal/mol）—— 越接近 0 越不易自折叠；
    (b) 种子区（PAM 近端 12 nt）在**局部上下文**中的未配对碱基比例 —— 越高越可及。
  不报：**任何预测编辑效率的绝对值**。涡虫没有任何可用的 on-target 训练数据，
  任何「效率 = x%」的写法都是把哺乳动物数据或纯臆测迁移过来 —— 恰好是本项目
  在两处已经否定过的那个动作（同源注释覆盖仅 40%；预训练模型无增益）。
  故本脚本的输出**不得**被表述为「预测编辑效率」。

【方法写死，不开放式调参】（时间盒 ≤3 小时，一轮）
  · 上下文 = guide 所在链 ± 两侧各 20 nt，共 60 nt；
  · 折叠用 ViennaRNA 默认参数（Turner 2004 等，`--noPS`，不指定 -P/-d）；
  · 种子未配对比例 = 结构串里种子 12 位中 '.' 的占比。
  不做参数扫描、不做窗口尺寸比较 —— 换窗口只会得到另一组同样无法验证的数字。

【静默失败防范】每一步都打印丢掉了什么、保留了什么；产不出即为产不出，不补零。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 25_accessibility.py
"""
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

# 用当前解释器同目录下的 RNAfold —— 不依赖 PATH 上恰好有它
RNAFOLD = os.path.join(os.path.dirname(sys.executable), "RNAfold")

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
PKG = f"{IETX}/results/sgrna"
GEN = f"{IETX}/genome/schMedS3_h1.fa"
CANDS = f"{PKG}/sgrna_candidates.tsv"
FLANK = 20
SEED_LEN = 12
COMP = str.maketrans("ACGTN", "TGCAN")


def read_fai(path):
    fai = {}
    for ln in open(path + ".fai"):
        c, length, off, lb, lw = ln.split("\t")[:5]
        fai[c] = (int(length), int(off), int(lb), int(lw))
    return fai


def get_seq(fh, fai, chrom, start, end):
    """1-based 闭区间；按 .fai 换算字节偏移（FASTA 每 60 bp 一个换行，必须补）。"""
    L, off, lb, lw = fai[chrom]
    start, end = max(1, start), min(L, end)
    if end < start:
        return ""
    fh.seek(off + (start - 1) // lb * lw + (start - 1) % lb)
    need, buf, got = end - start + 1, [], 0
    while got < need:
        line = fh.readline()
        if not line:
            break
        s = line.strip()
        buf.append(s)
        got += len(s)
    return "".join(buf)[:need].upper()


def run_rnafold(records):
    """records: [(id, seq)]. 返回 {id: (mfe, structure)}。RNAfold 缺一条即报错。"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False)
    try:
        for i, s in records:
            tmp.write(f">{i}\n{s}\n")
        tmp.close()
        p = subprocess.run([RNAFOLD, "--noPS", tmp.name],
                           capture_output=True, text=True)
    finally:
        os.unlink(tmp.name)
    if p.returncode != 0:
        raise SystemExit(f"RNAfold 失败: {p.stderr[:400]}")
    out, i = {}, 0
    lines = p.stdout.splitlines()
    while i < len(lines):
        if lines[i].startswith(">"):
            rid = lines[i][1:].strip()
            struct_line = lines[i + 2]
            m = re.match(r"^([.()\[\]{}<>|,:_~+-]*)\s+\(\s*(-?[0-9.]+)\s*\)\s*$", struct_line)
            if not m:
                out[rid] = (None, None)
            else:
                out[rid] = (float(m.group(2)), m.group(1))
            i += 3
        else:
            i += 1
    return out


def main():
    print("═" * 78)
    print("sgRNA 结构可及性（赛道子任务1 · 结构一项）")
    print("═" * 78)
    print(f"  方法写死：上下文 = guide 所在链 ±{FLANK} nt；"
          f"RNAfold 默认参数；种子 = PAM 近端 {SEED_LEN} nt")
    print("  ⚠️ 只报相对排序，**不报预测编辑效率的绝对值**（涡虫无可用训练数据）")

    fai = read_fai(GEN)
    d = pd.read_csv(CANDS, sep="\t", comment="#")
    print(f"\n  读入候选 {len(d)} 条（{CANDS}）")

    fh = open(GEN)
    recs, meta = [], []
    dropped = []
    for k, r in d.iterrows():
        c, rng, sd = r.位置.split(":")
        a, b = (int(x) for x in rng.split("-"))
        if c not in fai:
            dropped.append((r.序列, c, "contig 不在 .fai"))
            continue
        s0, e0 = a - FLANK, b + FLANK
        ctx = get_seq(fh, fai, c, s0, e0)
        if len(ctx) != b - a + 1 + 2 * FLANK:
            dropped.append((r.序列, f"{c}:{s0}-{e0}", f"取到 {len(ctx)} nt，非 {b-a+1+2*FLANK}"))
            continue
        # 统一到【guide 所在的那条链】：负链取反向互补，使上下文与 sgRNA 同向
        if sd == "-":
            ctx = ctx.translate(COMP)[::-1]
        guide = r.序列[:20]
        # 【构造自检】上下文中间 20 nt 必须就是 guide；种子必须是其末端 12 nt
        if ctx[FLANK:FLANK + 20] != guide:
            raise SystemExit(f"!! 上下文构造自检失败 {r.位置}："
                             f"ctx[20:40]={ctx[FLANK:FLANK+20]} != guide={guide}")
        seed = ctx[FLANK + 20 - SEED_LEN:FLANK + 20]
        if seed != guide[-SEED_LEN:]:
            raise SystemExit(f"!! 种子取位自检失败 {r.位置}")
        recs.append((f"C{k}", ctx.replace("T", "U")))
        recs.append((f"G{k}", guide.replace("T", "U")))
        meta.append((k, r, ctx, seed))
    fh.close()

    print(f"  保留 {len(meta)} 条；丢弃 {len(dropped)} 条")
    for x in dropped:
        print(f"    ✗ 丢弃 {x}")
    if not meta:
        raise SystemExit("无可用候选 —— 不产出空表")

    folds = run_rnafold(recs)
    miss = [i for i, _ in recs if folds.get(i, (None, None))[0] is None]
    if miss:
        raise SystemExit(f"!! RNAfold 未解析出 {len(miss)} 条：{miss[:5]} —— 停手，勿产出")
    print(f"  ✓ RNAfold 全部解析成功：{len(recs)} 条（上下文 {len(meta)} + guide 单链 {len(meta)}）")

    rows = []
    for k, r, ctx, seed in meta:
        cmfe, cstruct = folds[f"C{k}"]
        gmfe, _ = folds[f"G{k}"]
        # 种子在 60 nt 上下文中的位置：紧贴 guide 3' 端
        i0 = FLANK + 20 - SEED_LEN
        sub = cstruct[i0:i0 + SEED_LEN]
        if len(sub) != SEED_LEN:
            raise SystemExit(f"!! 结构串长度不足 {r.位置}")
        unpaired = sub.count(".")
        rows.append(dict(靶点=r.靶点, gene_id=r.gene_id, 序列=r.序列, 位置=r.位置,
                         外显子=r.外显子, 规则分=r.规则分,
                         上下文MFE_kcal=cmfe,
                         guide单链MFE_kcal=gmfe,
                         种子_unpaired=unpaired,
                         种子_unpaired比例=round(unpaired / SEED_LEN, 3),
                         种子结构=sub))
    out = pd.DataFrame(rows)
    # 相对排序：种子越可及越好；并列时 guide 自折叠越弱越好；再并列按位置定序（可复现）
    out = out.sort_values(["种子_unpaired比例", "guide单链MFE_kcal", "位置"],
                          ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "可及性排名", range(1, len(out) + 1))

    head = ("# 口径：① 上下文 = guide 所在链 ±20 nt 共 60 nt（负链已取反向互补，与 sgRNA 同向）；"
            "② RNAfold 默认参数（--noPS，未调参）；③ 种子 = guide 的 PAM 近端 12 nt；"
            "④ 本表只给【相对排序】，不得表述为预测编辑效率。\n")
    with open(f"{PKG}/accessibility.tsv", "w") as fh:
        fh.write(head)
        out.to_csv(fh, sep="\t", index=False)

    print(f"\n  写出 {PKG}/accessibility.tsv（{len(out)} 条）")
    print(f"  种子未配对比例：{out.种子_unpaired比例.value_counts().sort_index().to_dict()}")
    print("\n" + out[["可及性排名", "靶点", "序列", "位置", "种子_unpaired比例",
                      "guide单链MFE_kcal", "上下文MFE_kcal"]].to_string(index=False))
    print("\n  ⚠️ 全部为相对排序。**不构成对编辑效率的预测。**")


if __name__ == "__main__":
    main()
