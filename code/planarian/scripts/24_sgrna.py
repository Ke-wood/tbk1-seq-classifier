#!/usr/bin/env python3
"""
24_sgrna.py —— sgRNA 候选设计（赛道子任务1「底盘与向导模块升级」）

【为什么要做】申报摘要承诺了 sgRNA 集，而此前所有脚本都围绕 IRS 与载荷构建体，
  没有任何一个落在「向导模块」上。没有它，项目不落在赛道的基因编辑子任务上。

【三个必须交代的前提】
  1. Cas 与 PAM：用 **SpCas9 + NGG**。⚠️ 三轮文献检索**均未找到涡虫 CRISPR 的方案文献**，
     故这是**默认选择**，非涡虫特异证据支持。此事实如实写入产出。
  2. on-target 打分用【规则基础】，**禁止套用哺乳动物训练的打分模型**（如 Doench 规则集
     的 ML 版本）—— 那又是一次跨物种迁移，恰好印证本项目的全局论点。
  3. off-target 复用发卡那步的 CDS 精确匹配机器（28.6 Mb），报两个口径。

【时间盒】硬停 36 小时；做不完即停，在 PPT 里写"列为后续工作"，不硬凑。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 24_sgrna.py
"""
import os
import re
import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, REF, GEN = f"{IETX}/results", f"{IETX}/ref", f"{IETX}/genome"
GFF = f"{IETX}/annotation/schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3"
PKG = f"{OUT}/sgrna"
FA = f"{GEN}/schMedS3_h1.fa"

# 靶点（申报摘要点名）。smedwi-1 与 piwi-1 在本注释中指向同一 UniProt 条目（PIWI1_SCHMD），
# 故合并为一个靶点，并在产出中写明该判断。
TARGETS = [
    ("piwi-1 / smedwi-1", "h1SMcG0013999", "两个名称在 UniProt PIWI1_SCHMD 下指向同一序列，合并"),
    ("TBK1", "h1SMcG0013953", "本项目敲降已验证（log2FC −1.98，p=0.004）"),
]
GUIDE_LEN, PAM = 20, "NGG"
BLOB = {}          # 其它基因的 CDS 大串（脱靶背景）


def read_fai(path):
    """从 samtools 的 .fai 读 (length, offset, line_bases, line_width)。
    ⚠️ 【静默失败 · 首版】自己 in-place 建索引，seek 时用 `offset + start - 1` ——
       算漏了 FASTA 每 60 bp 一个换行。本基因组 .fai = `60 61`，故每条「外显子序列」
       实际从错位约 3 Mb 的上游读起，10 条 guide **全部不是靶基因的序列**；
       而坐标自洽、PAM 看着对、脚本零报错。唯一能抓住它的是
       【回取自检】：拿算出的坐标回基因组取序列，必须等于 guide 本身。
       该自检已加在本文件 scan_exon() 之后。"""
    fai = {}
    for ln in open(path + ".fai"):
        c, length, off, lb, lw = ln.split("\t")[:5]
        fai[c] = (int(length), int(off), int(lb), int(lw))
    return fai


class Fa:
    def __init__(self, path, fai):
        self.f = open(path)
        self.fai = fai

    def get(self, chrom, start, end):
        """1-based 闭区间取序列。字节偏移 = 起始字节 + 整行数×行宽 + 行内余数。"""
        L, off, lb, lw = self.fai[chrom]
        start, end = max(1, start), min(L, end)
        if end < start:
            return ""
        p0 = off + (start - 1) // lb * lw + (start - 1) % lb
        need, buf = end - start + 1, []
        self.f.seek(p0)
        got = 0
        while got < need:
            line = self.f.readline()
            if not line:
                break
            s = line.strip()
            buf.append(s)
            got += len(s)
        return "".join(buf)[:need].upper()


def load_exons():
    """基因 → 代表转录本的外显子表。
    ⚠️ 必须【按转录本】建表再选代表 —— 首版把同一基因所有转录本的外显子求和，
       TBK1 因此显示 83 个外显子 / 11,676 bp CDS（真值约 2,046 bp、十余个外显子），
       guide 位置与计数全错。取 CDS 总长最长的转录本作代表。"""
    tx_ex, tx_gene, tx_cdslen = {}, {}, {}
    with open(GFF) as fh:
        for ln in fh:
            if ln[0] == "#":
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            typ, attr = f[2], f[8]
            if typ == "transcript":
                m = re.search(r"(?:^|;)gene_id=([^;]+)", attr)
                t = re.search(r"(?:^|;)ID=([^;]+)", attr)
                if m and t:
                    tid = t.group(1).rstrip(".")
                    tx_gene[tid] = m.group(1)
                    tx_ex.setdefault(tid, [])
                    tx_cdslen.setdefault(tid, 0)
            elif typ == "exon":
                p = re.search(r"(?:^|;)Parent=([^;]+)", attr)
                if p:
                    tid = p.group(1).rstrip(".")
                    tx_ex.setdefault(tid, []).append((f[0], int(f[3]), int(f[4]), f[6]))
            elif typ == "CDS":
                p = re.search(r"(?:^|;)Parent=([^;]+)", attr)
                if p:
                    tid = p.group(1).rstrip(".")
                    tx_cdslen[tid] = tx_cdslen.get(tid, 0) + int(f[4]) - int(f[3]) + 1
    # 每基因选 CDS 最长的转录本
    best_tx = {}
    for tid, g in tx_gene.items():
        if not tx_ex.get(tid):
            continue
        if g not in best_tx or tx_cdslen.get(tid, 0) > tx_cdslen.get(best_tx[g], 0):
            best_tx[g] = tid
    out = {}
    for g, tid in best_tx.items():
        ex = sorted(tx_ex[tid], key=lambda x: x[1])
        # ⚠️ 外显子编号必须按【转录顺序】，不能按基因组坐标升序：
        #    负链基因的转录起点在【高】坐标处，直接升序编号会把外显子顺序整个倒过来。
        #    本项目两个靶点 (PIWI/TBK1) 都在负链上，故这条不是理论问题。
        if ex and ex[0][3] == "-":
            ex = ex[::-1]
        out[g] = [(c, s0, e0, st, i + 1) for i, (c, s0, e0, st) in enumerate(ex)]
    return out, best_tx


def scan_exon(fa, chrom, s, e, eno):
    """在一段【真实基因组序列】上扫【两条链】的 NGG 向导。
    返回 [(guide22, chrom, start, end, strand, exon_no), ...]
    序列口径：**22 nt = 20 nt guide + PAM 的 GG**（与附件(1) 表头一致，取 g[:20] 得 guide）。
    坐标口径：**guide 全长 20 nt 的基因组区间，不含 PAM**；正负链同一口径。

    ⚠️ 【静默失败 · 首版】三个错叠在一起：
       ① 正链分支返回 20 nt，反向互补扫出来的那条返回 23 nt（guide+PAM），
          两种口径混在同一列；
       ② 反向互补分支硬写 strand="-"，正链上的位点也被标成负链；
       ③ 该分支还带一个 `- GUIDE_LEN - 3` 的凑数偏移，坐标整体错 3 bp。
       三者都不报错，只有「拿坐标回基因组取序列、看是否等于 guide」能抓住。

    ⚠️ 【静默失败 · 本版初稿】改成按【外显子自身的链】分支去扫，看着更"对称"，
       实则把反义链整条丢了 —— 本项目两个靶点都在负链上，于是正链候选数恰好为 0。
       若按靶基因分别统计链分布就会发现，故已补该统计。
       **两条链都要扫，与外显子自身的链无关。**"""
    seq = fa.get(chrom, s, e)
    out = []
    pat = r"(?=([ACGT]{%d}GG))" % GUIDE_LEN
    # 正链：guide 即基因组正链序列，PAM 在其 3' 侧（高坐标）
    for m in re.finditer(pat, seq):
        i = m.start()
        out.append((m.group(1), chrom, s + i, s + i + GUIDE_LEN - 1, "+", eno))
    # 负链：在反向互补串上扫，再换算回基因组坐标
    r = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    for m in re.finditer(pat, r):
        i = m.start()
        # r[i:i+20] 取自 seq[L-1-i .. L-20-i] ⇒ 基因组 [s+L-20-i, s+L-1-i] = [e-19-i, e-i]
        out.append((m.group(1), chrom, e - 19 - i, e - i, "-", eno))
    return out


def verify_coords(fa, rows):
    """【回取自检】唯一能抓住坐标错位的检查。
    拿报告出去的坐标回基因组取 20 nt，正链须等于 guide、负链须等于其反向互补。
    任何一条不符即中止，不产出结果 —— 宁可空手，不可交一份自洽但错的表。"""
    comp = str.maketrans("ACGTN", "TGCAN")
    bad = [(g, c, a, b, sd) for g, c, a, b, sd, _ in rows
           if fa.get(c, a, b) != (g[:GUIDE_LEN] if sd == "+" else g[:GUIDE_LEN].translate(comp)[::-1])]
    if bad:
        print(f"!! 坐标回取自检失败 {len(bad)} 条：")
        for x in bad[:3]:
            print("   ", x)
        raise SystemExit("坐标口径错误 —— 停手，勿产出。")
    print(f"  ✓ 坐标回取自检通过：{len(rows)} 条 guide 的基因组 20 nt 区间与序列一致")


def rule_score(g):
    """【规则基础】on-target 打分。不使用任何跨物种训练的模型。
    参考的是 sgRNA 设计的通用生化约束，不是哺乳动物数据的拟合权重。"""
    gc = (g.count("G") + g.count("C")) / len(g)
    s, notes = 0.0, []
    # GC 40–70% 最优（过低结合弱、过高易二级结构）
    if 0.40 <= gc <= 0.70:
        s += 1.0
    elif 0.30 <= gc < 0.40 or 0.70 < gc <= 0.80:
        s += 0.5; notes.append("GC边缘")
    else:
        notes.append("GC极端")
    # 无 ≥4 同聚物（合成与转录困难）
    if not re.search(r"(A{4,}|T{4,}|G{4,}|C{4,})", g):
        s += 1.0
    else:
        notes.append("同聚物")
    # 无 poly-T（Pol III 终止信号，影响 sgRNA 转录）
    if "TTTT" not in g:
        s += 0.5
    else:
        notes.append("polyT")
    # 种子区（PAM 近端 12 nt）GC 不宜极端
    seed = g[-12:]
    sg = (seed.count("G") + seed.count("C")) / len(seed)
    if 0.25 <= sg <= 0.75:
        s += 1.0
    else:
        notes.append("种子GC极端")
    # 5' 端为 G（T7/Pol III 转录起始偏好）
    if g[0] == "G":
        s += 0.5
    return round(s / 4.0, 3), round(gc, 3), round(sg, 3), ";".join(notes)


def main():
    os.makedirs(PKG, exist_ok=True)
    print("═" * 78)
    print("sgRNA 候选设计（赛道子任务1 · 向导模块）")
    print("═" * 78)

    print("\n── 前提 1：Cas 与 PAM ──")
    print(f"  使用 SpCas9，PAM = {PAM}")
    print("  ⚠️ 依据：**三轮文献检索均未找到涡虫 CRISPR 的方案文献**，")
    print("     故 NGG 是【默认选择】，非涡虫特异证据支持。此限制写入产出文档。")

    FAI = read_fai(FA)
    fa = Fa(FA, FAI)
    exons, best_tx = load_exons()
    print(f"\n  基因组 contig {len(FAI)}；有外显子结构的基因 {len(exons)}")

    # 脱靶背景：其它基因的 CDS（与发卡那步同一台机器）
    # ⚠️ 同样必须【取最长】而不是求和 —— 求和会把 CDS 背景从 28.6 Mb 虚增到 53.6 Mb，
    #    脱靶计数随之虚高。口径与发卡那一步（22_payload.longest_cds）保持一致。
    cds = {}
    gid, buf = None, []
    for ln in open(f"{REF}/cds.fa"):
        if ln[0] == ">":
            if gid and len("".join(buf)) > len(cds.get(gid, "")):
                cds[gid] = "".join(buf).upper()
            m = re.search(r"gene_id=([^;]+)", ln)
            gid, buf = (m.group(1) if m else None), []
        elif gid:
            buf.append(ln.strip())
    if gid and len("".join(buf)) > len(cds.get(gid, "")):
        cds[gid] = "".join(buf).upper()
    blob_all = "|".join(cds.values())
    print(f"  脱靶背景：全部基因 CDS 共 {len(blob_all)/1e6:.1f} Mb")
    # 【计数器自证】候选的脱靶数是 0 —— 报 0 之前必须先证明这个计数器数得出数，
    # 否则「0 命中」与「计数器坏了」不可区分。这一段是产出 0 的前提条件，不是装饰。
    rng = np.random.default_rng(0)
    rand = "".join(rng.choice(list("ACGT"), 200 * 12))
    obs12 = float(np.mean([blob_all.count(rand[i * 12:(i + 1) * 12]) for i in range(200)]))
    exp12 = len(blob_all) / 4 ** 12
    print(f"  计数器自证：已知重复串 AAAAAAAAAA 命中 {blob_all.count('AAAAAAAAAA')}；")
    print(f"              200 条随机 12 nt 均值 {obs12:.2f}（理论 {exp12:.2f}）；"
          f"20 nt 理论 {len(blob_all)/4**20:.4f}")
    if not (0.5 * exp12 < obs12 < 2 * exp12):
        print("  !! 计数器自证未过 —— 脱靶计数不可信，勿采信 0 命中")
        raise SystemExit(1)

    # 同源簇（跨簇判定用）
    import importlib.util as u
    sp = u.spec_from_file_location("v16", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "16_validation.py"))
    v16 = u.module_from_spec(sp); sp.loader.exec_module(v16)
    clust, _ = v16.clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    # ⚠️ 首版这一列写成 `clusters_from_diamond.__name__ and (...)`，恒为 -1，读起来像坏列。
    #    第二版改成「不在簇表里 = 0」，**仍然错**，且错得更隐蔽：
    #    `clusters_from_diamond` 在建连通分量【之前】先剔除了自身命中（qg == sg），
    #    所以「无同源」的基因根本不出现在 clust.index 里 —— 报 0 看着像个占位值。
    #    但一个基因【至少与自己在同一个簇里】，成员数不可能为 0。
    #    正确口径：**≥1，含自身；1 = 无过阈同源（单例）**。
    #    （本项目两个靶点实测都是 1：TBK1 在 allvall 里有 5,340 行，但过阈的 36 行
    #      全是自身命中，剔除后无边；PIWI 19 行里过阈的 1 行同样是自身命中。）
    #    对向导设计有用：靶基因有近同源旁系时，guide 的特异性风险更高。
    CL_SIZE = {g: int((clust == clust[g]).sum()) for g in clust.index} if len(clust) else {}
    print(f"  同源簇表：{len(clust)} 个基因有过阈同源"
          f"（pident≥{v16.PID}%、alnlen≥{v16.ALN}aa、e≤{v16.EV}；自身命中已剔除）")
    for _, gid_, _ in TARGETS:
        n = CL_SIZE.get(gid_, 1)
        print(f"    {gid_} 同源簇成员数 = {n}"
              f"（{'单例 —— 无过阈同源' if gid_ not in CL_SIZE else '有近同源旁系'}；含自身）")

    rows = []
    for name, gid_, note in TARGETS:
        print(f"\n── 靶点 {name} ({gid_}) ──  {note}")
        if gid_ not in exons:
            print("  ⚠️ 无外显子结构，跳过")
            continue
        blob_other = blob_all.replace(cds.get(gid_, "|"), "|")  # 排除靶基因自身
        cds_t = cds.get(gid_, "")
        print(f"  CDS {len(cds_t)} bp；外显子 {len(exons[gid_])} 个")
        cands = []
        for (chrom, s, e, _strand, eno) in exons[gid_]:
            if e - s + 1 < 30:
                continue
            cands += scan_exon(fa, chrom, s, e, eno)
        n_raw = len(cands)
        cands = list(dict.fromkeys(cands))          # 按 (guide, 区间, 链) 去重
        print(f"  NGG 位点（外显子内）{n_raw} 个，去重后 {len(cands)} 个"
              f"（正链 {sum(1 for x in cands if x[4] == '+')}"
              f" / 负链 {sum(1 for x in cands if x[4] == '-')}）"
              f"  ← 两条链都必须非零，否则是扫描分支写错了")
        verify_coords(fa, cands)
        scored = []
        for g, c_, st_, en_, sd_, eno in cands:
            # ⚠️ 打分与脱靶只能喂 guide 的 20 nt —— 首版把整条 22 nt（含 PAM 的 GG）
            #    喂进 rule_score，于是 PAM 的两个 G 被算进 GC 与种子区，且可能凑出假
            #    的 G{4,} 同聚物。PAM 不是 guide 的一部分。
            guide = g[:GUIDE_LEN]
            sc, gc, sg, notes = rule_score(guide)
            if "GC极端" in notes or "同聚物" in notes:
                continue
            full_ot = blob_other.count(guide)
            seed_ot = blob_other.count(guide[-12:])   # 种子区 = guide 的 PAM 近端 12 nt
            scored.append(dict(靶点=name, gene_id=gid_, 序列=g, PAM="NGG",
                               位置=f"{c_}:{st_}-{en_}:{sd_}", 外显子=eno,
                               规则分=sc, GC=gc, 种子GC=sg, 备注=notes or "—",
                               脱靶_全长命中=full_ot, 脱靶_12nt种子=seed_ot,
                               同源簇大小=CL_SIZE.get(gid_, 1)))   # ≥1，含自身；1 = 单例
        d = pd.DataFrame(scored)
        if d.empty:
            print("  ⚠️ 无通过基础过滤的候选")
            continue
        # 特异性排序：脱靶少者优先，其次规则分。末位按「位置」定序，使结果可复现
        # （规则分有大量并列 —— 首版 10 条全为 0.875，不能声称「筛选出最优 5 条」，
        #   故并列时的取舍必须是一个写明了的、确定性的规则，而不是 sort 的默认顺序）。
        d = d.sort_values(["脱靶_全长命中", "脱靶_12nt种子", "规则分", "位置"],
                          ascending=[True, True, False, True])
        # 全长脱靶为 0 的里挑规则分最高的 5 条
        top = pd.concat([d[d.脱靶_全长命中 == 0].head(5),
                         d[d.脱靶_全长命中 > 0].head(2)]).drop_duplicates("序列").head(5)
        print(f"  规则分并列情况：{d.规则分.value_counts().to_dict()}"
              f"  （满分 {d.规则分.max():.3f}，最高分候选 {int((d.规则分 == d.规则分.max()).sum())} 条）")
        for _, r in top.iterrows():
            print(f"    {r.序列}  {r.PAM}  {r.位置}  外显子{r.外显子}  "
                  f"规则分={r.规则分:.2f}  GC={r.GC:.2f}  "
                  f"脱靶全长={r.脱靶_全长命中} 12nt={r.脱靶_12nt种子}")
        rows.append(top)

    if rows:
        allc = pd.concat(rows)
        # 文件头注释行：写明两处口径，避免下游再出现「20 nt / 23 nt 混用」这类歧义。
        # 读法：pd.read_csv(..., sep="\t", comment="#")
        head = ("# 口径：① 序列 = 22 nt（20 nt guide + PAM 的 GG）；"
                "② 位置 = guide 全长 20 nt 的基因组闭区间 scaffold:start-end:strand，"
                "不含 PAM，正负链同口径；③ 外显子编号按转录顺序（1 = 转录起点所在外显子）；"
                "④ 同源簇大小 = 该基因所在同源簇的成员数（≥1，含自身；1 = 无过阈同源的单例）。\n")
        with open(f"{PKG}/sgrna_candidates.tsv", "w") as fh:
            fh.write(head)
            allc.to_csv(fh, sep="\t", index=False)
        print(f"\n  写出 {PKG}/sgrna_candidates.tsv（{len(allc)} 条，含 1 行口径注释）")
    else:
        print("\n  ⚠️ 未产出任何候选")


if __name__ == "__main__":
    main()
