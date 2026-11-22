#!/usr/bin/env python3
"""
22_payload.py —— C2/C3：免疫逃逸型载荷表达盒设计与打分

【设计逻辑（已定前提，不再论证）】
  注射 DNA 不触发转录免疫应答（零结果）→ 骨架/CpG/修饰不是瓶颈
  限制转基因的是 TBK1 轴的组成型抑制
  序列层面的免疫调控语法信号弱 → "改序列躲识别"不可行
  ⇒ 免疫逃逸 = 【主动关闭 TBK1 轴】，不是被动躲避。

【效应器】C1 筛选结果为「无」（见 21_c1_screen.py）——上调集中不存在可识别的
  TBK1/NF-κB 负反馈调控因子。故效应器走【TBK1/MRC1 发卡】，这也正是 Drees 2026 的做法。

【启动子选择依据】
  规则 1（文献）优先。Drees 2026（Development 153:dev205222）用 PlanMine 查高表达管家基因，
  克隆 eF1α / tubulin / actin 的 ORF 上下游 800–2000 bp，驱动 Smed 密码子优化的报告基因。
  → 本项目采用 eF1α 启动子。
  规则 2 独立佐证：本项目自己的 38 样本数据里，eF1α（h1SMcG0010823）在
  「蛋白编码 + 全样本检出 + 有 1 kb 上游 + 非线粒体」的 7,815 个基因中按表达量排第 5。

【骨架】eGFP / NanoLuc / Cas9，均按【涡虫密码子使用】重编码（G4）。
  ⚠️ G4 要求的张力如实报告：优化改的正是碱基组成，而 IRS 恰恰吃组成 → 报优化前后 IRS 分数变化。

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 22_payload.py
"""
import os
import re
import numpy as np
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, REF, FEAT = f"{IETX}/results", f"{IETX}/ref", f"{IETX}/features"
PKG = f"{OUT}/payload"
BB = "/tmp/bb"
BASES = "TCAG"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
AA_OF = {c: AA[i] for i, c in enumerate(CODONS)}

# 常用克隆位点（合成 QC：优化后序列不得含这些）
RE_SITES = {
    "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC", "SacI": "GAGCTC", "KpnI": "GGTACC", "SmaI": "CCCGGG",
    "PstI": "CTGCAG", "SalI": "GTCGAC", "XbaI": "TCTAGA", "SpeI": "ACTAGT",
    "NheI": "GCTAGC", "AgeI": "ACCGGT", "AscI": "GGCGCGCC", "PacI": "TTAATTAA",
    "BglII": "AGATCT", "MluI": "ACGCGT", "NdeI": "CATATG", "SphI": "GCATGC",
}
STOP = {"TAA", "TAG", "TGA"}


# ─────────────────────────── 读取 ───────────────────────────
def read_fa(path, key=None):
    seq, name, buf = {}, None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name:
                    seq[name] = "".join(buf).upper()
                name = (re.search(r"gene_id=([^;]+)", ln).group(1)
                        if key == "gene_id" and re.search(r"gene_id=([^;]+)", ln)
                        else ln[1:].split("::")[0].strip())
                buf = []
            else:
                buf.append(ln.strip())
    if name:
        seq[name] = "".join(buf).upper()
    return seq


def longest_cds():
    """每个基因取最长 CDS（多转录本时）。"""
    best = {}
    for gid, s in read_fa(f"{REF}/cds.fa", key="gene_id").items():
        s = re.sub(r"[^ACGT]", "", s.replace("U", "T"))
        if len(s) > len(best.get(gid, "")):
            best[gid] = s
    return best


def codon_table(cds):
    """从本物种 CDS 全集自算密码子使用频率（G4 要求）。"""
    cnt = dict.fromkeys(CODONS, 0)
    for s in cds.values():
        for i in range(0, len(s) - 2, 3):
            c = s[i:i + 3]
            if c in cnt:
                cnt[c] += 1
    tot = sum(cnt.values())
    freq = {c: cnt[c] / tot for c in CODONS}
    aa_tot = {}
    for c in CODONS:
        aa_tot[AA_OF[c]] = aa_tot.get(AA_OF[c], 0) + freq[c]
    # 偏好密码子：每个氨基酸里使用频率最高的
    pref = {}
    for c in CODONS:
        a = AA_OF[c]
        if a == "*":
            continue
        if a not in pref or freq[c] > freq[pref[a]]:
            pref[a] = c
    return cnt, freq, pref, aa_tot


# ─────────────────────── 密码子优化 ───────────────────────
def opt_codon(prot, freq, pref, gc_lo=0.35, gc_hi=0.65, max_homopolymer=5, seed=0):
    """按偏好密码子优化；同义替换规避：限制酶位点、同聚物、GC 极值窗口。
    贪心：首选偏好密码子，冲突时在同义密码子里挑代价最小的。"""
    rng = np.random.default_rng(seed)
    syn = {}
    for c in CODONS:
        if AA_OF[c] != "*":
            syn.setdefault(AA_OF[c], []).append(c)
    out = []
    for i, aa in enumerate(prot):
        if aa == "*":
            out.append("TGA")            # 用 TGA 作终止（涡虫偏好，见 codon_table）
            continue
        # ⚠️ 不能永远取最高频密码子 —— 涡虫高频密码子极度偏 AT，取 argmax 会把 GC 从
        #    天然 36% 压到 24%，反而造出一堆 poly-A/T 串（实测同聚物 10、多个位点清不掉）。
        #    改为【按天然使用频率加权抽样】：既用偏好密码子，又保住碱基组成。
        pool = syn.get(aa, [])
        if not pool:
            raise ValueError(f"未知氨基酸 {aa!r}")
        wts = np.array([freq[c] for c in pool], dtype=float)
        wts = wts / wts.sum()
        order = list(np.random.default_rng(seed + i).choice(pool, size=len(pool),
                                                            replace=False, p=wts))
        cands = order + pool                       # 抽样序在前，兜底再遍历全体
        chosen = None
        for c in cands:
            trial = "".join(out) + c
            if any(s in "".join(out[-12:]) + c for s in ()):   # 占位
                pass
            tail = trial[-max_homopolymer:]
            if len(set(tail)) == 1 and len(tail) >= max_homopolymer:
                continue
            if any(site in c or site in trial[-8:] + c for site in RE_SITES.values()):
                continue
            g = (trial[-200:].count("G") + trial[-200:].count("C")) / max(len(trial[-200:]), 1)
            if i > 60 and not (gc_lo <= g <= gc_hi):
                continue
            chosen = c
            break
        out.append(chosen or cands[0])
    seq = "".join(out)
    # 交替清理：两者互为约束（消同聚物可能造出位点，反之亦然）
    for _ in range(4):
        seq = _kill_homopolymers(seq, prot, syn, freq, max_homopolymer)
        seq = _kill_sites(seq, prot, syn, freq)
        if not any(x in seq for x in RE_SITES.values()) and \
           not re.search(r"(A{%d,}|T{%d,}|G{%d,}|C{%d,})" % ((max_homopolymer,) * 4), seq):
            break
    return seq


def _retranslate(seq, prot, syn, freq, i):
    """在第 i 个密码子处换一个同义密码子（次偏好），返回新序列或 None。"""
    aa = prot[i]
    cur = seq[i * 3:i * 3 + 3]
    alts = [c for c in sorted(syn.get(aa, []), key=lambda x: -freq[x]) if c != cur]
    return alts


def _break_homopolymers(seq, prot, syn, freq, k):
    L = list(seq)
    for i in range(len(prot)):
        for _ in range(4):
            s = "".join(L)
            run = re.search(r"(A{%d,}|T{%d,}|G{%d,}|C{%d,})" % (k, k, k, k), s)
            if not run:
                return "".join(L)
            pos = run.start()
            ci = pos // 3
            if ci >= len(prot):
                break
            alts = _retranslate("".join(L), prot, syn, freq, ci)
            if not alts:
                break
            L[ci * 3:ci * 3 + 3] = list(alts[0])
    return "".join(L)


def _kill_sites(seq, prot, syn, freq, rounds=40):
    """清掉限制酶位点。
    ⚠️ 位点可跨密码子边界，只改「起始位置那个密码子」清不掉 ——
       必须尝试【所有与该位点重叠的密码子】。"""
    L = list(seq)
    # ⚠️ 不能「遇到一个清不掉的位点就整体放弃」—— 那会漏掉后面所有位点。
    #    改为：逐个位点尝试，失败的记下来跳过，继续处理其它位点。
    stuck = set()
    for _ in range(rounds):
        cur = "".join(L)
        todo = [(nm, pos) for nm, site in RE_SITES.items() if nm not in stuck
                for pos in [cur.find(site)] if pos >= 0]
        if not todo:
            break
        nm, pos = todo[0]
        fixed = False
        lo, hi = pos // 3, min((pos + 7) // 3, len(prot) - 1)
        for ci in range(lo, hi + 1):
            for alt in _retranslate("".join(L), prot, syn, freq, ci):
                trial = L[:]
                trial[ci * 3:ci * 3 + 3] = list(alt)
                t = "".join(trial)
                if not any(x in t for x in RE_SITES.values()):
                    L = trial; fixed = True; break
            if fixed:
                break
        if not fixed:
            stuck.add(nm)      # 该位点在此处清不掉，跳过它，继续处理别的
    return "".join(L)


def _kill_homopolymers(seq, prot, syn, freq, k=6, rounds=60):
    """清掉 >=k 的同聚物，同样允许改任一重叠密码子。"""
    L = list(seq)
    for _ in range(rounds):
        cur = "".join(L)
        m = re.search(r"(A{%d,}|T{%d,}|G{%d,}|C{%d,})" % (k, k, k, k), cur)
        if not m:
            break
        pos = m.start()
        lo, hi = pos // 3, min((pos + k) // 3, len(prot) - 1)
        fixed = False
        for ci in range(lo, hi + 1):
            for alt in _retranslate("".join(L), prot, syn, freq, ci):
                trial = L[:]
                trial[ci * 3:ci * 3 + 3] = list(alt)
                t = "".join(trial)
                if not re.search(r"(A{%d,}|T{%d,}|G{%d,}|C{%d,})" % (k, k, k, k), t):
                    L = trial; fixed = True; break
            if fixed:
                break
        if not fixed:
            break
    return "".join(L)


def translate(s):
    return "".join(AA_OF.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))


def cai(seq, freq):
    """密码子适应指数（w = f_c / f_max）。"""
    aa_max = {}
    for c in CODONS:
        a = AA_OF[c]
        if a != "*":
            aa_max[a] = max(aa_max.get(a, 0), freq[c])
    w = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3]
        if c in freq and AA_OF[c] != "*" and aa_max.get(AA_OF[c]):
            w.append(freq[c] / aa_max[AA_OF[c]])
    return float(np.exp(np.mean(np.log(w)))) if w else np.nan


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else np.nan


def cpg_oe(s):
    c, g, n = s.count("C"), s.count("G"), len(s)
    return (s.count("CG") * n) / (c * g) if c and g else np.nan


# ────────────────── 发卡设计与脱靶检查 ──────────────────
def design_hairpins(target_cds, other_cds_blob, other_gene_blob, gene, n=5,
                    stem=21, loop="TTCAAGAGA"):
    """在靶基因 CDS 上取 21 nt 作茎，做【基因级】脱靶检查。

    ⚠️ 两个必须避开的坑（首版都踩了）：
      1. 不能把【靶基因自己】的序列算作脱靶 —— 发卡本来就来自它的 CDS；
      2. 种子区不能用 7 nt —— 4^7=16k 种可能，在 60 Mb 的 CDS 里必然到处命中，
         会把所有候选毙掉。改用 12 nt 核心种子（位置 2–13）。
    """
    out = []
    for start in range(0, len(target_cds) - stem, 30):
        s = target_cds[start:start + stem]
        if "N" in s or not (0.30 <= gc(s) <= 0.70):
            continue
        if len(set(s[-4:])) == 1 or len(set(s[:4])) == 1:
            continue
        full_other = other_cds_blob.count(s)          # 其它基因的 CDS 里的精确命中
        seed = s[1:13]
        seed_other = other_cds_blob.count(seed)
        if full_other > 0 or seed_other > 2:
            continue
        anti = s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        out.append(dict(gene=gene, start=start + 1, stem=s,
                        hairpin=s + loop + anti + "TTTTTT",
                        full_hits_other=full_other, seed_hits_other=seed_other,
                        full_hits_self=1))
        if len(out) >= n:
            break
    return out


# ─────────────────────── IRS 打分 ───────────────────────
def irs_score(feats_row, model):
    import importlib.util as u
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "16_validation.py")
    if model.get("_mk") is None:
        sp = u.spec_from_file_location("v16", p)
        m = u.module_from_spec(sp)
        sp.loader.exec_module(m)
        model["_mk"] = m.mk
        from sklearn.linear_model import LogisticRegression
    return model


def score(feature_dict, fcols, pipe):
    """用【主模型】的 9 特征打分；缺失特征按训练集中位数填充并标注。"""
    x = np.array([[feature_dict.get(c, np.nan) for c in fcols]], dtype=float)
    return pipe.predict_proba(x)[:, 1][0]


def main():
    os.makedirs(PKG, exist_ok=True)
    os.makedirs(f"{PKG}/sequences", exist_ok=True)
    pd.set_option("display.width", 200)

    print("═" * 78)
    print("C2/C3：载荷表达盒设计与打分")
    print("═" * 78)

    cds = longest_cds()
    cnt, freq, pref, aa_tot = codon_table(cds)
    print(f"密码子表：本物种 CDS {len(cds)} 条、{sum(cnt.values()):,} 个密码子自算")
    print(f"  终止子偏好：TAA {cnt['TAA']} / TAG {cnt['TAG']} / TGA {cnt['TGA']}")

    # ── 骨架蛋白 ──
    print("\n── 骨架序列 ──")
    # eGFP：从 pEGFP-N1 (U55762) 提 CDS
    def gb_cds(path, label):
        """从 GenBank 记录里提 CDS。
        ⚠️ GenBank 的序列是小写 —— 首版写成 [^ACGT] 把整段滤空了，返回 0 aa。"""
        t = open(path).read()
        org = t.split("ORIGIN")[1].split("//")[0]
        org = re.sub(r"[^ACGT]", "", "".join(org.split("\n")[1:]).upper())
        m = re.search(r"CDS\s+(complement\()?(\d+)\.\.(\d+)", t)
        if not m:
            raise ValueError(f"{label}: 找不到 CDS 特征")
        lo, hi, comp = int(m.group(2)), int(m.group(3)), bool(m.group(1))
        sub = org[lo - 1:hi]
        if comp:
            sub = sub.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        p = translate(sub)
        assert p.startswith("M"), f"{label}: CDS 不以 ATG 起始（前 6 aa = {p[:6]}）"
        assert "*" in p, f"{label}: 无终止子"
        return p.split("*")[0]

    prots, raw = {}, {}
    prots["eGFP"] = gb_cds(f"{BB}/eGFP.gb", "eGFP")
    prots["NanoLuc"] = gb_cds(f"{BB}/NanoLuc.gb", "NanoLuc")
    from_p = {"eGFP": f"{BB}/eGFP.gb", "NanoLuc": f"{BB}/NanoLuc.gb"}


    def cds_dna(path):
        t = open(path).read()
        org = re.sub(r"[^ACGT]", "", "".join(t.split("ORIGIN")[1].split("//")[0].split("\n")[1:]).upper())
        m = re.search(r"CDS\s+(complement\()?(\d+)\.\.(\d+)", t)
        sub = org[int(m.group(2)) - 1:int(m.group(3))]
        return sub.translate(str.maketrans("ACGT", "TGCA"))[::-1] if m.group(1) else sub
    raw = {k: cds_dna(v) for k, v in from_p.items()}
    prots["Cas9"] = "".join(l.strip() for l in open(f"{BB}/cas9.faa") if l[0] != ">").rstrip("*")
    for k, v in prots.items():
        print(f"  {k:9s} {len(v):>5d} aa  → 优化后 CDS 约 {len(v)*3+3} bp")

    # ── 优化 ──
    opt = {}
    for k, p in prots.items():
        s = opt_codon(p, freq, pref, seed=0)
        assert translate(s).rstrip("*") == p.rstrip("*"), f"{k} 翻译不一致"
        opt[k] = s
        print(f"  {k:9s} 优化后 {len(s)} bp  GC={gc(s):.3f}  CAI={cai(s,freq):.3f}")

    # ── 启动子（eF1α）──
    print("\n── 启动子 ──")
    from importlib.util import spec_from_file_location, module_from_spec
    sp = spec_from_file_location("v16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "16_validation.py"))
    v16 = module_from_spec(sp); sp.loader.exec_module(v16)
    prom = read_fa(f"{FEAT}/promoters.fa")
    up1k = prom["h1SMcG0010823"]
    print(f"  eF1α (h1SMcG0010823) 上游 1 kb: GC={gc(up1k):.3f}  CpG o/e={cpg_oe(up1k):.3f}")

    # ── 效应器发卡 ──
    print("\n── 效应器发卡（TBK1 / MRC1）──")
    hair = []
    for gname, gid in (("TBK1", "h1SMcG0013953"), ("MRC1", "h1SMcG0017400")):
        if gid not in cds:
            print(f"  ⚠️ {gname} ({gid}) 无 CDS，跳过")
            continue
        blob = "|".join(v for k, v in cds.items() if k != gid)   # 其它基因的 CDS 拼串
        print(f"  {gname:6s} 脱靶背景：其它基因 CDS 共 {len(blob)/1e6:.1f} Mb")
        hs = design_hairpins(cds[gid], blob, None, gname, n=3)
        print(f"  {gname:6s} CDS {len(cds[gid])} bp → 通过脱靶筛的发卡 {len(hs)} 条")
        for h in hs:
            print(f"      @{h['start']:>6d}  {h['stem']}  其它基因精确命中={h['full_hits_other']} "
                  f"12nt 种子命中={h['seed_hits_other']}")
        hair += hs

    # ── 组装 + QC + IRS 打分 ──
    print("\n── 候选表达盒组装 ──")
    feats = v16.build_features()
    df = v16.load_label("down_loose", feats)
    X, y, fcols, idx = v16.usable(df, verbose=False)
    pipe = v16.mk().fit(X, y)
    train_med = df.loc[idx, fcols].median()

    rows = []
    cands = []

    def qc(seq):
        hp = max((len(mo.group()) for mo in re.finditer(r"(A+|T+|G+|C+)", seq)), default=0)
        return hp, [nm for nm, x in RE_SITES.items() if x in seq]

    # 模型分数的参照系（用于「距离阈值多远」）
    sc_tr = pipe.predict_proba(X)[:, 1]
    P_NEG = np.percentile(sc_tr[y == 0], [50, 90])
    THRESH = 0.5          # 平衡类别的隐式决策边界

    def mk(name, kind, parts, has_orf, note_domain=""):
        """has_orf=False 时【模型无对应特征】—— 不可给 IRS 分数（给了就是中位数填充的假值）。"""
        full = "".join(x[2] for x in parts)
        orf = next((x[2] for x in parts if x[0].startswith("ORF")), "")
        hairpin = next((x[2] for x in parts if "hairpin" in x[0]), "")
        hp_all, sites_all = qc(full)
        hp_syn, sites_syn = qc(orf or hairpin)
        if not has_orf:
            rows.append(dict(候选=name, 类型=kind, IRS适用性="不适用（无 CDS，模型无对应特征）",
                             IRS分数="—", 距阈值="—", 判定长度="—", 表达盒全长=len(full),
                             GC=f"{gc(full):.3f}", 适用域="—",
                             合成部分最长同聚物=hp_syn,
                             合成部分残留酶切位点=(",".join(sites_syn) or "无"),
                             天然启动子带入的酶切位点=(",".join(
                                 [x for x in sites_all if x not in sites_syn]) or "无"),
                             Gibson同源臂="两端各 25 bp，序列取决于骨架载体（未指定）"))
            cands.append((name, parts, full))
            print(f"  {name:16s} {kind:6s} 全长={len(full):>5d}  IRS=— (无 CDS，模型无对应特征)"
                  f"    合成部分[同聚物{hp_syn} 位点{rows[-1]['合成部分残留酶切位点']}]")
            return
        fv = dict(zip(fcols, train_med.values))       # 不可映射的特征用训练集中位数
        fv["prom_gc"] = gc(up1k); fv["prom_cpg_oe"] = cpg_oe(up1k)
        fv["cds_gc"] = gc(orf); fv["cds_cpg_oe"] = cpg_oe(orf); fv["cai"] = cai(orf, freq)
        L = len(orf)
        in_dom = 697 <= L <= 13615
        if not in_dom:
            rows.append(dict(候选=name, 类型=kind,
                             IRS适用性="不适用（出域）", IRS分数="—", 距阈值="—",
                             判定长度=L, 表达盒全长=len(full), GC=f"{gc(full):.3f}",
                             适用域=f"出域（ORF {L} bp < 下界 697）",
                             合成部分最长同聚物=hp_syn,
                             合成部分残留酶切位点=(",".join(sites_syn) or "无"),
                             天然启动子带入的酶切位点=(",".join(
                                 [x for x in sites_all if x not in sites_syn]) or "无"),
                             Gibson同源臂="两端各 25 bp，序列取决于骨架载体（未指定）"))
            cands.append((name, parts, full))
            print(f"  {name:16s} {kind:6s} 全长={len(full):>5d}  **出域，IRS 不适用**"
                  f"（ORF {L} bp < 697）")
            return
        scv = pipe.predict_proba(np.array([[fv[c] for c in fcols]], float))[:, 1][0]
        pct = float((sc_tr < scv).mean() * 100)
        rows.append(dict(候选=name, 类型=kind, IRS适用性="适用", IRS分数=f"{scv:.3f}",
                         距阈值=f"低于 0.5 共 {0.5-scv:.3f}；位于训练集第 {pct:.0f} 百分位",
                         判定长度=L, 表达盒全长=len(full), GC=f"{gc(full):.3f}",
                         适用域="在域内",
                         合成部分最长同聚物=hp_syn,
                         合成部分残留酶切位点=(",".join(sites_syn) or "无"),
                         天然启动子带入的酶切位点=(",".join(
                             [x for x in sites_all if x not in sites_syn]) or "无"),
                         Gibson同源臂="两端各 25 bp，序列取决于骨架载体（未指定）"))
        cands.append((name, parts, full))
        print(f"  {name:16s} {kind:6s} 全长={len(full):>5d}  IRS={scv:.3f}"
              f"（低于阈值 {0.5-scv:.3f}，训练集第 {pct:.0f} 百分位）"
              f"    合成部分[同聚物{hp_syn} 位点{rows[-1]['合成部分残留酶切位点']}]")

    def parts_for(orf_name=None, hair=None, hair_idx=None, raw_orf=None):
        P = [("promoter_eF1a_1kb", "h1SMcG0010823 上游 1 kb", up1k)]
        if orf_name:
            P.append((f"ORF_{orf_name}_SmedOpt", f"{orf_name}（涡虫密码子重编码）", opt[orf_name]))
        if raw_orf:
            P.append((f"ORF_{raw_orf}_native", f"{raw_orf} 天然密码子（对照臂）", raw[raw_orf]))
        if hair:
            P.append((f"effector_{hair}_hairpin_{hair_idx}", f"{hair} CDS @{hair_idx}bp", 
                      next(x["hairpin"] for x in hairpins if x["gene"] == hair and x["start"] == hair_idx)))
        return P

    hairpins = hair
    print("\n── 候选 1–3：报告/编辑载荷（eF1α 启动子 + 重编码 ORF）──")
    mk("pIE-eGFP", "报告", parts_for(orf_name="eGFP"), True)
    mk("pIE-Cas9", "编辑", parts_for(orf_name="Cas9"), True)

    print("\n── 外部锚点臂（保留，但 IRS 不适用）──")
    print("  理由：NanoLuc 是【唯一】有外部定量锚点的候选（Drees 2026：TBK1(RNAi) → 4–10×）。")
    print("        其 ORF 513 bp 低于适用域下界 → 出域。**出域本身就是要写出来的结论：IRS 对短载荷无话可说。**")
    mk("pIE-NanoLuc", "报告·锚点", parts_for(orf_name="NanoLuc"), True)

    print("\n── 候选 4–5：效应载荷（eF1α 启动子 + 发卡）──")
    print("  ⚠️ 按 G5：【不采用 2A/IRES 双顺反子】—— 其在涡虫中未经验证。")
    print("     效应载荷为【独立构建体】，与报告载荷共注射。")
    print("  ⚠️ 发卡构建体【无 CDS → 模型无对应特征 → IRS 不适用】。")
    print("     首版给它们算了 IRS=0.137，两条完全相同 —— 那是训练集中位数填充出来的假值，已撤销。")
    for gname in ("TBK1", "MRC1"):
        hs = [x for x in hairpins if x["gene"] == gname][:1]
        for h in hs:
            mk(f"pIE-{gname}hp", "效应", parts_for(hair=gname, hair_idx=h["start"]), False)

    print("\n── 基线对照臂（M2′ 要求的「两套待排序序列」之②）──")
    print("  有 CDS 的候选：补【天然密码子版本】作对照，用于归因「表型差异来自设计还是来自序列本身」。")
    print("  无 CDS 的发卡候选：对照 = 空载体（无发卡），因此处未指定骨架，仅登记该要求。")
    mk("pIE-eGFP-nat", "对照", parts_for(raw_orf="eGFP"), True)
    mk("pIE-NanoLuc-nat", "对照", parts_for(raw_orf="NanoLuc"), True)

    # ── 写出 ──
    with open(f"{PKG}/sequences/candidates_v1.fa", "w") as w:
        for name, parts, full, *_ in cands:
            w.write(f">{name}\n")
            for i in range(0, len(full), 70):
                w.write(full[i:i + 70] + "\n")
    pr = []
    for name, parts, full, *_ in cands:
        off = 0
        for pname, src, s in parts:
            if s:
                pr.append(dict(候选=name, 元件=pname, 来源=src, 起=off + 1, 止=off + len(s), 长度=len(s)))
                off += len(s)
    pd.DataFrame(pr).to_csv(f"{PKG}/parts_table.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(f"{PKG}/candidates_v1.tsv", sep="\t", index=False)
    print(f"\n写出 {PKG}/sequences/candidates_v1.fa  ({len(cands)} 条)")
    print(f"写出 {PKG}/parts_table.tsv  ({len(pr)} 行元件坐标)")
    print(f"写出 {PKG}/candidates_v1.tsv")


if __name__ == "__main__":
    main()
