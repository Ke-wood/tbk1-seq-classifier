#!/usr/bin/env python3
"""
27_audit_cds_background.py —— CDS 脱靶背景：28.6 Mb 还是 31.7 Mb？

【为什么要做】同一个量在包里出现了两个值：

    28.6 Mb ← scripts/24_sgrna.py:13,233 ；payload/README.md:121 ；附件1 §发卡段
    31.7 Mb ← 附件1 §sgRNA 段 ；results/sgrna/README.md:133,151,249

  理论 L/4^12（4^12 = 16,777,216）：31.7 Mb → 1.889（文本称"理论 1.9"）
                                    28.6 Mb → 1.705
  而实测 200 条随机 12 nt 均值 = 1.65（泊松 SE≈0.09）
  ⇒ 1.65 距 1.705 只有 0.6 SE，距 1.889 有 2.6 SE。**自证数据偏向 28.6 那一支。**

【本脚本做什么】按 22_payload.py 与 24_sgrna.py **各自的构造方式**各算一遍背景长度，
  在同一背景上重跑计数器自证，并列差。
  **只读，不改任何交付文件。**

【判据】见 判据预注册_outcome15_prereg.md §27。要点：
  · 差 ≤0.1 Mb → 文本有一处是陈旧值，按实测统一
  · 差  >0.1 Mb → **不得擅自统一**，报差异与两边构造
  · 计数器自证不过 → **阻塞项**，全包"0 命中"结论挂起
  · 不许为了让 1.65 对上某一支而改随机种子（种子固定 0，与 24_sgrna.py 一致）

用法: IETX_ROOT=... $IETX_ROOT/env/bin/python 27_audit_cds_background.py
"""
import os
import re

import numpy as np

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
REF = f"{IETX}/ref"
OUT = f"{IETX}/results"

# 交付文本里出现过的两个值
QUOTED = {"28.6（脚本注释 / 发卡段）": 28.6,
          "31.7（附件1 sgRNA 段 / sgrna README）": 31.7}
# 24_sgrna.py 的靶点（TARGETS），用于「其它基因」口径
TARGET_IDS = {
    "h1SMcG0013953": "TBK1",
    "h1SMcG0013999": "PIWI1 / smedwi-1",
}
SEED = 0
N_RAND = 200
K = 12


def cds_longest(path):
    """与 22_payload.longest_cds() / 24_sgrna.py 的循环**逐字同构**：
    每基因取最长 CDS。⚠️ 必须取最长而不是求和 —— 求和会把背景从 ~29 Mb 虚增到 53.6 Mb。"""
    best, gid, buf = {}, None, []
    for ln in open(path):
        if ln[0] == ">":
            if gid and len("".join(buf)) > len(best.get(gid, "")):
                best[gid] = "".join(buf).upper()
            m = re.search(r"gene_id=([^;]+)", ln)
            gid, buf = (m.group(1) if m else None), []
        elif gid:
            buf.append(ln.strip())
    if gid and len("".join(buf)) > len(best.get(gid, "")):
        best[gid] = "".join(buf).upper()
    return {g: re.sub(r"[^ACGT]", "", s.replace("U", "T")) for g, s in best.items()}


def counter_selftest(blob, tag):
    """与 24_sgrna.py 同一段自证：报 0 之前先证明计数器数得出数。"""
    rng = np.random.default_rng(SEED)          # ← 种子固定 0，与交付脚本一致
    rand = "".join(rng.choice(list("ACGT"), N_RAND * K))
    obs = float(np.mean([blob.count(rand[i * K:(i + 1) * K]) for i in range(N_RAND)]))
    theory = len(blob) / 4 ** K
    print(f"    [{tag}] 已知重复串 AAAAAAAAAA 命中 {blob.count('AAAAAAAAAA')}"
          f"；GCCGCCGCCGCC 命中 {blob.count('GCCGCCGCCGCC')}")
    print(f"    [{tag}] {N_RAND} 条随机 {K} nt 均值 {obs:.2f}"
          f"（本背景理论 {theory:.3f}；20 nt 理论 {len(blob) / 4 ** 20:.5f}）")
    return obs, theory


def main():
    print("═" * 78)
    print("27 · CDS 脱靶背景核对：28.6 vs 31.7 Mb")
    print("═" * 78)

    p = f"{REF}/cds.fa"
    if not os.path.exists(p):
        raise SystemExit(f"找不到 {p} —— 需要 04b 产出的 ref/cds.fa")
    cds = cds_longest(p)
    print(f"  {p}: {len(cds)} 个基因有过 CDS")

    # 口径①：24_sgrna.py —— **全部基因** CDS 串接（含靶基因）
    blob_all = "|".join(cds.values())
    # 口径②：22_payload.py —— **其它基因** CDS（排除靶基因自身）
    blob_other = blob_all
    for g in TARGET_IDS:
        if g in cds:
            blob_other = blob_other.replace(cds[g], "|")
    # 参考：求和口径（已知会虚增，仅列作对照）
    blob_sum = sum(len(s) for s in cds.values())

    print()
    print(f"  口径① 全部基因 CDS（24_sgrna.py:247 `'|'.join(cds.values())`）"
          f" = {len(blob_all) / 1e6:.1f} Mb")
    print(f"  口径② 其它基因 CDS（22_payload.py `other_cds_blob`，排除靶基因）"
          f" = {len(blob_other) / 1e6:.1f} Mb")
    print(f"  （对照）逐基因求和（**错误口径**，脚本注释警告过）"
          f" = {blob_sum / 1e6:.1f} Mb")
    tgt_bp = sum(len(cds.get(g, "")) for g in TARGET_IDS)
    print(f"  两口径之差 = 靶基因 CDS {tgt_bp} bp = {tgt_bp / 1e6:.4f} Mb"
          f"  ← 若文本里的 28.6↔31.7 差 3.1 Mb，**不可能**由这个解释")

    print()
    print("  ── 计数器自证（在口径① 上，与 24_sgrna.py 同段）──")
    obs, theory = counter_selftest(blob_all, "全基因")
    print()
    print("  ── 计数器自证（在口径② 上）──")
    obs2, theory2 = counter_selftest(blob_other, "其它基因")

    print()
    print("═" * 78)
    print("判据应用")
    print("═" * 78)
    print(f"  {'文本所载':38s} {'Mb':>7s} {'L/4^12':>9s} {'与实测 1.65 的距离(SE)':>24s}")
    se = np.sqrt(theory) / np.sqrt(N_RAND) if theory else float("nan")
    for tag, mb in QUOTED.items():
        th = mb * 1e6 / 4 ** K
        d = abs(obs - th) / se if se else float("nan")
        print(f"  {tag:38s} {mb:>7.1f} {th:>9.3f} {d:>24.1f}")
    print(f"  {'实测（200 随机 12 nt 均值）':38s} {'':>7s} {obs:>9.3f}")

    ok = True
    if not (0.5 * theory < obs < 2 * theory):
        print("\n  !! 计数器自证未过（口径①）—— **阻塞项**：全包所有『0 命中』结论挂起，立即上报")
        ok = False
    if not (0.5 * theory2 < obs2 < 2 * theory2):
        print("\n  !! 计数器自证未过（口径②）—— 同上")

    print()
    print("  ⚠️ 本脚本**不写入任何文件、不修改交付文本**。")
    print("  ⇒ 请把上面【两个口径的 Mb 值 + 实测均值】回传，由本地端统一三处注释与两处正文。")
    print("  ⇒ **不要**为了让实测值对上某一支而改随机种子 —— 对不上是发现，不是要修的东西。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
