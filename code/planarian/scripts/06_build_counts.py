#!/usr/bin/env python3
"""
06_build_counts.py —— tx→gene 汇总，产出基因层面计数矩阵

输入
  ref/transcripts.fa                     转录本序列（头部含 gene_id=）
  ref/salmon_idx/duplicate_clusters.tsv  Salmon 折叠掉的重复序列
  quant/<sample>/quant.sf                41 个样本的定量结果
  results/quant_summary.tsv              样本 QC 总表（用于筛除空样本）

输出
  results/tx2gene.tsv          转录本 → 基因 映射
  results/counts.tsv           基因 × 样本 计数矩阵（有效样本）
  results/counts_tpm.tsv       同上，TPM 版（仅作参考，不做差异分析）
  results/excluded_genes.tsv   因序列折叠而恒为 0 的基因名单（技术性零值）
  results/sample_meta.tsv      样本分组元数据

用法
  $IETX_ROOT/env/bin/python 06_build_counts.py
"""
import os
import re
import sys
import glob
import pandas as pd

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
MIN_READS = 1_000_000      # 有效样本下限：低于此按空样本剔除
MIN_GENE_COUNT = 10        # 基因过滤：至少这么多样本中 count >= 10 才保留


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. tx→gene
def build_tx2gene():
    """从 transcripts.fa 头部解析 gene_id。
    头部形如:
      >h1SMcT0000001.1 CDS=7-594 loc:chr1_h1|38305-38992|+ ... gene_id=h1SMcG0000001;Name=...
    """
    rows = []
    with open(f"{IETX}/ref/transcripts.fa") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            head = line[1:].rstrip("\n")
            tx = head.split()[0]
            m = re.search(r"gene_id=([^;\s]+)", head)
            if not m:
                log(f"  !! 无 gene_id 的转录本: {tx}")
                continue
            rows.append((tx, m.group(1)))
    df = pd.DataFrame(rows, columns=["transcript_id", "gene_id"])
    log(f"  tx2gene: {len(df)} 条转录本, {df.gene_id.nunique()} 个基因")
    return df


# ------------------------------------------------- 2. 被 Salmon 折叠的序列
def load_collapsed(tx2gene):
    """Salmon 建索引时折叠了完全相同的序列，这些转录本不会出现在 quant.sf 里。
    若某基因的**全部**转录本都被折叠，该基因在计数矩阵里会恒为 0 ——
    这是技术性零值，不是生物学信号，必须单独列出。"""
    p = f"{IETX}/ref/salmon_idx/duplicate_clusters.tsv"
    if not os.path.exists(p):
        log("  !! 找不到 duplicate_clusters.tsv，跳过")
        return set(), set()
    dup = set()
    with open(p) as fh:
        next(fh, None)                       # 表头
        for line in fh:
            parts = [x.strip() for x in line.rstrip("\n").split("\t")]
            dup.update(x for x in parts[1:] if x)   # 第 0 列是保留的代表序列
    # 折算到基因层面
    by_tx = dict(zip(tx2gene.transcript_id, tx2gene.gene_id))
    g2tx = tx2gene.groupby("gene_id").transcript_id.apply(set).to_dict()
    kept_tx = set(tx2gene.transcript_id) - dup
    kept_genes = {by_tx[t] for t in kept_tx if t in by_tx}
    orphan = {g for g in g2tx if g not in kept_genes}
    log(f"  被折叠的转录本: {len(dup)} 条 → 失去全部转录本的基因: {len(orphan)} 个")
    return orphan, dup


# ------------------------------------------------------- 3. 样本筛选与读取
def pick_samples():
    meta = pd.read_csv(f"{IETX}/results/quant_summary.tsv", sep="\t")
    meta["reads"] = pd.to_numeric(meta["fastp_reads"], errors="coerce").fillna(0).astype(int)
    good = meta[meta["reads"] >= MIN_READS].copy()
    bad = meta[meta["reads"] < MIN_READS]
    # 解析分组
    parsed = good["sample"].str.extract(r"^(eGFP|TBK1)(DNA|Mock|intact)(\d+h)?(\d)$")
    parsed.columns = ["rnai", "treatment", "timepoint", "rep"]
    parsed["timepoint"] = parsed["timepoint"].fillna("intact")
    good = pd.concat([good.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    good["group"] = good["rnai"] + "_" + good["treatment"] + "_" + good["timepoint"]
    log(f"  有效样本: {len(good)}   剔除: {len(bad)}")
    for s in bad["sample"]:
        log(f"    ✗ {s}")
    return good


def read_quant(sample):
    p = f"{IETX}/quant/{sample}/quant.sf"
    df = pd.read_csv(p, sep="\t", usecols=["Name", "NumReads", "TPM"])
    return df.set_index("Name")


# ------------------------------------------------------------------- main
def main():
    log("════ tx→gene 汇总 ════")
    os.makedirs(f"{IETX}/results", exist_ok=True)

    tx2gene = build_tx2gene()
    orphan, dup_tx = load_collapsed(tx2gene)
    meta = pick_samples()

    cnt, tpm = {}, {}
    for s in meta["sample"]:
        q = read_quant(s)
        # how="right"：以 tx2gene 为准。被 Salmon 折叠掉的转录本不在 quant.sf 里，
        # 右连接后会得到 NaN → 汇总为 0（这正是要单独标记的技术性零值）
        j = q.join(tx2gene.set_index("transcript_id"), how="right")
        cnt[s] = j.groupby("gene_id")["NumReads"].sum(min_count=1)
        tpm[s] = j.groupby("gene_id")["TPM"].sum(min_count=1)

    counts = pd.DataFrame(cnt).fillna(0).astype(int)
    tpms = pd.DataFrame(tpm).fillna(0)
    log(f"  原始矩阵: {counts.shape[0]} 基因 × {counts.shape[1]} 样本")

    # ---- 基因过滤：低表达基因会拖累 DESeq2 的多重检验校正 ----
    keep = (counts >= 10).sum(axis=1) >= max(3, int(0.2 * counts.shape[1]))
    log(f"  过滤后: {keep.sum()} 基因（至少 20% 样本中 count>=10）")

    ann = pd.DataFrame({"gene_id": counts.index})
    ann["excluded_zero"] = ann.gene_id.isin(orphan)
    counts = counts[keep]
    tpms = tpms.reindex(counts.index).fillna(0)

    # ---- 写出 ----
    tx2gene.to_csv(f"{IETX}/results/tx2gene.tsv", sep="\t", index=False)
    counts.to_csv(f"{IETX}/results/counts.tsv", sep="\t")
    tpms.to_csv(f"{IETX}/results/counts_tpm.tsv", sep="\t")
    ann[ann.excluded_zero].to_csv(f"{IETX}/results/excluded_genes.tsv",
                                  sep="\t", index=False)
    meta[["sample", "rnai", "treatment", "timepoint", "rep", "group",
          "reads", "salmon_rate"]].to_csv(f"{IETX}/results/sample_meta.tsv",
                                          sep="\t", index=False)

    n_zero = int(ann[ann.excluded_zero].shape[0])
    log(f"\n  写出:")
    log(f"    counts.tsv          {counts.shape[0]} 基因 × {counts.shape[1]} 样本")
    log(f"    tx2gene.tsv         {len(tx2gene)} 条")
    log(f"    excluded_genes.tsv  {n_zero} 个技术性零值基因")
    log(f"    sample_meta.tsv     {len(meta)} 个样本")
    log(f"  总计数 (百万): {counts.values.sum()/1e6:.1f} M reads")
    log("════ 完成 ════")


if __name__ == "__main__":
    main()
