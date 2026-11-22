#!/usr/bin/env bash
# =============================================================================
# 04b_build_refs.sh —— 参考序列与索引构建（原流程中缺失的一环）
#
# ⚠️ 【为什么补这个脚本】
#   原流程里下列文件被下游脚本【消费】，却没有任何脚本【生产】它们：
#       ref/transcripts.fa    ← 05_quant_all.sh 的 salmon 索引来源
#       ref/cds.fa            ← 10_features_seq.sh / 14_features_cds.py
#       ref/proteins.fa       ← DIAMOND
#       ref/salmon_idx/       ← 05_quant_all.sh
#       homology/prot.dmnd    ← DIAMOND 库
#       homology/allvall.tsv  ← 16/17/18/19 的同源簇定义
#   它们是探索阶段临时敲的，没落成脚本 —— 结果是「一键重跑」从零跑不通。
#   可追踪性要求补上，故有此脚本。
#
# ⚠️ 【等价但非位级一致】
#   原始 DIAMOND 调用的确切参数【未记录】（logs/ 里只有版本头，无命令行）。
#   本脚本用与下游筛选阈值相容的保守参数，产出的 allvall.tsv 与原始文件
#   **等价但不是位级一致** —— 同源簇边界可能有极小差异。
#   若需完全复现交付报告里的数字，请直接使用随包提供的 allvall.tsv.gz。
#
# 用法: IETX_ROOT=... bash 04b_build_refs.sh
# =============================================================================
set -euo pipefail

IETX="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
E="$IETX/env/bin"
FA="$IETX/genome/schMedS3_h1.fa"
GFF="$IETX/annotation/schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3"
REF="$IETX/ref"
HOM="$IETX/homology"
THREADS="${THREADS:-8}"
mkdir -p "$REF" "$HOM"

echo "── 1/4 gffread 提转录本 / CDS / 蛋白 ──"
# ⚠️ 基因组与注释必须同源（同为 Zenodo 13798866）。混用 NCBI 版会提出 0 条 —— 见 02 脚本注释。
[ -s "$REF/transcripts.fa" ] || \
  "$E/gffread" -g "$FA" -w "$REF/transcripts.fa" -x "$REF/cds.fa" -y "$REF/proteins.fa" "$GFF"
echo "  转录本 $(grep -c '^>' "$REF/transcripts.fa")  CDS $(grep -c '^>' "$REF/cds.fa")  蛋白 $(grep -c '^>' "$REF/proteins.fa")"

echo "── 2/4 salmon 索引（k=31）──"
[ -s "$REF/salmon_idx/info.json" ] || \
  "$E/salmon" index -t "$REF/transcripts.fa" -i "$REF/salmon_idx" -k 31 -p "$THREADS"

echo "── 3/4 DIAMOND 建库 ──"
[ -s "$HOM/prot.dmnd" ] || \
  "$E/diamond" makedb --in "$REF/proteins.fa" -d "$HOM/prot" --threads "$THREADS"

echo "── 4/4 DIAMOND all-vs-all（自比）──"
# 下游 clusters_from_diamond 自己会按 pident≥70 & 比对≥200aa & e≤1e-5 过滤，
# 故此处放宽到 e≤1e-3 保留足够候选，避免在比对阶段就把边砍掉。
[ -s "$HOM/allvall.tsv" ] || \
  "$E/diamond" blastp -q "$REF/proteins.fa" -d "$HOM/prot" \
      -o "$HOM/allvall.tsv" --very-sensitive -e 1e-3 \
      --max-target-seqs 500 --threads "$THREADS"

# 自检：DIAMOND 默认输出 12 列，下游按 12 列解析（曾按 6 列解析导致假巨分量）
bad=$(awk -F'\t' 'NF!=12' "$HOM/allvall.tsv" | head -1)
[ -z "$bad" ] || { echo "!! allvall.tsv 存在非 12 列的行，下游解析会错"; exit 1; }
echo "  allvall.tsv $(wc -l < "$HOM/allvall.tsv") 条命中，列数自检通过"
echo "── 完成 ──"
