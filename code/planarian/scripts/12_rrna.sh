#!/usr/bin/env bash
# 12_rrna.sh —— 读段四分类：rDNA / 外显子 / 内含子 / 基因间（任务书 §6 加分项）
#
# ⚠️ 首版把「非外显子」一律叫「未注释」，结果 40% —— 那个数字是误导的：
#    基因组比对(minimap2, 94%) 与 Salmon 转录本比对(77%) 本来就不是一回事，
#    Salmon 只认与转录本相容的读段，落在内含子里的读段全被算作「未比对」。
#    故必须把「基因内非外显子(=内含子)」单独拆出来，否则会把内含子滞留
#    误报成「基因组未注释区」。
#
# 分类优先级：rDNA > 外显子 > 内含子 > 基因间
# （rRNA 基因也可能被注释成 gene，先判 rDNA 避免重复计入）
#
# 可重入：已算完的样本跳过，崩溃重跑不浪费。
# 用法: bash 12_rrna.sh
set -euo pipefail

IETX=${IETX_ROOT:-}
E="$IETX/env/bin"
FA="$IETX/genome/schMedS3_h1.fa"
GFF="$IETX/annotation/schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3"
R="$IETX/rRNA"
NREAD=${NREAD:-300000}
THREADS=${THREADS:-4}          # 降到 4：首版 8 线程与 RNAfold 抢内存被杀
SAMPLES=${SAMPLES:-"eGFPDNA24h1 eGFPMock24h1 TBK1DNA24h1 eGFPintact1"}
mkdir -p "$R"

if [ ! -s "$R/rrna_loci.bed" ]; then
  echo "── 1/4 定位 rDNA 位点 ──"
  "$E/minimap2" -ax asm5 -t "$THREADS" --secondary=no "$FA" "$R/rrna_ref.fa" 2>/dev/null \
    | "$E/samtools" view -bS - 2>/dev/null | "$E/bedtools" bamtobed -i - \
    | awk -F'\t' -v OFS='\t' '{print $1,$2,$3,$4}' > "$R/rrna_loci_raw.bed"
  # ⚠️ rrna_ref.fa 里混了线粒体基因组(NC_022448) —— 它落在 scaffold_67_h1 的 26 kb 上，
  #    占「rDNA」命中的 99%。线粒体转录本本来就极其丰富，混进来会把 mt 误报成 rRNA。
  #    必须拆开。
  awk -F'\t' -v OFS='\t' '$4 ~ /^NC_022448/ {print $1,$2,$3}' "$R/rrna_loci_raw.bed" \
    | sort -k1,1 -k2,2n | "$E/bedtools" merge -i - > "$R/mt_loci.bed"
  awk -F'\t' -v OFS='\t' '$4 !~ /^NC_022448/ {print $1,$2,$3}' "$R/rrna_loci_raw.bed" \
    | sort -k1,1 -k2,2n | "$E/bedtools" merge -i - > "$R/rrna_loci.bed"
else
  echo "── 1/4 rDNA 位点已建，跳过 ──"
fi
# ⚠️ 每个 BED 单独判存在：早先写成「rrna_loci 或 exons 任一存在就整体跳过」，
#    结果 exons.bed 已有时 genes_span.bed 从未生成，第 3 步直接崩。
echo "── 2/4 建外显子 / 基因跨度 BED ──"
[ -s "$R/exons.bed" ] || awk -F'\t' -v OFS='\t' '$3=="exon"{print $1,$4-1,$5}' "$GFF" \
  | sort -k1,1 -k2,2n | "$E/bedtools" merge -i - > "$R/exons.bed"
[ -s "$R/genes_span.bed" ] || awk -F'\t' -v OFS='\t' '$3=="gene"{print $1,$4-1,$5}' "$GFF" \
  | sort -k1,1 -k2,2n | "$E/bedtools" merge -i - > "$R/genes_span.bed"
echo "  rDNA $(wc -l < "$R/rrna_loci.bed") 段 / 外显子 $(wc -l < "$R/exons.bed") 段 / 基因跨度 $(wc -l < "$R/genes_span.bed") 段"

echo "── 3/4 逐样本分类 ──"
[ -s "$R/read_class.tsv" ] || printf "sample\ttotal\tmapped\trRNA\tmtDNA\texon\tintron\tintergenic\n" > "$R/read_class.tsv"
for S in $SAMPLES; do
  if awk -F'\t' -v s="$S" '$1==s{f=1} END{exit !f}' "$R/read_class.tsv"; then
    echo "  跳过 $S（已有结果）"; continue
  fi
  F="$IETX/qc/$S.fq.gz"
  [ -s "$F" ] || { echo "  跳过 $S（无输入）"; continue; }
  # 复用时 .fq 已被删，故 TOT 不能靠 .fq 行数算 —— 直接用抽样上限
  if [ -s "$R/_$S.bed" ]; then
    TOT="$NREAD"
  else
    "$E/seqkit" head -n "$NREAD" "$F" > "$R/_$S.fq" 2>/dev/null
    TOT=$(awk 'END{print NR/4}' "$R/_$S.fq")
  fi
  if [ -s "$R/_$S.bed" ]; then
    echo "    复用已有读段 BED（跳过重比对）"
  else
    "$E/minimap2" -ax sr -t "$THREADS" --secondary=no "$FA" "$R/_$S.fq" 2>/dev/null \
      | "$E/samtools" view -bS - 2>/dev/null \
      | "$E/bedtools" bamtobed -i - | cut -f1-3 > "$R/_$S.bed"
  fi
  MAP=$(wc -l < "$R/_$S.bed")
  bedtools_i(){ "$E/bedtools" intersect "$@" 2>/dev/null; }
  bedtools_i -u -a "$R/_$S.bed" -b "$R/rrna_loci.bed"                 > "$R/_$S.rrna"
  bedtools_i -u -a "$R/_$S.bed" -b "$R/mt_loci.bed"                   > "$R/_$S.mt"
  bedtools_i -v -a "$R/_$S.bed" -b "$R/rrna_loci.bed" \
    | bedtools_i -v -a - -b "$R/mt_loci.bed" \
    | bedtools_i -u -a - -b "$R/exons.bed"                            > "$R/_$S.exon"
  bedtools_i -v -a "$R/_$S.bed" -b "$R/rrna_loci.bed" \
    | bedtools_i -v -a - -b "$R/mt_loci.bed" \
    | bedtools_i -v -a - -b "$R/exons.bed" \
    | bedtools_i -u -a - -b "$R/genes_span.bed"                       > "$R/_$S.intron"
  NR_=$(wc -l < "$R/_$S.rrna"); NM=$(wc -l < "$R/_$S.mt")
  NE=$(wc -l < "$R/_$S.exon"); NI=$(wc -l < "$R/_$S.intron")
  NG=$((MAP-NR_-NM-NE-NI))
  printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" "$S" "$TOT" "$MAP" "$NR_" "$NM" "$NE" "$NI" "$NG" >> "$R/read_class.tsv"
  awk -v s="$S" -v t="$TOT" -v m="$MAP" -v r="$NR_" -v x="$NM" -v e="$NE" -v i="$NI" -v g="$NG" 'BEGIN{
    printf "  %-14s 比对 %6.1f%%  rRNA %5.2f%%  线粒体 %5.2f%%  外显子 %5.2f%%  内含子 %5.2f%%  基因间 %5.2f%%\n",
           s, m*100/t, r*100/t, x*100/t, e*100/t, i*100/t, g*100/t}'
  rm -f "$R/_$S.fq"
done

echo "── 4/4 汇总 ──"
"$E/python" - "$R/read_class.tsv" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], sep="\t")
t = d[["total","mapped","rRNA","mtDNA","exon","intron","intergenic"]].sum()
print(f"\n  合计 {int(t.total):,} 读段（{len(d)} 样本）")
for k, lab in [("mapped","比对到基因组"),("rRNA","真 rRNA 位点"),
               ("mtDNA","线粒体基因组"),("exon","外显子"),
               ("intron","内含子"),("intergenic","基因间")]:
    print(f"    {lab:12s} {int(t[k]):>10,}  占总 {t[k]/t.total*100:>6.2f}%   "
          f"占已比对 {t[k]/t.mapped*100:>6.2f}%")
print(f"\n  关键对读：")
print(f"    · 真 rRNA 仅 {t.rRNA/t.total*100:.2f}%，未比对读段【不是】rRNA 造成的")
print(f"    · 线粒体 {t.mtDNA/t.total*100:.1f}% —— 全基因组匀浆里 mt 转录本本来就多，"
      f"且带 polyA 故逃过 polyA 富集，属预期")
print(f"    · 基因内合计 {(t.exon+t.intron)/t.mapped*100:.1f}%，"
      f"基因间仅 {t.intergenic/t.mapped*100:.1f}%")
print(f"    · Salmon 的 ~23% 未比对 ≈ 内含子滞留 + 注释外转录本，"
      f"而非「基因组未注释区」")
PY
echo "── 完成 ──  $R/read_class.tsv"
