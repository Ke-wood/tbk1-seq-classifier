#!/usr/bin/env bash
# 10_features_seq.sh —— §5.4 序列特征提取（第一步：抽序列 + 快特征）
#
# 拆两步跑的原因是「关键路径」：
#   快特征（长度/转录本数/GC/CpG o/e/重复）几分钟出，负类分层匹配马上能用；
#   RNAfold MFE 是长尾（5.9 万条 × 1 kb），放后台并行，不挡主分析。
#
# 产出
#   features/genes.bed            基因坐标（+ 转录本数、基因长度）
#   features/promoters.fa         每个基因上游 1 kb（链感知）
#   features/features_fast.tsv    全基因快特征表
#   features/mfe.tsv              RNAfold MFE（后台，稍后落盘）
#
# 用法: bash 10_features_seq.sh          （前台跑快特征）
#       bash 10_features_seq.sh rnafold  （后台跑 MFE）
set -euo pipefail

IETX=${IETX_ROOT:-}
E="$IETX/env/bin"
GFF="$IETX/annotation/schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3"
FA="$IETX/genome/schMedS3_h1.fa"
F="$IETX/features"
UP=1000
mkdir -p "$F"

# GFF3 属性取值：ID=h1SMcG0000001;gene_id=...;Name=...  —— 按 key= 精确取，不靠正则
ATTR='function attr(s,k,  n,a,i){n=split(s,a,";");for(i=1;i<=n;i++)if(index(a[i],k"=")==1)return substr(a[i],length(k)+2);return ""}'

# ─────────────────────────────────────────────────────────────
# 第二步：RNAfold MFE（后台）
# ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "rnafold" ]; then
  [ -s "$F/promoters.fa" ] || { echo "!! 缺 $F/promoters.fa，先跑前台"; exit 1; }
  echo "[$(date +%H:%M:%S)] RNAfold 开始，$(grep -c '^>' "$F/promoters.fa") 条"
  # --noPS 不画点阵图；-j 并行。输出每条两行：序列行 + 「结构 (MFE)」
  "$E/RNAfold" --noPS --jobs="${JOBS:-8}" "$F/promoters.fa" \
    > "$F/promoters.fold" 2> "$IETX/logs/rnafold.log"
  # 第二行末尾括号里就是 MFE，单位 kcal/mol。
  # ⚠️ 首版写成 /^\\(([-0-9.]+)\\)/ —— 字符类里的 . 是字面点，于是结构串内部的
  #    "(.....)" 先被匹配上，m[1] 全是点串（实测第二列真的是 "....."），
  #    而且 ID 带着 "::chr1:start-end(+)" 后缀 → 下游 join 全落空，MFE 从未进模型。
  #    两处都必须修：正则要求「数字+小数点」且锚定行尾；ID 砍掉坐标后缀。
  awk '/^>/{n=substr($0,2); sub(/::.*/,"",n)}
       /^[.()]+ /{if(match($0,/\(([-0-9]+\.[0-9]+)\)[[:space:]]*$/,m)) print n"\t"m[1]}' \
      "$F/promoters.fold" > "$F/mfe.tsv"
  # 自检 —— 这次的失败是【静默】的：第二列全是点串，join 全落空，下游一个报错都没有，
  # MFE 以全 NaN 的形态被覆盖率过滤器悄悄丢掉。教训：任何 join 上游都要留一个
  # 「值域看着像不像真的」的断言，不能只查行数（行数是对的）。
  bad=$(awk -F'\t' '$2 !~ /^-?[0-9]+\.[0-9]+$/' "$F/mfe.tsv" | wc -l)
  [ "$bad" -eq 0 ] || { echo "!! mfe.tsv 有 $bad 行第二列不是数字 —— 提取失败，勿用"; exit 1; }
  # ID 自检：必须能对上主表，否则下游 join 会再次静默落空
  miss=$(comm -23 <(cut -f1 "$F/mfe.tsv" | sort) <(cut -f1 "$F/features_fast.tsv" | sort) | wc -l)
  [ "$miss" -eq 0 ] || { echo "!! mfe.tsv 有 $miss 个 ID 不在 features_fast.tsv 里"; exit 1; }
  echo "[$(date +%H:%M:%S)] RNAfold 完成 $(wc -l < "$F/mfe.tsv") 条 → $F/mfe.tsv"
  exit 0
fi

# ─────────────────────────────────────────────────────────────
# 第一步：快特征
# ─────────────────────────────────────────────────────────────
echo "── 1/4 解析 GFF3 ──"
# 基因坐标（BED 0-based half-open）；链感知的上游 1 kb 靠 bedtools flank -s
awk -F'\t' -v OFS='\t' "$ATTR"' $3=="gene"{print $1,$4-1,$5,attr($9,"gene_id"),0,$7}' \
    "$GFF" | sort -k4,4 > "$F/genes_raw.bed"
# 转录本数
awk -F'\t' "$ATTR"' $3=="transcript"{print attr($9,"gene_id")}' "$GFF" \
  | sort | uniq -c | awk -v OFS='\t' '{print $2,$1}' | sort -k1,1 > "$F/_ntx.tsv"
echo "  基因 $(wc -l < "$F/genes_raw.bed")   转录本 $(awk '{s+=$2}END{print s}' "$F/_ntx.tsv")"

echo "── 2/4 基因长度 + CDS 长度 ──"
# CDS 长度按基因【求和】（cds.fa 头里带 gene_id=）。
# ⚠️ cds.fa 每个转录本一条记录，故这里得到的是【该基因全部转录本的 CDS 之和】，
#    不是单个 CDS 的长度。多转录本基因会被系统性高估。
#    模型不读这一列（9 个特征里用的是 features_cds.tsv 的 cds_len_longest，那里取【最长】），
#    但该列会出现在 features_fast.tsv 且被 describe() 打印 —— 容易被误读，故在此标注。
#    若要修正为「最长 CDS」，改用 14_features_cds.py 的 cds_len_longest 即可。
awk '/^>/{match($0,/gene_id=[^;]+/); g=substr($0,RSTART+8,RLENGTH-8); next}
     {n[g]+=length($0)} END{for(k in n) print k"\t"n[k]}' "$IETX/ref/cds.fa" \
  | sort -k1,1 > "$F/_cdslen.tsv"
echo "  有 CDS 的基因 $(wc -l < "$F/_cdslen.tsv")"

# 合并成主表
awk -F'\t' -v OFS='\t' '
  NR==FNR{ntx[$1]=$2; next}                       # _ntx.tsv
  FNR==1{ while((getline l < "'"$F"'/_cdslen.tsv")>0){split(l,a,"\t"); cds[a[1]]=a[2]} }
  {print $4, $3-$2, ntx[$4]+0, cds[$4]+0}' \
  "$F/_ntx.tsv" "$F/genes_raw.bed" > "$F/genes.tsv"
# genes.tsv: gene_id  gene_span  n_transcripts  cds_len
{
  echo -e "gene_id\tgene_span\tn_transcripts\tcds_len"
  cat "$F/genes.tsv"
} > "$F/_tmp" && mv "$F/_tmp" "$F/genes.tsv"
mv "$F/genes_raw.bed" "$F/genes.bed"
echo "  写出 features/genes.tsv"

echo "── 3/4 抽上游 1 kb 启动子 ──"
"$E/bedtools" flank -i "$F/genes.bed" -g "$FA.fai" -l "$UP" -r 0 -s \
  | awk -F'\t' -v OFS='\t' '$2<$3' > "$F/promoters.bed"
"$E/bedtools" getfasta -fi "$FA" -bed "$F/promoters.bed" -s -name \
  -fo "$F/promoters.fa" 2>/dev/null
echo "  启动子 $(grep -c '^>' "$F/promoters.fa") 条"

echo "── 4/4 快特征（GC / CpG o/e / 低复杂度重复）──"
"$E/python" - "$F" <<'PY'
import sys, re, numpy as np, pandas as pd
F = sys.argv[1]

def read_fa(path):
    name, buf = None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name: yield name, "".join(buf).upper()
                name, buf = ln[1:].split("::")[0].strip(), []
            else: buf.append(ln.strip())
    if name: yield name, "".join(buf).upper()

def cpg_oe(s):
    """CpG o/e = (N_CG * L) / (N_C * N_G)  —— 标准化到 1.0 为期望值"""
    n = len(s)
    if n < 2: return np.nan
    c, g, cg = s.count("C"), s.count("G"), s.count("CG")
    return (cg * n) / (c * g) if c and g else np.nan

# 低复杂度重复比例：同核苷酸长串 —— 用 2-mer 熵的粗糙代理即可，
# 不必上 dustmasker（那是另一个依赖，且这里只作协变量用）
def lowcomp(s):
    if len(s) < 10: return np.nan
    k = 2
    from collections import Counter
    cnt = Counter(s[i:i+k] for i in range(len(s)-k+1))
    p = np.array(list(cnt.values()), float); p /= p.sum()
    return float(1 + (p * np.log2(p)).sum() / np.log2(16))   # 1=最乱, 0=最单一

rows = []
for name, s in read_fa(f"{F}/promoters.fa"):
    rows.append((name.split("::")[0], len(s), (s.count("G")+s.count("C"))/len(s) if s else np.nan,
                 cpg_oe(s), lowcomp(s)))
prom = pd.DataFrame(rows, columns=["gene_id","prom_len","prom_gc","prom_cpg_oe","prom_lowcomp"])

g = pd.read_csv(f"{F}/genes.tsv", sep="\t")
df = g.merge(prom, on="gene_id", how="left")
df["cds_gc"] = np.nan   # CDS 版 GC/CpG 由 rnafold 步骤之外单独补（见 11_features.py）
df.to_csv(f"{F}/features_fast.tsv", sep="\t", index=False)
print(f"  features_fast.tsv: {len(df)} 基因 × {df.shape[1]} 列")
print(f"  启动子缺失: {df.prom_gc.isna().sum()}（通常为 contig 边界基因）")
print(df[["gene_span","n_transcripts","cds_len","prom_len","prom_gc","prom_cpg_oe","prom_lowcomp"]].describe().T.to_string())
PY

echo "── 完成 ──  下一步（后台）: bash 10_features_seq.sh rnafold"
