#!/usr/bin/env bash
# 55_build_ref.sh —— 人类侧参考与注释（T-02 第 1 步）
#
# 锁定单套注释：Ensembl release-110 GRCh38（≡ GENCODE v44），基因组与 GTF 同源。
# 产出
#   ref/genome.fa / .fai
#   ref/genes.bed            基因（0-based half-open，带链）
#   ref/n_tx.tsv             每基因转录本数
#   ref/cds.fa / proteins.fa gffread 从【同一套】注释+基因组提取
#   ref/promoters.bed/.fa    链感知上游 1 kb
#   ref/mfe.tsv              RNAfold MFE
#   homology/prot.dmnd, allvall.tsv    DIAMOND 全对全（与涡虫侧同参数）
set -euo pipefail

R=${IETX_HUMAN:?set IETX_HUMAN}
E=${IETX_ENV:-env/bin}
THREADS=${THREADS:-20}
UP=1000
mkdir -p "$R/ref" "$R/homology" "$R/logs"
cd "$R/ref"

# ── 1/6 解压 + 索引 ───────────────────────────────────────────
# ⚠️ 先验 gz 完整性。这条线路断过好几次，半截的 .gz 会被 gunzip 解出一半基因组，
#    后面 getfasta 对缺失的染色体静默返回空序列（不报错）。
if [ ! -s genome.fa ]; then
  gzip -t Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
    || { echo "!! 基因组 .gz 校验失败（下载不完整），先跑 get_genome.sh"; exit 2; }
  gunzip -c Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz > genome.fa
fi
[ -s genome.fa.fai ] || "$E/samtools" faidx genome.fa
# ⚠️ 不要 grep -c '^>' genome.fa —— 3.1 GB 在网络盘上、机器负载 180 时要几分钟，
#    每次重启都白等一次。.fai 一行一条序列，瞬间出数。
echo "  基因组 $(wc -l < genome.fa.fai) 条序列"

# ── 2/6 解析 GTF（Ensembl 格式：key "value";）────────────────────
# ⚠️ Ensembl GTF 不是 GFF3：属性是 `gene_id "ENSG...";` 而不是 `gene_id=...;`。
#    照抄涡虫侧 GFF3 的解析函数会取到空字符串 —— 那样后面 getfasta 全空、
#    RNAfold 空跑、join 全落空，而且【一条报错都没有】。
# ⚠️ awk 不能直读 .gz —— 必须 zcat 管道（首版直接喂 .gz，genes.bed 得到 0 行，
#    而且 exit 0 没有报错。下面的空表断言就是为这个坑留的。）
ATTR='function attr(s,k,  p,q,pat){pat=k" \"";p=index(s,pat);if(!p)return "";q=index(substr(s,p+length(pat)),"\"");return substr(s,p+length(pat),q-1)}'
[ -s annotation.gtf ] || zcat Homo_sapiens.GRCh38.110.gtf.gz > annotation.gtf
[ -s genes.bed ] || \
  awk -F'\t' -v OFS='\t' "$ATTR"' $3=="gene"{print $1,$4-1,$5,attr($9,"gene_id"),0,$7}' \
    annotation.gtf | sort -k4,4 > genes.bed
[ -s n_tx.tsv ] || \
  awk -F'\t' "$ATTR"' $3=="transcript"{print attr($9,"gene_id")}' annotation.gtf \
  | sort | uniq -c | awk -v OFS='\t' '{print $2,$1}' | sort -k1,1 > n_tx.tsv
echo "  基因 $(wc -l < genes.bed)   转录本 $(awk '{s+=$2}END{print s}' n_tx.tsv)"
[ -s genes.bed ] || { echo "!! genes.bed 为空 —— GTF 属性解析失败"; exit 1; }
[ "$(head -1 genes.bed | cut -f4)" != "" ] || { echo "!! gene_id 解析为空"; exit 1; }

# ── 3/6 取 CDS / 蛋白（与基因组同源）────────────────────────────
# ⚠️ gffread 0.12.9 在这台机器上【没编 zlib】：直接喂 .gtf.gz 会把压缩字节当文本解析，
#    刷出几十万行 "Error parsing strand (8) from GFF line" 后产出 0 字节的 cds.fa/proteins.fa。
#    必须先解压成纯文本 GTF。
[ -s annotation.gtf ] || zcat Homo_sapiens.GRCh38.110.gtf.gz > annotation.gtf
if [ ! -s cds.fa ] || [ ! -s proteins.fa ]; then
  "$E/gffread" -g genome.fa -x cds.fa -y proteins.fa annotation.gtf
  echo "  CDS $(grep -c '^>' cds.fa)   蛋白 $(grep -c '^>' proteins.fa)"
fi
[ -s cds.fa ] || { echo "!! cds.fa 为空 —— 基因组与注释不同源？"; exit 1; }

# ── 4/6 启动子（链感知，上游 1 kb）──────────────────────────────
# 与涡虫侧同一算子：gene bed + bedtools flank -s。人类基因有多个 TSS，
# 这里取基因 5' 端最上游的那个（= gene feature 的起/止点），与涡虫侧一致。
# ⚠️ 机器负载高时，getfasta 对 3.1 GB 网络盘上的基因组做随机读，实测慢到
#    400 条/分钟（62,754 条要 3 小时）。若 /dev/shm 里有副本就用副本 ——
#    同一份基因组拷到 tmpfs 后同样的活 1–2 分钟完成。
if [ ! -s promoters.fa ] || [ "$(grep -c '^>' promoters.fa)" -lt 60000 ]; then
  FA_SRC=genome.fa
  if [ -s /dev/shm/genome.fa ] && [ -s /dev/shm/genome.fa.fai ]; then
    FA_SRC=/dev/shm/genome.fa; echo "  （用 /dev/shm 上的基因组副本）"
  fi
  "$E/bedtools" flank -i genes.bed -g "$FA_SRC.fai" -l "$UP" -r 0 -s \
    | awk -F'\t' -v OFS='\t' '$2<$3' > promoters.bed
  "$E/bedtools" getfasta -fi "$FA_SRC" -bed promoters.bed -s -name \
    -fo promoters.fa 2>/dev/null
fi
echo "  启动子 $(grep -c '^>' promoters.fa) 条"

# ── 5/6 RNAfold MFE ──────────────────────────────────────────
if [ ! -s mfe.tsv ]; then
  echo "[$(date +%H:%M:%S)] RNAfold 开始，$(grep -c '^>' promoters.fa) 条"
  "$E/RNAfold" --noPS --jobs="$THREADS" promoters.fa > promoters.fold 2> ../logs/rnafold.log
  awk '/^>/{n=substr($0,2); sub(/::.*/,"",n)}
       /^[.()]+ /{if(match($0,/\(([-0-9]+\.[0-9]+)\)[[:space:]]*$/,m)) print n"\t"m[1]}' \
      promoters.fold > mfe.tsv
  bad=$(awk -F'\t' '$2 !~ /^-?[0-9]+\.[0-9]+$/' mfe.tsv | wc -l)
  [ "$bad" -eq 0 ] || { echo "!! mfe.tsv 有 $bad 行第二列不是数字"; exit 1; }
  miss=$(comm -23 <(cut -f1 mfe.tsv | sort) <(cut -f4 genes.bed | sort) | wc -l)
  [ "$miss" -eq 0 ] || { echo "!! mfe.tsv 有 $miss 个 ID 不在 genes.bed 里"; exit 1; }
  echo "[$(date +%H:%M:%S)] MFE 完成 $(wc -l < mfe.tsv) 条"
fi

# ── 6/6 DIAMOND 全对全（参数照抄涡虫侧 04b_build_refs.sh）───────
if [ ! -s "$R/homology/allvall.tsv" ]; then
  # ⚠️ gffread 对不完整密码子输出 "."（本例 591 个），DIAMOND makedb 直接报
  #    "Invalid character in sequence: '.'" 退出。换成 X（未知残基）即可。
  if [ ! -s proteins.clean.fa ]; then
    awk '/^>/{print; next}{gsub(/\./,"X"); print}' proteins.fa > proteins.clean.fa
    bad=$(awk '!/^>/{for(i=1;i<=length($0);i++){c=substr($0,i,1);
             if(c!~/[ACDEFGHIKLMNPQRSTVWYXBZJUO]/){print; exit}}}' proteins.clean.fa)
    [ -z "$bad" ] || { echo "!! proteins.clean.fa 仍有非法字符"; exit 1; }
  fi
  [ -s "$R/homology/prot.dmnd" ] || \
    "$E/diamond" makedb --in proteins.clean.fa -d "$R/homology/prot" --threads "$THREADS"
  # 下游按 pident≥70 & alnlen≥200 & e≤1e-5 过滤，此处放宽到 1e-3 保留候选
  "$E/diamond" blastp -q proteins.clean.fa -d "$R/homology/prot" \
      -o "$R/homology/allvall.tsv" --very-sensitive -e 1e-3 \
      --max-target-seqs 500 --threads "$THREADS"
  bad=$(awk -F'\t' 'NF!=12' "$R/homology/allvall.tsv" | head -1)
  [ -z "$bad" ] || { echo "!! allvall.tsv 存在非 12 列的行"; exit 1; }
fi
echo "  allvall.tsv $(wc -l < "$R/homology/allvall.tsv") 条命中"

# 转录本 → 基因（16_validation.read_avs 要读 $IETX/results/tx2gene.tsv）
# ⚠️ 键必须与 DIAMOND 输出里的 qseqid 完全一致。实测 gffread 0.12.9 的 -x/-y 头就是
#    裸转录本 ID（`>ENST00000378513`），【不带版本、也不带 gene_id】。GTF 里的
#    transcript_id 也正好不带版本，两边天然对齐 —— 所以从 GTF 建，不要从 FASTA 头建。
mkdir -p "$R/results"
# ⚠️ 必须带表头：16_validation.read_avs 用 pd.read_csv(...).set_index("transcript_id")
#    按列名取，无表头会直接 KeyError。
if [ ! -s "$R/results/tx2gene.tsv" ]; then
  { echo -e "transcript_id\tgene_id"
    awk -F'\t' "$ATTR"' $3=="transcript"{print attr($9,"transcript_id")"\t"attr($9,"gene_id")}' \
      annotation.gtf
  } > "$R/results/tx2gene.tsv"
fi
[ "$(wc -l < "$R/results/tx2gene.tsv")" -gt 10000 ] \
  || { echo "!! tx2gene 解析失败（行数 $(wc -l < "$R/results/tx2gene.tsv")）"; exit 1; }
head -1 "$R/results/tx2gene.tsv" | grep -q $'\t' || { echo "!! tx2gene 无制表符"; exit 1; }

# 预过滤：只为省内存/读盘时间，阈值与下游 clusters_from_diamond 完全一致
if [ ! -s "$R/homology/allvall_filt.tsv" ]; then
  awk -F'\t' '$3>=70 && $4>=200 && $11<=1e-5' "$R/homology/allvall.tsv" \
    > "$R/homology/allvall_filt.tsv"
fi
echo "  allvall_filt.tsv $(wc -l < "$R/homology/allvall_filt.tsv") 条过阈命中"
echo "── 完成 ──"
