#!/usr/bin/env bash
# 63_utr_seq.sh —— 取 UTR 序列 + RNAfold
#
# 产出
#   ref/utr5.fa  ref/utr3.fa      每个基因一条（多段 UTR 按转录方向拼接）
#   ref/utr5_mfe.tsv              gene_id / mfe（kcal/mol）
#
# ⚠️ 基因组必须从 /dev/shm 读。网络盘上 getfasta 实测 400 条/分钟（T-02 踩过）。
# ⚠️ getfasta 必须 -s：反义链基因不做反向互补的话，GC 恰好不变、CpG o/e 与 MFE 全错。
set -euo pipefail
R=${IETX_T03:?set IETX_T03}
E=${IETX_ENV:-env/bin}
THREADS=${THREADS:-20}
cd "$R"

[ -s /dev/shm/genome.fa ] && [ -s /dev/shm/genome.fa.fai ] \
  || { echo "!! /dev/shm 没有基因组副本，先拷过去"; exit 1; }

for K in utr5 utr3; do
  [ -s ref/$K.raw.fa ] || \
    "$E/bedtools" getfasta -fi /dev/shm/genome.fa -bed ref/$K.bed -s -name \
       -fo ref/$K.raw.fa 2>/dev/null
  echo "  $K 原始片段 $(grep -c '^>' ref/$K.raw.fa)"
done

# 多段 UTR 按段序号拼接（getfasta 头：>GENE|2::chr:start-end(-)）
"$E/python" - <<'PY'
import collections, re
for k in ("utr5", "utr3"):
    seg = collections.defaultdict(dict)
    name = None
    for ln in open(f"ref/{k}.raw.fa"):
        if ln[0] == ">":
            name = ln[1:].split("::")[0].strip()
        else:
            g, i = name.rsplit("|", 1)
            seg[g][int(i)] = ln.strip().upper()
    with open(f"ref/{k}.fa", "w") as fh:
        for g, d in seg.items():
            s = "".join(d[i] for i in sorted(d))
            s = re.sub(r"[^ACGT]", "", s)
            fh.write(f">{g}\n{s}\n")
    print(f"  {k}.fa: {len(seg)} 条")
PY

# RNAfold —— 空跑不报错，先验条数
n=$(grep -c '^>' ref/utr5.fa)
[ "$n" -gt 1000 ] || { echo "!! utr5.fa 只有 $n 条，停止"; exit 1; }
if [ ! -s ref/utr5_mfe.tsv ]; then
  echo "[$(date +%H:%M:%S)] RNAfold 开始，$n 条"
  "$E/RNAfold" --noPS --jobs="$THREADS" ref/utr5.fa > ref/utr5.fold 2> logs/utr5_rnafold.log
  # 第二行末尾括号里是 MFE；⚠️ 必须锚定行尾、要求「数字.数字」，
  #    否则会匹配到结构串内部的 "(.....)"（T-02 踩过，第二列全是点串）
  awk '/^>/{n=substr($0,2); sub(/::.*/,"",n)}
       /^[.()]+ /{if(match($0,/\(([-0-9]+\.[0-9]+)\)[[:space:]]*$/,m)) print n"\t"m[1]}' \
      ref/utr5.fold > ref/utr5_mfe.tsv
  bad=$(awk -F'\t' '$2 !~ /^-?[0-9]+\.[0-9]+$/' ref/utr5_mfe.tsv | wc -l)
  [ "$bad" -eq 0 ] || { echo "!! mfe.tsv 有 $bad 行第二列不是数字"; exit 1; }
  echo "[$(date +%H:%M:%S)] MFE 完成 $(wc -l < ref/utr5_mfe.tsv) 条"
fi
