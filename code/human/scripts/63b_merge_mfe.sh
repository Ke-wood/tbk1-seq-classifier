#!/usr/bin/env bash
# 63b_merge_mfe.sh —— 合并 RNAfold 输出，写 ref/utr5_mfe.tsv
#
# 为什么分三批：19,694 条里最后 ~1,900 条（含 >7 kb 的超长 5′UTR）会把
# RNAfold 的 --jobs 并行吃成单线程（长序列排队，实测掉到 100 条/分钟）。
# 于是把未完成的单拎出来，按长度轮转分成 20 桶，20 个进程各跑一桶，
# 最后按基因去重合并。
set -euo pipefail
R=${IETX_T03:?set IETX_T03}
cd "$R"

ls ref/utr_chunks/c*.fold > /dev/null 2>&1 || { echo "!! 缺 ref/utr_chunks/*.fold"; exit 1; }
cat ref/utr5.fold ref/utr_chunks/c*.fold > ref/utr5.all.fold

# ⚠️ 必须锚定行尾、要求「数字.数字」：否则会匹配到结构串内部的 "(.....)"。
# ⚠️ 括号内允许空格：ViennaRNA 对小数值会右对齐补空格，输出形如 "(  0.00)"、
#    "(-0.50)"。不允许空格的话，所有短序列（本例 1,547 条，中位 30 bp）会被静默漏掉。
awk '/^>/{n=substr($0,2); sub(/::.*/,"",n)}
     /^[.()]+ /{ if (match($0,/\([ ]*(-?[0-9]+\.[0-9]+)[ ]*\)[[:space:]]*$/,m)) {
                   gsub(/ /,"",m[1]); print n"\t"m[1] } }' \
    ref/utr5.all.fold | awk '!seen[$1]++' > ref/utr5_mfe.tsv

bad=$(awk -F'\t' '$2 !~ /^-?[0-9]+\.[0-9]+$/' ref/utr5_mfe.tsv | wc -l)
[ "$bad" -eq 0 ] || { echo "!! $bad 行第二列不是数字"; exit 1; }
echo "MFE $(wc -l < ref/utr5_mfe.tsv) 条"
