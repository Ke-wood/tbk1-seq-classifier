#!/usr/bin/env bash
# =============================================================================
# 05_quant_all.sh —— Phase B 前半：fastp 质控 + Salmon 定量（全部 41 个样本）
#
# 与建模决策无关：本脚本只产出转录本层面的 quant.sf，
# tx→gene 汇总与 DESeq2 是后续独立步骤。
#
# 用法（务必 tmux / nohup，约 85 分钟）:
#   tmux new -s quant
#   bash 05_quant_all.sh
#
# 幂等：已有 quant.sf 的样本跳过定量，但仍写入汇总表（不会缺行），可安全重跑。
# =============================================================================
set -uo pipefail

IETX="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
DREES="$IETX/drees"
QC="$IETX/qc"
QUANT="$IETX/quant"
REF="$IETX/ref/salmon_idx"
LOG="$IETX/logs/05_quant.log"
SUMMARY="$IETX/results/quant_summary.tsv"
mkdir -p "$QC" "$QUANT" "$IETX/results"

FP="$IETX/env/bin/fastp"
SALMON="$IETX/env/bin/salmon"
FP_THREADS=6
SAL_THREADS=8
# ⚠️ 必须与 06_build_counts.py 的 MIN_READS 保持一致。
#    首版这里没有这个门槛：只要 salmon 出了 quant.sf 就写 OK，
#    于是 93 条读段的 TBK1intact3 和 415 条的 TBK1DNA6h3 都被标成 OK，
#    quant_summary.tsv 里"OK"形同虚设，靠人去发现空样本是不可靠的。
MIN_READS=${MIN_READS:-1000000}

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 状态分档：定量失败 / 定量成功但读段不足 / 正常
stat_of(){ awk -v r="${1:-0}" -v m="$MIN_READS" 'BEGIN{exit !(r+0>=m)}' \
           && echo OK || echo LOW_READS; }

# ---------------------------------------------------------------------------
# 从 fastp JSON + salmon meta_info.json 采集指标
# 输出 5 个制表符分隔字段: reads  q30  libtype  mapped  rate
#
# ⚠️ 一律读 JSON，不解析日志。salmon 把详细日志打到 stderr（未落盘），
#    logs/salmon_quant.log 只有 298 B 摘要，且其输出含 ANSI 转义码。
# ---------------------------------------------------------------------------
collect(){
  python3 - "$QC/$1.fastp.json" "$QUANT/$1/aux_info/meta_info.json" <<'PY'
import json, sys
fp, mi = sys.argv[1], sys.argv[2]
reads = q30 = libtype = mapped = rate = ""
try:
    # ⚠️ fastp 的 JSON 是嵌套的：顶层是 summary / filtering_result /
    #    duplication / adapter_cutting / read1_before_filtering ...，
    #    统计量在 summary.after_filtering 下，**不是**顶层。
    #    写成 d["after_filtering"] 会 KeyError，导致这两列静默变空。
    a = json.load(open(fp))["summary"]["after_filtering"]
    reads = a["total_reads"]
    q30 = "%.4f" % a["q30_rate"]
except Exception:
    pass
try:
    d = json.load(open(mi))
    libtype = d.get("detected_library_type") or (d.get("library_types") or [""])[0]
    mapped = d.get("num_mapped", "")
    rate = "%.4f" % d.get("percent_mapped", 0)
except Exception:
    pass
print("%s\t%s\t%s\t%s\t%s" % (reads, q30, libtype, mapped, rate))
PY
}

log "════ Phase B 定量开始 ════"
printf "sample\tdrees_MB\tfastp_reads\tq30_after\tfastp_sec\tlibtype\tsalmon_mapped\tsalmon_rate\tstatus\n" > "$SUMMARY"

n=0
for f in "$DREES"/*_sequence.txt.gz; do
  [ -s "$f" ] || continue
  n=$((n+1))
  sample=$(basename "$f" | sed 's/.*_lane1//; s/_sequence\.txt\.gz$//')
  fq="$QC/${sample}.fq.gz"
  qd="$QUANT/${sample}"
  mb=$(awk -v b="$(stat -c%s "$f")" 'BEGIN{printf "%.1f", b/1e6}')

  # ---- 幂等：已有结果则跳过定量，但仍记入汇总 ----
  if [ -s "$qd/quant.sf" ] && [ -s "$QC/${sample}.fastp.json" ]; then
    IFS=$'\t' read -r reads q30 libtype mapped rate <<<"$(collect "$sample")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s(cached)\n" \
      "$sample" "$mb" "$reads" "$q30" "-" "$libtype" "$mapped" "$rate" "$(stat_of "$reads")" >> "$SUMMARY"
    log "[$n] $sample —— 已有结果，跳过  libtype=${libtype:-?}  比对率=${rate:-?}%"
    continue
  fi

  log "[$n] $sample  (${mb} MB)"

  # ---- B1: fastp ----
  t0=$(date +%s)
  "$FP" -i "$f" -o "$fq" -q 15 -u 40 -l 30 -w "$FP_THREADS" \
        -j "$QC/${sample}.fastp.json" -h "$QC/${sample}.fastp.html" >>"$LOG" 2>&1
  if [ ! -s "$fq" ]; then
    log "    !! fastp 失败或无输出"
    printf "%s\t%s\t\t\t\t\t\t\tFASTP_FAIL\n" "$sample" "$mb" >> "$SUMMARY"
    continue
  fi
  t1=$(date +%s)

  # ---- B4: salmon quant ----
  # -l A 自动判定；实测 libtype 由 meta_info.json 回读，跑完做一致性审计
  "$SALMON" quant -i "$REF" -l A -r "$fq" -p "$SAL_THREADS" -o "$qd" >>"$LOG" 2>&1
  st=$?
  t2=$(date +%s)

  if [ $st -ne 0 ] || [ ! -s "$qd/quant.sf" ]; then
    log "    !! salmon 失败 (exit=$st)"
    printf "%s\t%s\t\t\t%s\t\t\t\tSALMON_FAIL\n" "$sample" "$mb" "$((t1-t0))" >> "$SUMMARY"
    continue
  fi

  IFS=$'\t' read -r reads q30 libtype mapped rate <<<"$(collect "$sample")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$sample" "$mb" "$reads" "$q30" "$((t1-t0))" "$libtype" "$mapped" "$rate" "$(stat_of "$reads")" >> "$SUMMARY"
  log "    OK  fastp $((t1-t0))s  salmon $((t2-t1))s  libtype=${libtype:-?}  比对率=${rate:-?}%"
done

# ---- 汇总 ----
{
  echo ""
  echo "════ 完成汇总 ════"
  echo "  扫描样本 : $n"
  echo "  OK       : $(tail -n +2 "$SUMMARY" | cut -f9 | grep -c '^OK')"
  echo "  读段不足 : $(tail -n +2 "$SUMMARY" | cut -f9 | grep -c '^LOW_READS')  (< $MIN_READS)"
  echo "  失败     : $(grep -cE 'FAIL' "$SUMMARY" || true)"
  echo ""
  echo "  libtype 分布（关键一致性检查）:"
  tail -n +2 "$SUMMARY" | awk -F'\t' '$6!="" && $6!="-"{c[$6]++} END{for(k in c) printf "    %-6s %d 个样本\n", k, c[k]}'
  echo ""
  echo "  比对率分布:"
  tail -n +2 "$SUMMARY" | awk -F'\t' '$8!=""{s+=$8; m++; if(mn==""||$8+0<mn)mn=$8+0; if($8+0>mx)mx=$8+0} END{if(m) printf "    最低 %.2f%%  平均 %.2f%%  最高 %.2f%%  (n=%d)\n", mn, s/m, mx, m}'
  echo ""
  echo "  明细表: $SUMMARY"
} | tee -a "$LOG"

log "════ Phase B 定量结束 ════"
