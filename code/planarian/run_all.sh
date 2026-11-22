#!/usr/bin/env bash
# =============================================================================
# run_all.sh —— IRS 计算分析全流程一键重跑
#
# 用法:
#   export IETX_ROOT=/path/to/工作目录
#   bash run_all.sh            # 全流程（约 5.5 小时）
#   bash run_all.sh --from 13  # 从第 13 步开始
#   bash run_all.sh --only 16  # 只跑第 16 步
#
# 已完成的步骤会自动跳过（各脚本自带幂等判断），重复执行安全。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${IETX_ROOT:?未设置 IETX_ROOT —— 指向工作目录；见 README.md}"
export IETX_ROOT
# ⚠️ 小矩阵上 20 线程 BLAS 是纯开销（单次 fit 0.44 s vs 0.012 s），全程限 1 线程
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# HuggingFace 直连不通，走镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$IETX_ROOT/hf_cache}"

E="$IETX_ROOT/env/bin"
S="$HERE/scripts"
LOG="$IETX_ROOT/logs/run_all.log"
mkdir -p "$IETX_ROOT/logs"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

FROM=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

# 步骤表：编号 | 类型 | 命令 | 说明 | 预计时长
steps=(
"00|sh|00_check_server.sh|环境体检|<5s"
"01|sh|01_setup_env.sh|建 micromamba 环境|15min"
"02|sh|02_download_data.sh|下载 Zenodo 数据|65min"
"03|py|03_verify_data.py|数据体检|<1min"
"04|sh|04_verify_drees.sh|md5 校验|2min"
"04b|sh|04b_build_refs.sh|参考序列与索引|25min"
"05|sh|05_quant_all.sh|fastp + salmon 定量|75min"
"06|py|06_build_counts.py|建计数矩阵|2min"
"07|py|07_qc_plots.py|QC 图与大小因子|<1min"
"08|py|08_deseq2.py|DNA vs Mock 对比|5min"
"09|py|09_tbk1_deseq.py|TBK1 vs eGFP（主）|10min"
"10|sh|10_features_seq.sh|启动子侧快特征|1min"
"10r|sh|10_features_seq.sh rnafold|RNAfold MFE|155min"
"11|py|11_label_validation.py|富集检验 + 串联簇折叠|20min"
"12|sh|12_rrna.sh|读段四分类|3min"
"13|py|13_labels.py|标签构造|1min"
"14|py|14_features_cds.py|CDS 侧特征|2min"
"15|py|15_train.py|建模 + 符号稳健性|6min"
"16|py|16_validation.py|泄漏控制 / 簇级 CI / 置换 / jackknife|10min"
"17|py|17_domain.py|适用域界定与诊断|8min"
"18|py|18_embed.py|第二层表征|25min"
"19|py|19_ceiling.py|有界收尾实验|25min"
)

run_one(){
  local id="$1" typ="$2" cmd="$3" desc="$4" eta="$5"
  log "▶ [$id] $desc  （预计 $eta）"
  local t0=$SECONDS
  if [ "$typ" = "sh" ]; then
    bash "$S" $cmd >>"$LOG" 2>&1
  else
    "$E/python" "$S/$cmd" >>"$LOG" 2>&1
  fi
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "✗ [$id] 失败（exit=$rc）—— 详见 $LOG"
    exit $rc
  fi
  log "✓ [$id] 完成，用时 $(( (SECONDS-t0)/60 ))m$(( (SECONDS-t0)%60 ))s"
}

log "════ IRS 全流程开始  IETX_ROOT=$IETX_ROOT ════"
for s in "${steps[@]}"; do
  IFS='|' read -r id typ cmd desc eta <<<"$s"
  num="${id%[a-z]}"
  if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then continue; fi
  if [ -z "$ONLY" ] && [ "$num" -lt "$FROM" ] 2>/dev/null; then continue; fi
  run_one "$id" "$typ" "$cmd" "$desc" "$eta"
done
log "════ 全流程结束 ════"
