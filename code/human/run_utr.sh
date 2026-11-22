#!/usr/bin/env bash
# run_utr.sh —— T-03 全流程（幂等，可断点续跑）
# 用法: setsid nohup bash scripts/run_utr.sh > /dev/null 2>&1 &
set -uo pipefail
R=${IETX_T03:?set IETX_T03}
E=${IETX_ENV:-env/bin}
cd "$R"
mkdir -p logs
LOG=$R/logs/run_utr.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "1/4 抽 UTR 区间"
[ -s ref/utr5.bed ] || $E/python scripts/62_utr_extract.py >> "$LOG" 2>&1

say "2/4 取序列 + RNAfold"
bash scripts/63_utr_seq.sh >> "$LOG" 2>&1 || { say "!! 63 失败"; exit 1; }

say "3/4 五个特征"
[ -s 03_features/features_utr.tsv ] || $E/python scripts/64_utr_features.py >> "$LOG" 2>&1 \
  || { say "!! 64 失败"; exit 1; }

say "4/4 建模"
$E/python scripts/65_utr_model.py >> "$LOG" 2>&1 || { say "!! 65 失败"; exit 1; }

say "完成"
