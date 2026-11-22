#!/usr/bin/env bash
# run_utr_rest.sh —— RNAfold 收尾之后的三步：合并 MFE → 特征 → 建模
set -uo pipefail
R=${IETX_T03:?set IETX_T03}
E=${IETX_ENV:-env/bin}
cd "$R"
LOG=$R/logs/run_utr.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "合并 MFE"
bash scripts/63b_merge_mfe.sh >> "$LOG" 2>&1 || { say "!! 63b 失败"; exit 1; }

say "五个 UTR 特征"
$E/python scripts/64_utr_features.py >> "$LOG" 2>&1 || { say "!! 64 失败"; exit 1; }

say "建模 M9 / M13 / M14"
$E/python scripts/65_utr_model.py >> "$LOG" 2>&1 || { say "!! 65 失败"; exit 1; }

say "完成"
