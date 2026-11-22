#!/usr/bin/env bash
# run_models.sh —— 只跑模型层（标签已完成），修完第 4 行的 bug 后重算
set -uo pipefail
R=${IETX_HUMAN:?set IETX_HUMAN}
E=${IETX_ENV:-env/bin}
cd "$R"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=$R/logs/models.log

for G in GSE351189 GSE309167; do
  echo "[$(date +%T)] $G pooled" >> "$LOG"
  $E/python scripts/59_model.py "$G" pooled >> "$LOG" 2>&1 || echo "!! $G pooled 失败" >> "$LOG"
  for C in ifna_6h ifng_6h ifna_72h ifng_72h ifna2 ifnb ifnw; do
    [ -s "人类侧_$G/05_label/label_both_$C.tsv" ] || continue
    echo "[$(date +%T)] $G $C" >> "$LOG"
    $E/python scripts/59_model.py "$G" "$C" >> "$LOG" 2>&1 || echo "!! $G $C 失败" >> "$LOG"
  done
done
$E/python scripts/60_compare.py >> "$LOG" 2>&1
echo "DONE $(date '+%F %T')" >> "$LOG"
echo "DONE $(date '+%F %T')" > "$R/logs/STATUS"
