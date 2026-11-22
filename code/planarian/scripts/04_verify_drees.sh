#!/usr/bin/env bash
# =============================================================================
# 04_verify_drees.sh —— A4 收尾：等下载跑完 → 逐个 md5 校验 → 产出核对表
#
# 用法（后台跑，下载结束后自动执行）:
#   nohup bash 04_verify_drees.sh > /dev/null 2>&1 &
#
# 校验基准是 Zenodo API 返回的 files[].checksum，不是附属的 .md5sum 文件
# （有 2 条记录缺 .md5sum，但 API 对全部 41 条都给了 md5）。
# =============================================================================
set -uo pipefail

IETX="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
DREES="$IETX/drees"
LOG="$IETX/logs/04_verify_drees.log"
OUT="$IETX/results/drees_check.tsv"
mkdir -p "$IETX/results"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 1. 等下载会话结束 ----
if tmux has-session -t dl 2>/dev/null; then
  log "等待下载会话 'dl' 结束（每 60 s 探一次）…"
  while tmux has-session -t dl 2>/dev/null; do sleep 60; done
fi
log "下载会话已结束，开始校验"

# ---- 2. 逐个 md5 校验 ----
: > "$OUT"
printf "key\tsize_expected\tsize_actual\tmd5_expected\tmd5_actual\tstatus\n" >> "$OUT"
ok=0; bad=0; miss=0; sizebad=0

while IFS=$'\t' read -r key size md5 url; do
  [ -z "${key:-}" ] && continue
  f="$DREES/$key"
  if [ ! -s "$f" ]; then
    printf "%s\t%s\t-\t%s\t-\tMISSING\n" "$key" "$size" "$md5" >> "$OUT"
    miss=$((miss+1)); continue
  fi
  actual_size=$(stat -c%s "$f")
  actual_md5=$(md5sum "$f" | cut -d' ' -f1)
  if [ "$actual_size" != "$size" ]; then
    st="SIZE_MISMATCH"; sizebad=$((sizebad+1))
  elif [ "$actual_md5" != "$md5" ]; then
    st="MD5_MISMATCH"; bad=$((bad+1))
  else
    st="OK"; ok=$((ok+1))
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$key" "$size" "$actual_size" "$md5" "$actual_md5" "$st" >> "$OUT"
done < "$DREES/_drees_gz.tsv"

# ---- 3. 汇总 ----
total=$((ok+bad+miss+sizebad))
{
  log ""
  log "════ A4 校验汇总 ════"
  log "  总数      : $total"
  log "  md5 通过  : $ok"
  log "  md5 不符  : $bad"
  log "  尺寸不符  : $sizebad"
  log "  缺失      : $miss"
  log "  明细      : $OUT"
  log ""
  log "  实测总体积: $(du -sh "$DREES" | cut -f1)"
} | tee -a "$LOG"

[ "$bad" -eq 0 ] && [ "$miss" -eq 0 ] && [ "$sizebad" -eq 0 ] \
  && log "  ✅ A4 全部通过" || log "  ⚠️ A4 有异常项，见明细表"

# ---- 4. 异常样本单独列出（供 Phase B 建模决策参考）----
{
  log ""
  log "════ 样本矩阵缺口（供 Phase B 决策）════"
  for g in eGFPDNA24h eGFPMock24h TBK1DNA24h TBK1Mock24h; do
    n=$(grep -c "lane1${g}[0-9]_sequence" "$DREES/_drees_gz.tsv" 2>/dev/null || echo 0)
    log "  $g  n=$n"
  done
  log ""
  log "  体积异常小（疑似测序失败）:"
  awk -F'\t' '$2 < 20000000 {printf "    %8.2f MB  %s\n", $2/1e6, $1}' "$DREES/_drees_gz.tsv" | sed 's/^/  /' | tee -a "$LOG"
} | tee -a "$LOG"

log ""
log "==== A4 收尾完成 ===="
