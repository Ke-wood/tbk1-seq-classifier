#!/usr/bin/env bash
# =============================================================================
# 34_cross_species_seq.sh —— 跨物种序列统计对比（中心论点的第 4 条证据）
#
# 【要补的】中心论点「跨物种迁移在涡虫上受限」现有 3 条证据：
#     ① 同源注释覆盖仅 40%   ② 预训练基因组 LM 无增益   ③ CpG 不预测 TBK1 依赖程序
#   全是"资源/模型/功能"层面，**缺一条纯序列统计层面**的。本脚本补这一条。
#
# 【⚠️ 依赖外网下载】人/鼠 CDS 取自 Ensembl。**跑不通就直接停，不要用别处凑数** ——
#   换数据源会让"跨物种对比"这句话本身失真。
#
# 【判据】见 判据预注册_outcome15_prereg.md §34：
#   · 三物种**取样量必须相同**，并报出实际取样数
#   · 必须写明涡虫侧只覆盖注释到的部分（覆盖率约 40%），**不等于全基因组**
#   · 预注册方向性预测：涡虫 CpG O/E 应**低于**人/鼠。**若相反，如实报告**
#
# 用法: IETX_ROOT=/path/to/工作目录 bash 34_cross_species_seq.sh 2>&1 | tee 34.log
#      （可加 IETX_PROXY=http://127.0.0.1:7890 指定代理）
# =============================================================================
set -uo pipefail

IETX_ROOT="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY="${IETX_PROXY:-http://127.0.0.1:7890}"
EXT="${IETX_ROOT}/external"
LOG="${IETX_ROOT}/logs/34_cross_species.log"
RELEASE="${ENSEMBL_RELEASE:-112}"
mkdir -p "$EXT" "${IETX_ROOT}/logs"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== 34 · 跨物种序列统计对比 ===="
log "Ensembl release = $RELEASE   代理 = $PROXY"

URL_H="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz"
URL_M="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz"

# 先试直连，再试代理 —— 本项目历史上 zenodo 被 DNS 黑洞过，ensembl 未必
try_get(){
  local url="$1" out="$2"
  [ -s "$out" ] && { log "  已存在，跳过 $out"; return 0; }
  log "  下载 $(basename "$out") …"
  if curl -fsSL --max-time 1800 "$url" -o "${out}.part"; then
    mv "${out}.part" "$out"; return 0
  fi
  log "  直连失败，改走代理 …"
  if curl -fsSL --max-time 1800 --proxy "$PROXY" "$url" -o "${out}.part"; then
    mv "${out}.part" "$out"; return 0
  fi
  rm -f "${out}.part"
  return 1
}

if ! try_get "$URL_H" "$EXT/Homo_sapiens.cds.fa.gz"; then
  log "!! 人 CDS 下载失败 —— **停手**。跨物种对比换数据源会失真，不得用别处凑数。"
  exit 1
fi
if ! try_get "$URL_M" "$EXT/Mus_musculus.cds.fa.gz"; then
  log "!! 鼠 CDS 下载失败 —— **停手**。"
  exit 1
fi

for f in "$EXT/Homo_sapiens.cds.fa.gz" "$EXT/Mus_musculus.cds.fa.gz"; do
  log "  $(basename "$f"): $(du -h "$f" | cut -f1)"
done

[ -f "${IETX_ROOT}/ref/cds.fa" ] || { log "!! 缺 ${IETX_ROOT}/ref/cds.fa（涡虫侧）—— 停手"; exit 1; }

log ""
log "-------- 交给 34_cross_species_stats.py --------"
IETX_ROOT="$IETX_ROOT" "$IETX_ROOT/env/bin/python" "$HERE/34_cross_species_stats.py"
RC=$?
log "退出码 = $RC"
exit "$RC"
