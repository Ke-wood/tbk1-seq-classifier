#!/usr/bin/env bash
# =============================================================================
# 02_download_data.sh —— Phase A 取数（针对新服务器重写）
#
# 与旧版的三个关键差异（旧版在本机跑必失败）：
#   1. Zenodo 走代理。本机 zenodo.org 被 DNS 黑洞（解析到 0.0.0.0），
#      但 mihomo 代理 127.0.0.1:7890 通。裸 curl 必失败。
#   2. 必须用 API 返回的 files[].links.self，不能自己拼 URL。
#      record 17072252 是版本化条目，其文件实际托管在 record 19045792，
#      手工拼的 records/17072252/files/... 会 404。
#   3. 不用 PlanMine（铁律 3 禁用 + 本机直连/代理均不通）。
#      注释一律取 Zenodo 13798866。
#
# 用法（务必 tmux / nohup）:
#   tmux new -s dl
#   bash 02_download_data.sh
# =============================================================================
set -uo pipefail   # 不用 -e：允许单个源失败后继续

IETX_ROOT="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
PROXY="${IETX_PROXY:-http://127.0.0.1:7890}"
export https_proxy="$PROXY" http_proxy="$PROXY"

GENOME="$IETX_ROOT/genome"
ANN="$IETX_ROOT/annotation"
DREES="$IETX_ROOT/drees"
LOGS="$IETX_ROOT/logs"
mkdir -p "$GENOME" "$ANN" "$DREES" "$LOGS"
LOG="$LOGS/02_download.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# aria2c：多连接下载器（单连接 0.47 MB/s → 8 连接 45 MB/s）
# 必须用环境里的那份。~/.local/bin/aria2c 是指向 mamba base 的软链，
# 单独 cp 二进制会丢 libaria2.so.0（error while loading shared libraries）。
ARIA="$IETX_ROOT/env/bin/aria2c"
[ -x "$ARIA" ] || { echo "!! 缺 $ARIA，先跑: micromamba install -p \$IETX_ROOT/env -c conda-forge aria2"; exit 1; }
ARIA_OPTS=(-x8 -s8 -k1M -c --max-tries=5 --retry-wait=5 --file-allocation=none
           --all-proxy="$PROXY" --console-log-level=warn --summary-interval=0)

log "==== Phase A 取数开始 ===="
log "IETX_ROOT = $IETX_ROOT"
log "代理      = $PROXY"

# ---------------------------------------------------------------------------
# 通用：拉取 Zenodo 条目清单 -> TSV(key, size, md5, links.self)
# ---------------------------------------------------------------------------
fetch_manifest(){
  local rec="$1" out="$2"
  curl -fsSL "https://zenodo.org/api/records/$rec" -o "${out}.json" || {
    log "!! 条目 $rec 元数据拉取失败"; return 1; }
  python3 - "$out" <<'PY'
import json, sys
base = sys.argv[1]
d = json.load(open(base + ".json", encoding="utf-8"))
with open(base + ".tsv", "w", encoding="utf-8") as fh:
    for f in d.get("files", []):
        url = (f.get("links") or {}).get("self", "")
        if not url:
            continue
        fh.write("%s\t%s\t%s\t%s\n" % (f["key"], f.get("size", 0),
                                       (f.get("checksum") or "").replace("md5:", ""), url))
print("  %d files, %.2f GB" % (len(d.get("files", [])),
      sum(f.get("size", 0) for f in d.get("files", [])) / 1e9))
PY
}

# 通用：按清单下载 + 校验
dl_manifest(){
  local tsv="$1" dest="$2" label="$3"
  local n=0 ok=0 skip=0 fail=0
  while IFS=$'\t' read -r key size md5 url; do
    [ -z "${key:-}" ] && continue
    out="$dest/$key"
    if [ -s "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
      skip=$((skip+1)); continue
    fi
    n=$((n+1))
    log "  [下载 $n] $key  ($(python3 -c "print(f'{$size/1e6:.1f}')") MB)"
    if "$ARIA" "${ARIA_OPTS[@]}" -d "$dest" -o "$key" "$url" >>"$LOG" 2>&1; then
      got=$(stat -c%s "$out" 2>/dev/null || echo 0)
      if [ "$got" = "$size" ]; then ok=$((ok+1))
      else log "    !! 尺寸不符: 期望 $size 实得 $got"; fail=$((fail+1)); fi
    else
      log "    !! 下载失败: $key"; fail=$((fail+1))
    fi
  done < "$tsv"
  log "  [$label] 新下 $ok / 跳过 $skip / 失败 $fail"
  [ "$fail" -eq 0 ]
}

# ===========================================================================
# [1/3] Drees 2026 RNA-seq —— Zenodo 17072252（版本化，文件实际在 19045792）
# ===========================================================================
log ""
log "======== [1/3] Drees 2026 RNA-seq (Zenodo 17072252) ========"
if fetch_manifest 17072252 "$DREES/_manifest" | tee -a "$LOG"; then
  # 只下 .gz 测序数据。Zenodo 每条已带 md5，.md5sum 附属文件冗余，不下。
  grep -P '\.gz\t' "$DREES/_manifest.tsv" > "$DREES/_drees_gz.tsv"
  log "  待下载 .gz: $(wc -l < "$DREES/_drees_gz.tsv") 个"
  dl_manifest "$DREES/_drees_gz.tsv" "$DREES" "Drees" && log "  [OK] Drees 全部到位" \
    || log "  !! Drees 有失败项，重跑本脚本可续传"
fi

# ===========================================================================
# [2/3] 参考基因组 —— Zenodo 13798866 的 schMedS3_h1.fa.gz
#
# ⚠️ 不用 NCBI 的 GCA_045838265.1_schMedS3h1_genomic.fna.gz。A3 校验已证伪：
#      GFF3 seqid 用 Sequence-Name（chr1_h1…），NCBI FASTA 用 GenBank-Accn
#      （CM099442.1…），交集 0，gffread 会提出 0 条转录本。
#      且 NCBI 版缺 GFF3 用到的 scaffold_67_h1（662 条 vs 663 条，差 39,843 bp），
#      所以"重命名/liftover"这条退路也不通。
#      Zenodo 同作者的 FASTA 与 GFF3 seqid 282/282 全匹配，坐标原生兼容。
# 决策日期 2026-09-24，经用户确认。
# ===========================================================================
log ""
log "======== [2/3] 参考基因组 Zenodo schMedS3_h1.fa.gz ========"
if fetch_manifest 13798866 "$ANN/_manifest" | tee -a "$LOG"; then
  # 基因组 + 基因注释 + TE 注释，全部取自同一条目 → 坐标体系自洽
  #   schMedS3_h1.fa                   = 基因组（chr1_h1 命名）
  #   schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3 = 基因注释（gffread 提转录本用）
  #   schMedS3_h1.fa.mod.EDTA.TEanno.gff3       = 转座子注释（§5.4 重复序列占比）
  # ⚠️ 用 awk 精确匹配第 1 列，不用 grep 正则。原因见下：
  #    本机 /usr/bin/grep 是 ugrep 7.8.4（不是 GNU grep），其 -E 模式不把 \t
  #    解释为制表符（-P 才会）。原先写成 grep -E '...\.gz\t' 会静默匹配 0 行，
  #    而 dl_manifest 对空清单返回成功 → 脚本打印「到位」却什么都没下。
  awk -F'\t' '$1=="schMedS3_h1.fa.gz" ||
              $1=="schMedS3_h1_ENCODE_hybrid_agat_hconf.gff3.gz" ||
              $1=="schMedS3_h1.fa.mod.EDTA.TEanno.gff3.gz"' \
    "$ANN/_manifest.tsv" > "$ANN/_ref_pick.tsv"
  log "  选定 $(wc -l < "$ANN/_ref_pick.tsv") 个文件:"
  cut -f1 "$ANN/_ref_pick.tsv" | sed 's/^/    /' | tee -a "$LOG"
  grep -P '\.fa\.gz\t' "$ANN/_ref_pick.tsv"  > "$ANN/_genome_pick.tsv"
  grep -P '\.gff3\.gz\t' "$ANN/_ref_pick.tsv" > "$ANN/_annotation_pick.tsv"
  dl_manifest "$ANN/_genome_pick.tsv" "$GENOME" "基因组" && log "  [OK] 基因组到位" \
    || log "  !! 基因组下载失败"
  dl_manifest "$ANN/_annotation_pick.tsv" "$ANN" "注释" && log "  [OK] 注释到位" \
    || log "  !! 注释有失败项"
fi

# ===========================================================================
# [3/3] 旁证材料 —— NCBI assembly report（不用于分析，仅供 A3 审计留档）
# ===========================================================================
log ""
log "======== [3/3] NCBI assembly_report.txt（审计留档，非分析输入）========"
env -u https_proxy -u http_proxy curl -fsSL \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/045/838/265/GCA_045838265.1_schMedS3h1/GCA_045838265.1_schMedS3h1_assembly_report.txt" \
  -o "$GENOME/assembly_report.txt" 2>>"$LOG" && log "  [OK] assembly_report.txt" \
  || log "  !! 取失败（不影响分析）"

log ""
log "==== 取数结束。下一步: bash 04_verify_phaseA.sh ===="
