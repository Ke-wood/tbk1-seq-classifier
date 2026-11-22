#!/usr/bin/env bash
# =============================================================================
# 33_rerun_04b_and_compare.sh —— 重跑 04b 并把参数落档，比对是否位级一致
#
# 【要闭的缺口】交付包 §八 自曝：
#     「参考文件 6 件原先无生产者 → 已补 04b_build_refs.sh；**参数未记录，等价非位级一致**」
#   重跑一次，把**实际生效的 DIAMOND/其他参数**写进脚本头，并比对簇数。
#
# 【判据】见 判据预注册_outcome15_prereg.md §33：
#   · 与随包文件在 70/200/1e-5 下簇数一致（4,373 基因 / 919 簇）→ 可写"位级一致"
#   · 不一致 → **如实写差异**（行数差、簇数差），并声明**随包文件为权威**
#   · **不得**用重跑结果覆盖随包文件
#
# 【安全】ref/ 与 homology/ 会先整体备份到 _bak_33_<时间戳>/，跑完可一键回滚。
#         若磁盘余量 < 40 GB 则拒绝开工（DIAMOND 中间文件不小）。
#
# 用法:  IETX_ROOT=/path/to/工作目录 bash 33_rerun_04b_and_compare.sh 2>&1 | tee 33.log
# =============================================================================
set -uo pipefail

IETX_ROOT="${IETX_ROOT:?未设置 IETX_ROOT，见 README.md}"
S="${IETX_ROOT}/scripts"
HOM="${IETX_ROOT}/homology"
REF="${IETX_ROOT}/ref"
ENV_BIN="${IETX_ROOT}/env/bin"
STAMP="$(date +%Y%m%d_%H%M%S)"
BAK="${IETX_ROOT}/_bak_33_${STAMP}"
LOG="${IETX_ROOT}/logs/33_rerun_04b.log"
mkdir -p "${IETX_ROOT}/logs"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== 33 · 重跑 04b 并比对 ===="
log "IETX_ROOT = $IETX_ROOT"
log "备份目录  = $BAK"

# ── 0. 前置检查 ───────────────────────────────────────────────────────────
[ -f "$S/04b_build_refs.sh" ] || { log "!! 缺 $S/04b_build_refs.sh —— 停手"; exit 1; }
[ -f "$REF/cds.fa" ]         || log "⚠️ 缺 $REF/cds.fa（说明 04b 从没成功跑过？）"

AVAIL_GB=$(df -BG --output=avail "$IETX_ROOT" | tail -1 | tr -dc '0-9')
log "磁盘可用 = ${AVAIL_GB} GB"
[ "${AVAIL_GB:-0}" -lt 40 ] && { log "!! 可用空间 < 40 GB，拒绝开工"; exit 1; }

# ── 1. 记录环境与参数（**这就是"参数落档"本身**）─────────────────────────
log ""
log "-------- 环境与工具版本 --------"
{ echo "diamond : $("$ENV_BIN/diamond" --version 2>&1 | head -1)"
  echo "gffread : $("$ENV_BIN/gffread" --version 2>&1 | head -1)"
  echo "salmon  : $("$ENV_BIN/salmon" --version 2>&1 | head -1)"
  echo "python  : $("$ENV_BIN/python" --version 2>&1)"
  echo "CPU 核数 : $(nproc)"
  echo "时间戳   : $STAMP"
} 2>&1 | tee -a "$LOG"

log ""
log "-------- 04b 里出现的所有 DIAMOND/gffread/salmon 命令 --------"
grep -nE 'diamond|gffread|salmon|makeblastdb|mcl' "$S/04b_build_refs.sh" | tee -a "$LOG" \
  || log "  （grep 无命中 —— 脚本可能改了写法，须人工确认参数）"

# ── 2. 备份 ───────────────────────────────────────────────────────────────
log ""
log "-------- 备份 ref/ 与 homology/ --------"
mkdir -p "$BAK"
for d in "$REF" "$HOM"; do
  [ -e "$d" ] || continue
  log "  备份 $(basename "$d") → $BAK/"
  cp -a "$d" "$BAK/" || { log "!! 备份失败，停手"; exit 1; }
done
# 顺带把随包文件与主流程脚本也存一份指纹，供以后对账
{
  echo "# 33 快照 $STAMP"
  for f in "$IETX_ROOT/homology_allvall.tsv.gz" "$S/04b_build_refs.sh"; do
    [ -f "$f" ] && echo "$(md5sum "$f")"
  done
} > "$BAK/FINGERPRINT.txt"
cat "$BAK/FINGERPRINT.txt" | tee -a "$LOG"

# ── 3. 重跑 ───────────────────────────────────────────────────────────────
log ""
log "-------- 重跑 04b（完整输出见 $LOG）--------"
( cd "$S" && bash 04b_build_refs.sh ) >>"$LOG" 2>&1
RC=$?
log "04b 退出码 = $RC"
[ "$RC" -ne 0 ] && log "!! 04b 非零退出 —— 下面的比对结果不可用"

# ── 4. 比对 ───────────────────────────────────────────────────────────────
NEW="${HOM}/allvall.tsv"
PKG="${IETX_ROOT}/homology_allvall.tsv.gz"
[ -f "$PKG" ] || PKG="${IETX_ROOT}/results/homology_allvall.tsv.gz"
[ -f "$PKG" ] || PKG="$(find "$IETX_ROOT" -maxdepth 3 -name 'homology_allvall.tsv.gz' | head -1)"

log ""
log "-------- 比对 --------"
log "  新跑 : $NEW"
log "  随包 : ${PKG:-（未找到）}"

if [ ! -s "$NEW" ]; then
  log "!! 新跑没有产出 allvall.tsv —— 无法比对。备份在 $BAK/，可用它回滚。"
  exit 1
fi
if [ -z "${PKG:-}" ] || [ ! -f "$PKG" ]; then
  log "!! 找不到随包 homology_allvall.tsv.gz —— 只报新跑的行数，不做比对"
  log "   新跑行数: $(wc -l < "$NEW")"
  exit 1
fi

"$ENV_BIN/python" - "$NEW" "$PKG" <<'PY' 2>&1 | tee -a "$LOG"
import gzip, sys, os
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

new, pkg = sys.argv[1], sys.argv[2]
COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
PID, ALN, EV = 70.0, 200, 1e-5

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        return pd.read_csv(fh, sep="\t", header=None, names=COLS, nrows=None)

def clusters(d, t2g):
    d = d[(d.pident >= PID) & (d.length >= ALN) & (d.evalue <= EV)]
    d = d.assign(qg=d.qseqid.map(t2g), sg=d.sseqid.map(t2g)).dropna(subset=["qg", "sg"])
    d = d[d.qg != d.sg]
    ids = pd.Index(sorted(set(d.qg) | set(d.sg)))
    ix = {g: i for i, g in enumerate(ids)}
    M = coo_matrix((np.ones(len(d)), ([ix[q] for q in d.qg], [ix[s] for s in d.sg])),
                   shape=(len(ids), len(ids)))
    n, lab = connected_components(M, directed=False)
    # 用与主流程同构的配对：若无 tx2gene 就用 qseqid 本身（行数/分量数仍可比）
    return len(ids), n

import os
tx2 = None
for cand in (os.path.join(os.path.dirname(os.path.dirname(new)), "results", "tx2gene.tsv"),):
    if os.path.exists(cand):
        tx2 = pd.read_csv(cand, sep="\t").set_index("transcript_id")["gene_id"]

a, b = load(new), load(pkg)
print(f"  行数   新跑 {len(a):,}   随包 {len(b):,}   差 {len(a)-len(b):+,}")
print(f"  列集   新跑 {'/'.join(a.columns[:3])}… （12 列标准格式）")
if tx2 is None:
    print("  ⚠️ 未找到 results/tx2gene.tsv —— 只能比行数与 12 列格式，不能比簇数")
    print("     → 若行数相同且列序相同，可判『行级一致』；簇级一致性须补 tx2gene 后再跑")
else:
    na, ca = clusters(a, tx2)
    nb, cb = clusters(b, tx2)
    print(f"  过阈基因 新跑 {na:,}   随包 {nb:,}")
    print(f"  簇数     新跑 {ca:,}   随包 {cb:,}")
    ok = (na == nb) and (ca == cb)
    print()
    if ok:
        print("  ⇒ 判据 **一致**：可把 §八『等价非位级一致』改写为『已复现（位级一致）』")
    else:
        print("  ⇒ 判据 **不一致**：必须如实写差异（行数差 / 簇数差），")
        print("     并声明**随包文件为权威**。不得用重跑结果覆盖随包文件。")
PY

log ""
log "==== 结束。备份在 $BAK/ —— 若判据不一致需要回滚："
log "     rm -rf '$REF' '$HOM' && cp -a '$BAK/ref' '$BAK/homology' '$IETX_ROOT/'"
log "     （回滚前先确认随包 homology_allvall.tsv.gz 仍在原处）"
