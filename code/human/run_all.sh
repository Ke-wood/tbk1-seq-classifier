#!/usr/bin/env bash
# run_all.sh —— T-02 全流程（断网可续，跑完写 DONE）
#
# 用法（脱离终端，断线不停）:
#   setsid nohup bash scripts/run_all.sh > /dev/null 2>&1 &
# 看进度:
#   tail -f logs/run_all.log ; cat logs/STATUS
#
# 每一步都幂等：已有产物就跳过。网络断了会自己等，不会退出。
set -uo pipefail
R=${IETX_HUMAN:?set IETX_HUMAN}
E=${IETX_ENV:-env/bin}
cd "$R"
mkdir -p logs
LOG=$R/logs/run_all.log
STATUS=$R/logs/STATUS

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
mark() { say "$*"; echo "$*" > "$STATUS"; }
trap 'say "!! 脚本异常退出（行 $LINENO）"; echo "FAILED 行 $LINENO $(date)" > /dev/null' ERR

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
THREADS=${THREADS:-20}

# ── 0 基因组：下到完整为止（断网就等）──────────────────────────
mark "0/6 下载基因组"
n=0
# ⚠️ 已经解压并建好索引就别再跑 gzip -t —— 882 MB 在网络盘上、机器负载 180 时
#    要跑 40 分钟以上，纯属白等。校验只对「还没解压过」的首次下载有意义。
if [ -s ref/genome.fa ] && [ -s ref/genome.fa.fai ]; then
  say "基因组已解压并索引，跳过下载与校验"
  n=-1                       # -1 表示已就绪
fi
until [ "$n" -eq -1 ] || gzip -t ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz 2>/dev/null; do
  n=$((n+1))
  say "基因组未完整（第 $n 轮），尝试续传…"
  bash scripts/get_genome.sh >> "$LOG" 2>&1
  if ! gzip -t ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz 2>/dev/null; then
    # 字节数对上了但 gzip 校验不过 = 有脏段，光续传补不回来，必须整段重来
    if [ -s ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz ] && [ $((n % 3)) -eq 0 ]; then
      say "第 $n 轮：整包校验失败，清空分片重下"
      rm -rf ref/chunks ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    fi
    say "下载未成功，等 120 s 再来"
    sleep 120
  fi
done
say "基因组校验通过"

# ── 1 参考 + 注释 + 启动子 + MFE + DIAMOND ──────────────────
mark "1/6 建参考（含 RNAfold / DIAMOND，最慢）"
bash scripts/55_build_ref.sh >> "$LOG" 2>&1
[ -s ref/mfe.tsv ] && [ -s homology/allvall_filt.tsv ] \
  || { say "!! 55_build_ref 未完成，见 $LOG"; exit 1; }

# ── 2 9 维特征 ─────────────────────────────────────────────
mark "2/6 特征"
$E/python scripts/57_features.py >> "$LOG" 2>&1 || { say "!! 57 失败"; exit 1; }

# ── 3 差异表达（已跑过会跳过）──────────────────────────────
mark "3/6 DESeq2"
for G in GSE351189 GSE309167; do
  $E/python scripts/56_deseq.py "$G" >> "$LOG" 2>&1 || { say "!! 56 $G 失败"; exit 1; }
done

# ── 4 标签 ─────────────────────────────────────────────────
mark "4/6 标签"
for G in GSE351189 GSE309167; do
  for C in pooled ifna_6h ifng_6h ifna_72h ifng_72h ifna2 ifnb ifnw; do
    [ -s "人类侧_$G/02_deg/$C.tsv" ] || continue
    $E/python scripts/58_labels.py "$G" "$C" >> "$LOG" 2>&1 \
      || { say "!! 58 $G $C 失败"; exit 1; }
  done
done

# ── 5 模型（pooled 全口径 + 各扰动 both 稳健性）──────────────
mark "5/6 模型"
for G in GSE351189 GSE309167; do
  $E/python scripts/59_model.py "$G" pooled >> "$LOG" 2>&1 \
    || { say "!! 59 $G pooled 失败"; exit 1; }
  for C in ifna_6h ifng_6h ifna_72h ifng_72h ifna2 ifnb ifnw; do
    [ -s "人类侧_$G/05_label/label_both_$C.tsv" ] || continue
    $E/python scripts/59_model.py "$G" "$C" >> "$LOG" 2>&1 || say "⚠️ 59 $G $C 失败（不影响主口径）"
  done
done

# ── 6 汇总 ─────────────────────────────────────────────────
mark "6/6 汇总"
$E/python scripts/60_compare.py >> "$LOG" 2>&1 || say "⚠️ 60 失败"
$E/python scripts/61_qc.py >> "$LOG" 2>&1 || say "⚠️ 61 失败"

say "全部完成"
echo "DONE $(date '+%F %T')" > "$STATUS"
