#!/usr/bin/env bash
# =============================================================================
# 01_setup_env.sh —— IE-Tx 生信环境【完全隔离版】
#
# 隔离原则（在学校共享服务器上很重要）:
#   1. 不需要 root / sudo —— 全程用你自己的权限
#   2. 不碰系统 Python —— 自装 micromamba，自带 Python
#   3. 不写公共缓存 —— 包缓存全部限制在 IETX_ROOT 内（不写 ~/.conda ~/.cache）
#   4. 不改 .bashrc —— 用 activate_ietx.sh 临时激活，退出即干净
#   5. 所有文件都在一个目录里 —— 不想要了直接 rm -rf 这一个目录即可
#
# 用法:  bash scripts/01_setup_env.sh
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# 0. 决定 IETX_ROOT（优先放大容量的 work / scratch 盘）
#    可用环境变量覆盖:  export IETX_ROOT=/your/path
# ---------------------------------------------------------------------------
if [ -z "${IETX_ROOT:-}" ]; then
  for cand in "${WORK:-}/ietx" "${SCRATCH:-}/ietx" "/scratch/$(whoami)/ietx" \
              "/work/$(whoami)/ietx" "$HOME/ietx"; do
    case "$cand" in /ietx|"") continue ;; esac
    parent="$(dirname "$cand")"
    if [ -d "$parent" ] && [ -w "$parent" ]; then IETX_ROOT="$cand"; break; fi
  done
fi
IETX_ROOT="${IETX_ROOT:-$HOME/ietx}"

ENV_PREFIX="$IETX_ROOT/env"                 # 环境本体（不是命名环境，是固定路径）
MAMBA_BIN="$IETX_ROOT/bin/micromamba"
MAMBA_ROOT="$IETX_ROOT/mamba_root"          # micromamba 的根
PKGS_DIR="$IETX_ROOT/mamba_root/pkgs"       # 包缓存（隔离关键）
PIP_CACHE="$IETX_ROOT/.pip-cache"
LOGS="$IETX_ROOT/logs"

mkdir -p "$IETX_ROOT/bin" "$MAMBA_ROOT" "$PKGS_DIR" "$PIP_CACHE" "$LOGS"
LOG="$LOGS/01_setup_env.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# 隔离环境变量：所有缓存/临时文件都锁在 IETX_ROOT 内
# 这样绝不会写到 ~/.conda、~/.cache、/tmp 等公共位置
# ---------------------------------------------------------------------------
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export CONDA_PKGS_DIRS="$PKGS_DIR"
export MAMBA_PKGS_DIRS="$PKGS_DIR"
export PIP_CACHE_DIR="$PIP_CACHE"
export XDG_CACHE_HOME="$IETX_ROOT/.cache"
mkdir -p "$XDG_CACHE_HOME"

log "============================================================"
log "  IE-Tx 环境搭建（完全隔离版）"
log "============================================================"
log "IETX_ROOT   = $IETX_ROOT"
log "ENV_PREFIX  = $ENV_PREFIX"
log "包缓存      = $PKGS_DIR   (不会写 ~/.conda)"
log "用户        = $(whoami)  (不会用 sudo)"
log ""

# ---------------------------------------------------------------------------
# 0b. 软件源选择（国内服务器强烈建议走镜像，否则极慢）
#     启用镜像:  export IETX_CN_MIRROR=1   然后重跑本脚本
# ---------------------------------------------------------------------------
CHANNELS=(-c conda-forge -c bioconda)
PIP_INDEX=""
if [ "${IETX_CN_MIRROR:-0}" = "1" ]; then
  log "★ 启用中国大陆镜像源（清华 TUNA）—— 速度可提升 10 倍以上"
  CHANNELS=(
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  )
  PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
else
  log "使用官方源（若很慢，请用 export IETX_CN_MIRROR=1 重跑）"
fi
log ""

# ---------------------------------------------------------------------------
# 1. 安全检查：确认不会干扰别人
# ---------------------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
  log "!! 警告: 你当前是 root。本方案设计为普通用户使用，"
  log "   以 root 运行会导致文件属主错误。建议退出 root 再跑。"
  read -r -p "   仍要继续? (yes/no) " a; [ "$a" = "yes" ] || exit 1
fi

# 若检测到系统已激活 conda 环境，提示（避免嵌套冲突）
if [ -n "${CONDA_PREFIX:-}" ]; then
  log "提示: 检测到已激活的 conda 环境: $CONDA_PREFIX"
  log "      本脚本会使用自己的隔离环境，不受影响。"
fi
log ""

# ---------------------------------------------------------------------------
# 2. 安装 micromamba（自包含，不需要 root）
# ---------------------------------------------------------------------------
if [ -x "$MAMBA_BIN" ]; then
  log "[跳过] micromamba 已存在: $MAMBA_BIN"
else
  log "---- 下载 micromamba（自包含静态包）----"
  TMPD="$IETX_ROOT/.tmp"; mkdir -p "$TMPD"
  if curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "$TMPD/mm.tar.bz2"; then
    if tar -xjf "$TMPD/mm.tar.bz2" -C "$TMPD" bin/micromamba 2>>"$LOG"; then
      mv "$TMPD/bin/micromamba" "$MAMBA_BIN"
      chmod +x "$MAMBA_BIN"
      log "  [OK] $MAMBA_BIN"
    else
      log "  !! 解压失败"
    fi
    rm -rf "$TMPD"
  else
    log "  !! 下载失败（可能需要代理，见 00_check_server.sh 第 5 节）"
    log "     若有代理: export https_proxy=http://代理:端口  后重跑"
  fi
fi

if [ ! -x "$MAMBA_BIN" ]; then
  log ""
  log "==== 无法安装 micromamba，改用服务器自带 conda（若存在）===="
  if command -v conda >/dev/null 2>&1; then
    log "找到系统 conda: $(command -v conda)"
    log "为避免污染，请手动执行下面命令（用 -p 指定路径，仍保持隔离）:"
    log "  conda create -y -p $ENV_PREFIX -c conda-forge -c bioconda python=3.11 \\"
    log "    numpy pandas scipy scikit-learn matplotlib biopython pyfaidx \\"
    log "    pybedtools requests tqdm openpyxl statsmodels \\"
    log "    samtools bedtools seqkit entrez-direct viennarna"
    log ""
    log "  ⚠️ 注意: 若系统 conda 默认往 ~/.conda/pkgs 写缓存，可先设置:"
    log "     export CONDA_PKGS_DIRS=$PKGS_DIR"
  else
    log "也没有系统 conda。请联系管理员，或配置代理后重跑本脚本。"
  fi
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. 创建隔离环境
#    用 -p（固定路径）而非 -n（命名环境），环境就落在 IETX_ROOT 里，一目了然
# ---------------------------------------------------------------------------
if [ -x "$ENV_PREFIX/bin/python" ]; then
  log "[跳过] 环境已存在: $ENV_PREFIX"
else
  log "---- 创建隔离环境（Python 3.11）----"
  log "     （首次约 3–10 分钟，取决于网速）"
  "$MAMBA_BIN" create -y -p "$ENV_PREFIX" \
      "${CHANNELS[@]}" \
      python=3.11 \
      numpy pandas scipy scikit-learn matplotlib \
      biopython pyfaidx pybedtools requests tqdm openpyxl statsmodels \
      samtools bedtools seqkit entrez-direct viennarna \
      >>"$LOG" 2>&1
  if [ $? -eq 0 ] && [ -x "$ENV_PREFIX/bin/python" ]; then
    log "  [OK] 环境创建成功"
  else
    log "  !! 创建失败。日志末尾 30 行:"
    tail -30 "$LOG" | sed 's/^/     /'
    exit 1
  fi
fi

# pip 也指向国内镜像（若启用）
if [ -n "$PIP_INDEX" ] && [ -x "$ENV_PREFIX/bin/pip" ]; then
  "$MAMBA_BIN" run -p "$ENV_PREFIX" pip config set global.index-url "$PIP_INDEX" >>"$LOG" 2>&1 \
    && log "  [OK] pip 已指向 $PIP_INDEX"
fi

# ---------------------------------------------------------------------------
# 4. 自检
# ---------------------------------------------------------------------------
log ""
log "---- 环境自检 ----"
"$MAMBA_BIN" run -p "$ENV_PREFIX" python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib, sys
print(f"python {sys.version.split()[0]}")
ok = True
for m in ["numpy","pandas","scipy","sklearn","matplotlib","Bio","pyfaidx","requests","tqdm"]:
    try:
        mod = importlib.import_module(m)
        print(f"[OK]   {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False
        print(f"[MISS] {m:12s} {type(e).__name__}")
print("env OK" if ok else "env INCOMPLETE")
PY

for t in samtools bedtools seqkit RNAfold; do
  if "$MAMBA_BIN" run -p "$ENV_PREFIX" bash -c "command -v $t" >/dev/null 2>&1; then
    log "  [OK] $t"
  else
    log "  [MISS] $t"
  fi
done

# ---------------------------------------------------------------------------
# 5. 生成激活脚本（不污染 .bashrc）
# ---------------------------------------------------------------------------
cat > "$IETX_ROOT/activate_ietx.sh" <<EOF
# 激活 IE-Tx 隔离环境。用法:  source $IETX_ROOT/activate_ietx.sh
# 退出环境:  micromamba deactivate    或直接关掉终端
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export CONDA_PKGS_DIRS="$PKGS_DIR"
export MAMBA_PKGS_DIRS="$PKGS_DIR"
export PIP_CACHE_DIR="$PIP_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export IETX_ROOT="$IETX_ROOT"
eval "\$("$MAMBA_BIN" shell hook -s bash)"
micromamba activate "$ENV_PREFIX"
echo "[IE-Tx] 隔离环境已激活: $ENV_PREFIX"
echo "[IE-Tx] IETX_ROOT=$IETX_ROOT"
EOF
chmod +x "$IETX_ROOT/activate_ietx.sh"
log ""
log "  [OK] 激活脚本已生成: $IETX_ROOT/activate_ietx.sh"

# 记录安装清单，便于日后卸载
cat > "$IETX_ROOT/UNINSTALL.txt" <<EOF
本目录 ($IETX_ROOT) 是 IE-Tx 项目的全部隔离环境。
卸载方法（删除本目录即可，不影响系统和其他人）:
    rm -rf $IETX_ROOT
EOF

# ---------------------------------------------------------------------------
log ""
log "============================================================"
log "  完成"
log "============================================================"
log ""
log "以后每次使用前，先激活环境："
log "    source $IETX_ROOT/activate_ietx.sh"
log ""
log "不想激活也可以，用 run 方式执行单条命令："
log "    $MAMBA_BIN run -p $ENV_PREFIX python3 scripts/03_verify_data.py"
log ""
log "停用环境："
log "    micromamba deactivate"
log ""
log "彻底卸载（慎用）："
log "    rm -rf $IETX_ROOT"
log ""
