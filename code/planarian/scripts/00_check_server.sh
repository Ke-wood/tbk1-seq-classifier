#!/usr/bin/env bash
# =============================================================================
# 00_check_server.sh —— 服务器体检（不改动任何东西，只读）
# 目的: 摸清服务器现状 → 决定"环境隔离"方案怎么定
# 用法:  bash scripts/00_check_server.sh
# =============================================================================
set -uo pipefail

line(){ printf '%s\n' "------------------------------------------------------------------------"; }
sec(){ echo ""; line; echo "  $*"; line; }

echo "============================================================"
echo "  服务器体检报告   $(date '+%F %T')"
echo "============================================================"

# ---------------------------------------------------------------------------
sec "1. 我是谁 / 在哪台机器"
echo "  用户名   : $(whoami)"
echo "  主机名   : $(hostname)"
echo "  家目录   : $HOME"
echo "  操作系统 : $(uname -srm 2>/dev/null)"
[ -f /etc/os-release ] && echo "  发行版   : $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"

# ---------------------------------------------------------------------------
sec "2. 磁盘：家目录 / work / scratch 在哪、还剩多少"
echo "  ---- 总体磁盘使用 ----"
df -h 2>/dev/null | grep -vE '^(tmpfs|devtmpfs|overlay|shm)' | head -15 | sed 's/^/  /'

echo ""
echo "  ---- 常见临时盘（大数据要放这里，别放家目录） ----"
for d in "$WORK" "$SCRATCH" "$TMPDIR" "$HOME/work" "$HOME/scratch" \
         /work /scratch /scratch/"$(whoami)" /work/"$(whoami)" \
         /data /lustre /gpfs /dfs /bigdata /share; do
  [ -z "${d:-}" ] && continue
  if [ -d "$d" ]; then
    if [ -w "$d" ]; then w="[可写✓]"; else w="[只读✗]"; fi
    avail=$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')
    echo "  $w  $d   (可用 $avail)"
  fi
done
echo "  (环境变量 WORK=${WORK:-未设置}  SCRATCH=${SCRATCH:-未设置}  TMPDIR=${TMPDIR:-未设置})"

# ---------------------------------------------------------------------------
sec "3. 家目录配额（HPC 家目录通常很小，装不下基因组数据）"
if command -v quota >/dev/null 2>&1; then
  quota -s 2>/dev/null | sed 's/^/  /' || echo "  (quota 命令存在但无输出)"
else
  echo "  (无 quota 命令)"
fi
echo "  家目录当前占用: $(du -sh "$HOME" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------------------
sec "4. 已有的 Python / 包管理器（要避免冲突）"
for c in python3 python conda mamba micromamba uv module; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "  [有] $c -> $(command -v "$c")"
  else
    echo "  [无] $c"
  fi
done
echo ""
echo "  python3 版本: $(python3 --version 2>&1)"
echo "  python3 位置: $(command -v python3 2>/dev/null || echo '无')"

# 是否 HPC module 系统
if command -v module >/dev/null 2>&1; then
  echo ""
  echo "  ---- 检测到 module 系统（HPC 常见）。已加载的模块: ----"
  module list 2>&1 | head -20 | sed 's/^/  /'
  echo "  ---- 可用 conda/python 相关模块（前 20 个）----"
  module avail 2>&1 | tr ' ' '\n' | grep -iE 'conda|python|anaconda|miniconda' | head -20 | sed 's/^/  /' || echo "  (无)"
fi

# 重要：是否会污染系统
echo ""
echo "  ⚠️ 是否能用 sudo: $(command -v sudo >/dev/null 2>&1 && echo '有 sudo（但本方案不需要它，也不会用）' || echo '无 sudo（正常，本方案不需要 root）')"

# ---------------------------------------------------------------------------
sec "5. 网络连通性（决定能不能自动下载）"
test_url(){
  local name="$1" url="$2"
  if curl -fsS -o /dev/null --max-time 12 "$url" 2>/dev/null; then
    echo "  [通] $name"
  else
    echo "  [不通] $name  ($url)"
  fi
}
test_url "conda-forge 源"   "https://conda.anaconda.org/conda-forge/noarch/repodata.json"
test_url "bioconda 源"      "https://conda.anaconda.org/bioconda/noarch/repodata.json"
test_url "micromamba 下载"  "https://micro.mamba.pm/api/micromamba/linux-64/latest"
test_url "Zenodo"           "https://zenodo.org/api/records/17072252"
test_url "NCBI FTP"         "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/"
test_url "PlanMine"         "https://planmine.mpinat.mpg.de/planmine/begin.do"
echo ""
echo "  ℹ️ 若有 [不通]，说明需走学校代理。代理地址问管理员，然后："
echo "     export http_proxy=http://代理地址:端口"
echo "     export https_proxy=http://代理地址:端口"

# ---------------------------------------------------------------------------
sec "6. 机器资源"
echo "  CPU 核数 : $(nproc 2>/dev/null || echo '?')"
echo "  内存     : $(free -h 2>/dev/null | awk 'NR==2{print $2" (可用 "$7")"}' || echo '?')"
echo "  单文件句柄上限: $(ulimit -n 2>/dev/null || echo '?')"
echo "  shell    : $SHELL"

# ---------------------------------------------------------------------------
sec "7. 环境变量里有没有别人设的坑"
echo "  PATH 里与 python/conda 相关的条目:"
echo "$PATH" | tr ':' '\n' | grep -iE 'python|conda|mamba|module' | sed 's/^/    /' || echo "    (无)"
echo ""
echo "  相关变量:"
for v in PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV MAMBA_ROOT_PREFIX LD_LIBRARY_PATH; do
  eval "val=\${$v:-}"; [ -n "$val" ] && echo "    $v=$val"
done
echo "  (以上若为空，是好事)"

# ---------------------------------------------------------------------------
sec "8. 结论与建议"
echo "  ▸ 把 IETX_ROOT 设在:"
if [ -n "${WORK:-}" ] && [ -w "${WORK:-}" ]; then
  echo "      export IETX_ROOT=$WORK/ietx        ← 推荐（WORK 盘，空间大）"
elif [ -n "${SCRATCH:-}" ] && [ -w "${SCRATCH:-}" ]; then
  echo "      export IETX_ROOT=$SCRATCH/ietx     ← 推荐（SCRATCH 盘，空间大）"
elif [ -d /scratch ] && [ -w /scratch ]; then
  echo "      export IETX_ROOT=/scratch/$(whoami)/ietx"
elif [ -d /work ] && [ -w /work ]; then
  echo "      export IETX_ROOT=/work/$(whoami)/ietx"
else
  echo "      export IETX_ROOT=$HOME/ietx        ← 没找到 work 盘。"
  echo "         ⚠️ 注意家目录配额（上面第 3 节），可能装不下基因组数据。"
  echo "         若配额小，务必问管理员要一个 work/scratch 目录。"
fi
echo ""
echo "  ▸ 环境隔离方案: micromamba 私装（不需要 root、不碰系统）"
echo "      - 全部文件都在 IETX_ROOT 内"
echo "      - 包缓存也在 IETX_ROOT 内（不写 ~/.conda，不写 ~/.cache）"
echo "      - 不修改 .bashrc（用 activate_ietx.sh 临时激活）"
echo ""
echo "  下一步:  bash scripts/01_setup_env.sh"
echo ""
