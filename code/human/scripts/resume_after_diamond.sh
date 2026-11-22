#!/usr/bin/env bash
# resume_after_diamond.sh —— 等 DIAMOND 跑完再续跑 run_all.sh
#
# 55_build_ref.sh 在 DIAMOND 那一步失败后整个 run_all 就退出了；DIAMOND 是单独
# 手工拉起来的。这个脚本守着它，一结束就自动把 run_all 接上（run_all 自己会跳过
# 已经完成的步骤），不用人盯。
set -uo pipefail
R=${IETX_HUMAN:?set IETX_HUMAN}
cd "$R"
echo "[$(date '+%F %T')] 等待 DIAMOND…"
while pgrep -f "diamond blastp" > /dev/null; do sleep 120; done
if [ ! -s homology/allvall.tsv ]; then
  echo "[$(date '+%F %T')] !! DIAMOND 结束但 allvall.tsv 为空，不续跑"; exit 1
fi
echo "[$(date '+%F %T')] DIAMOND 完成（$(wc -l < homology/allvall.tsv) 行），续跑 run_all"
setsid nohup bash scripts/run_all.sh > /dev/null 2>&1 < /dev/null &
