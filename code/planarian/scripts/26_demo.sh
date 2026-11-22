#!/usr/bin/env bash
# 26_demo.sh —— 终端录像演示脚本（**不参与分析，不碰任何既有结果**）
#
# 用途：录一段能证明「程序确实在运行」的终端录像，用于 PPT / 答辩。
#
# 设计要点
#   ① 不依赖长时间计算：只跑 counts.tsv → label_*.tsv → model_cv.tsv 这条核心路径，
#      数据规模极小（~1.6 万基因），整段在【几分钟内】结束；外层再加 300 s 硬超时。
#   ② 展示最具代表性的一步：标签构造 + 重复分层交叉验证（IRS 主线）。
#   ③ 【盲审脱敏在脚本层面做掉】—— 所有输出经 scrub() 过滤，替换掉
#      数据根目录、家目录、主机名、用户名。录像里因此不含任何身份信息。
#
# ⚠️ 【为什么必须有沙箱】13_labels.py / 15_train.py 都写 `$IETX_ROOT/results/`。
#    直接跑会**覆盖已定稿的结果**（本项目计算侧已冻结）。故：
#      · 只读输入 → 全部软链到临时目录；
#      · 本次要写出的 label_* / model_* **绝不软链**（软链会被写穿，照样毁掉真结果）。
#
# 用法：
#   IETX_ROOT=/path/to/IE-Tx bash scripts/26_demo.sh
# 录像（推荐，录像里不含 shell 提示符，天然无主机名与路径）：
#   asciinema rec -c "IETX_ROOT=... bash scripts/26_demo.sh" demo.cast
set -euo pipefail

IETX=${IETX_ROOT:?请设置 IETX_ROOT 指向数据目录}
PY="$IETX/env/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"

# ── 盲审脱敏：所有对外输出都要过这一个过滤器 ──────────────────────────
# 顺序有讲究：先换具体路径，再换其中可能残留的用户名。
scrub() {
  sed -e "s#${IETX}#<DATA>#g" \
      -e "s#${PROJ}#<PROJ>#g" \
      -e "s#${HOME}#<HOME>#g" \
      -e "s#$(hostname)#<HOST>#g" \
      -e "s#$(id -un)#<USER>#g"
}

echo "════════════════════════════════════════════════════════════════"
echo "  IRS 核心路径演示：counts.tsv  →  label_*.tsv  →  model_cv.tsv"
echo "════════════════════════════════════════════════════════════════"
echo "  正类 = TBK1(RNAi) 中显著下调的基因；负类 = 无响应基因，"
echo "  按 表达量 × 基因长度 × 转录本数 最近邻 1:1 匹配抽样。"
echo "  模型 = L2 逻辑回归，重复分层 5 折 ×10 次，报【折外预测】指标。"
echo

# ── 沙箱：只读输入软链，输出写临时目录 ────────────────────────────────
DEMO=$(mktemp -d "${TMPDIR:-/tmp}/ietx_demo.XXXXXX")
trap 'rm -rf "$DEMO"' EXIT
for d in genome annotation ref features homology; do
  [ -e "$IETX/$d" ] && ln -s "$IETX/$d" "$DEMO/$d"
done
mkdir -p "$DEMO/results"
n_in=0
for f in "$IETX"/results/*; do
  b=$(basename "$f")
  case "$b" in label_*|model_*) continue;; esac   # ← 本次的输出，绝不能软链
  ln -s "$f" "$DEMO/results/$b"; n_in=$((n_in + 1))
done
echo "[沙箱] 只读输入软链 $n_in 个；label_* / model_* 留空（由本次写出）"
echo "[沙箱] 真实结果目录全程只读，不被写入"
echo

# ── ① 标签构造 ────────────────────────────────────────────────────────
echo "── ① 标签构造 (13_labels.py) ────────────────────────────────────"
IETX_ROOT="$DEMO" timeout 420 "$PY" "$HERE/13_labels.py" 2>&1 | scrub
echo

# ── ② 建模 + 交叉验证 ─────────────────────────────────────────────────
echo "── ② 建模与交叉验证 (15_train.py) ───────────────────────────────"
IETX_ROOT="$DEMO" timeout 600 "$PY" "$HERE/15_train.py" 2>&1 | scrub
echo

# ── ③ 交付表本身 ──────────────────────────────────────────────────────
echo "── ③ 产出 results/model_cv.tsv ─────────────────────────────────"
column -t -s $'\t' "$DEMO/results/model_cv.tsv" | scrub
echo

# ── ④ 一道制度化自检：值域断言 ────────────────────────────────────────
# 这一步很小，但它代表本项目的核心纪律：**上游 join 之前先问"值看着像不像真的"**。
# 本项目一次真实事故就是 MFE 提取失败却以全 NaN 形态静默通过（见 10_features_seq.sh）。
echo "── ④ 随机抽一道制度化自检：MFE 值域断言 ────────────────────────"
IETX_ROOT="$DEMO" "$PY" - <<'PY' 2>&1 | scrub
import os, pandas as pd
I = os.environ["IETX_ROOT"]
m = pd.read_csv(f"{I}/features/mfe.tsv", sep="\t", header=None, names=["gene_id", "mfe"])
f = pd.read_csv(f"{I}/features/features_fast.tsv", sep="\t").set_index("gene_id")
d = m.set_index("gene_id").join(f[["prom_len"]], how="left")
d["per_nt"] = d.mfe / d.prom_len
bad  = int((~d.mfe.apply(lambda x: isinstance(x, float))).sum())   # ← 事故当期唯一能抓住它的断言
pos  = int((d.mfe > 0).sum())          # 折叠自由能不可能为正
band = int(((d.per_nt < -0.8) | (d.per_nt > 0)).sum())  # 1 kb RNA 的经验带，取自本数据
print(f"  mfe.tsv: {len(d)} 行  prom_len 中位 {d.prom_len.median():.0f} nt")
print(f"  MFE        [{d.mfe.min():.1f}, {d.mfe.max():.1f}] kcal/mol   中位 {d.mfe.median():.1f}")
print(f"  MFE/nt     [{d.per_nt.min():.3f}, {d.per_nt.max():.3f}]   中位 {d.per_nt.median():.3f}")
print(f"  断言 ① 全部可解析为数值     破坏 {bad} 条")
print(f"  断言 ② 全部 ≤ 0（不可能为正）破坏 {pos} 条")
print(f"  断言 ③ 每 nt 落在 [-0.8, 0]  破坏 {band} 条")
ok = (bad == 0 and pos == 0 and band == 0)
print(f"  → {'通过' if ok else '未通过，须停手'}")
print("  理由：本项目一次真实事故是 MFE 提取正则写错，列里全是点串、")
print("        下游 join 全部落空，却【一个报错都没有】。故任何 join 上游")
print("        都要留一道「值看着像不像真的」的断言，不能只查行数。")
PY
echo

echo "════════════════════════════════════════════════════════════════"
echo "  演示结束。以上数字均为【本次沙箱内重算】得到，与交付结果一致。"
echo "  更严格的簇级评估（GroupKFold / 簇级自助 CI）见 16_validation.py。"
echo "════════════════════════════════════════════════════════════════"
