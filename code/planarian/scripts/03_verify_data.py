#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_verify_data.py —— Phase 0 数据体检

目的: 在写任何分析脚本之前，先搞清楚数据到底长什么样。
     本脚本只读不写（除了一份报告），安全可反复运行。

用法:
    python3 scripts/03_verify_data.py
    # 报告同时打印到屏幕并写入 $IETX_ROOT/logs/03_verify_report.txt
"""
from __future__ import annotations
import os, sys, glob, gzip, json, textwrap
from pathlib import Path

IETX_ROOT = Path(os.environ.get("IETX_ROOT", Path.home() / "ietx"))
GENOME, ANN, DREES, LOGS = (IETX_ROOT / d for d in ("genome", "annotation", "drees", "logs"))
LOGS.mkdir(parents=True, exist_ok=True)
REPORT = LOGS / "03_verify_report.txt"

_lines: list[str] = []
def out(s: str = "") -> None:
    print(s)
    _lines.append(s)

def hr(title: str) -> None:
    out("")
    out("=" * 78)
    out(title)
    out("=" * 78)

def open_maybe_gz(p: Path):
    """按扩展名决定是否 gzip 解包，返回文本句柄。"""
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.suffix == ".gz" \
           else open(p, "r", encoding="utf-8", errors="replace")

def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

# =============================================================================
# 0. 目录概览
# =============================================================================
hr("0. 目录概览")
for d in (GENOME, ANN, DREES):
    out(f"\n[{d}]")
    if not d.exists():
        out("  (目录不存在)"); continue
    files = [p for p in d.rglob("*") if p.is_file()]
    if not files:
        out("  (空)"); continue
    for p in sorted(files):
        out(f"  {human(p.stat().st_size):>9}  {p.relative_to(IETX_ROOT)}")
    out(f"  小计: {len(files)} 个文件")

# =============================================================================
# 1. 基因组 FASTA 体检
# =============================================================================
hr("1. 基因组 FASTA")
fa_files = []
for pat in ("*.fna", "*.fa", "*.fasta", "*.fna.gz", "*.fa.gz"):
    fa_files += list(GENOME.rglob(pat))
fa_files = sorted(set(fa_files))

if not fa_files:
    out("  未找到 FASTA。请先跑 02_download_data.sh。")
else:
    for fa in fa_files:
        out(f"\n--- {fa.name} ({human(fa.stat().st_size)}) ---")
        try:
            lengths, name, cur = [], None, 0
            with open_maybe_gz(fa) as fh:
                for line in fh:
                    if line.startswith(">"):
                        if name is not None: lengths.append(cur)
                        name = line[1:].strip().split()[0]; cur = 0
                    else:
                        cur += len(line.strip())
            if name is not None: lengths.append(cur)
            lengths.sort(reverse=True)
            total = sum(lengths)
            # N50
            acc = 0; n50 = 0
            for L in lengths:
                acc += L
                if acc >= total / 2: n50 = L; break
            out(f"  序列条数 : {len(lengths)}")
            out(f"  总长     : {total:,} bp ({total/1e6:.1f} Mb)")
            out(f"  最长     : {lengths[0]:,} bp")
            out(f"  N50      : {n50:,} bp")
            if n50 > 1_000_000:
                out("  -> 判定: 很可能是【染色体级】装配 ✓")
            elif n50 > 50_000:
                out("  -> 判定: 中等连续性（scaffold 级）")
            else:
                out("  -> 判定: 碎片化（contig 级）；若要染色体级请用 Ivanković 2024 那份")
            # 打印头几条序列名
            with open_maybe_gz(fa) as fh:
                names = []
                for line in fh:
                    if line.startswith(">"):
                        names.append(line[1:].strip().split()[0])
                        if len(names) >= 5: break
            out(f"  示例序列名: {names}")
        except Exception as e:
            out(f"  !! 解析失败: {type(e).__name__}: {e}")

# =============================================================================
# 2. 注释文件体检 (GTF / GFF3)
# =============================================================================
hr("2. 注释文件 (GTF/GFF)")
ann_files = []
for pat in ("*.gtf", "*.gff", "*.gff3", "*.gtf.gz", "*.gff3.gz"):
    ann_files += list(ANN.rglob(pat))
ann_files = sorted(set(ann_files))

if not ann_files:
    out("  未找到 GTF/GFF。PlanMine 的 SMESG 基因模型可能是 GFF3，也可能需手工下载。")
else:
    for g in ann_files:
        out(f"\n--- {g.name} ({human(g.stat().st_size)}) ---")
        try:
            feats, chroms, gene_ids, n = {}, set(), [], 0
            with open_maybe_gz(g) as fh:
                for line in fh:
                    if line.startswith("#") or not line.strip(): continue
                    n += 1
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 9: continue
                    chrom, _, feat, _, _, _, _, _, attrs = parts[:9]
                    chroms.add(chrom)
                    if feat in ("gene", "mRNA", "transcript", "CDS", "exon", "UTR",
                                "five_prime_UTR", "three_prime_UTR"):
                        feats[feat] = feats.get(feat, 0) + 1
                    if feat == "gene" and len(gene_ids) < 5:
                        gene_ids.append(attrs[:120])
                    if n >= 400_000: break   # 大文件只读前 40 万行做抽样
            out(f"  扫描行数 : {n:,}{'（已截断抽样）' if n >= 400_000 else ''}")
            out(f"  染色体数 : {len(chroms)}")
            out(f"  特征统计 : {feats}")
            if gene_ids:
                out("  示例 gene 行:")
                for a in gene_ids: out(f"    {a}")
                # 推断基因 ID 体系
                joined = " ".join(gene_ids)
                if "SMESG" in joined:
                    out("  -> ID 体系: 【SMESG】 (PlanMine 3.0 基因组锚定预测)")
                elif "dd_Smed" in joined or "ddSmed" in joined:
                    out("  -> ID 体系: 【dd_Smed_v6】 (Grohme 2018 转录本簇)")
                else:
                    out("  -> ID 体系: 未识别，请人工查看上面的示例行")
            else:
                out("  !! 未找到 feat=='gene' 行 —— 可能是 GFF3 且基因层级写法不同，请人工抽查前 20 行非注释行。")
        except Exception as e:
            out(f"  !! 解析失败: {type(e).__name__}: {e}")

# =============================================================================
# 3. Drees 2026 数据体检
# =============================================================================
hr("3. Drees 2026 (Zenodo) 数据表")
tables = [p for p in DREES.rglob("*") if p.is_file() and p.suffix.lower() in
          (".csv", ".tsv", ".txt", ".xlsx", ".xls") and not p.name.startswith("_")]
if not tables:
    out("  未找到表格文件。Zenodo 记录可能: (a) 尚未下载; (b) 只含 PDF/图片; (c) 在子目录里。")
    out(f"  请查看 {DREES/'zenodo_record.json'} 的 files[] 字段确认内容。")
    zj = DREES / "zenodo_record.json"
    if zj.exists():
        try:
            d = json.load(open(zj, encoding="utf-8"))
            out(f"  record title: {d.get('metadata',{}).get('title','?')}")
            out("  files:")
            for f in d.get("files", []):
                out(f"    - {f.get('key') or f.get('filename')}  ({f.get('size','?')} bytes)")
        except Exception as e:
            out(f"  !! 解析 zenodo_record.json 失败: {e}")
else:
    import csv
    for t in tables:
        out(f"\n--- {t.name} ({human(t.stat().st_size)}) ---")
        try:
            if t.suffix.lower() in (".xlsx", ".xls"):
                try:
                    import pandas as pd
                    xl = pd.ExcelFile(t)
                    out(f"  sheet 页: {xl.sheet_names}")
                    for sn in xl.sheet_names[:3]:
                        df = xl.parse(sn, nrows=5)
                        out(f"  [{sn}] shape≈{df.shape}, 列: {list(df.columns)}")
                except ImportError:
                    out("  !! 需要 pandas + openpyxl 才能读 Excel")
            else:
                # 探测分隔符
                with open(t, "r", encoding="utf-8", errors="replace") as fh:
                    head = [next(fh) for _ in range(5)]
                sep = "\t" if head[0].count("\t") > head[0].count(",") else ","
                out(f"  推断分隔符: {'TAB' if sep==chr(9) else 'COMMA'}")
                out(f"  前 3 行预览:")
                for h in head[:3]:
                    out("    " + h.rstrip()[:200])
                with open(t, "r", encoding="utf-8", errors="replace") as fh:
                    rd = csv.reader(fh, delimiter=sep)
                    try:
                        cols = next(rd)
                        out(f"  列名 ({len(cols)}): {cols[:30]}")
                        nrow = sum(1 for _ in rd)
                        out(f"  数据行数: {nrow:,}")
                    except StopIteration:
                        out("  (空文件)")
        except Exception as e:
            out(f"  !! 读取失败: {type(e).__name__}: {e}")

# =============================================================================
# 4. 结论与下一步
# =============================================================================
hr("4. 结论")
def has(d: Path, pats) -> bool:
    return any(list(d.rglob(p)) for p in pats)

checks = {
    "染色体级基因组 FASTA": bool(fa_files),
    "注释 (GTF/GFF)":        bool(ann_files),
    "Drees 数据表":          bool(tables),
}
for k, v in checks.items():
    out(f"  [{'✓' if v else '✗'}] {k}")

out("")
out("下一步:")
out("  1) 把本报告 (logs/03_verify_report.txt) 贴回给我")
out("  2) 我据此编写 Phase 1 脚本:")
out("     - 10_build_irs_labels.py   (从 Drees DEG 表构造正/负类)")
out("     - 11_extract_features.py   (启动子/UTR/CDS 序列特征)")
out("     - 12_train_irs.py          (训练 + 交叉验证 + 打分)")
out("     - 20_design_sgrna.py       (M1' sgRNA)")
out("     - 30_design_payload.py     (M2' 载荷)")

REPORT.write_text("\n".join(_lines), encoding="utf-8")
print(f"\n[报告已写入] {REPORT}")
