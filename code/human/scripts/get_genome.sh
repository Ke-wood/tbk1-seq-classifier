#!/usr/bin/env bash
# get_genome.sh —— 分段并发 + 断点续传下载 GRCh38 主装配（Ensembl 110）
# ⚠️ 这条线路不稳定：单流 ~170 KB/s，aria2 多线程会在大文件尾部卡死，
#    curl 一次性 range 会被 900 s 超时截断。故按段切分、每段 -C - 续传、直到字节数对上。
set -uo pipefail
R=${IETX_HUMAN:?set IETX_HUMAN}
cd "$R/ref"
U="https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
OUT=S.full.fa.gz
N=881964081
P=12
mkdir -p chunks
SZ=$(( (N + P - 1) / P ))

for i in $(seq 0 $((P-1))); do
  a=$(( i * SZ )); b=$(( a + SZ - 1 )); [ $b -ge $N ] && b=$(( N - 1 ))
  want=$(( b - a + 1 ))
  (
    tries=0
    while :; do
      got=$(stat -c%s "chunks/p$i" 2>/dev/null || echo 0)
      [ "$got" -eq "$want" ] && break
      [ "$got" -gt "$want" ] && { : > "chunks/p$i"; got=0; }
      # ⚠️ 不用 curl -C -：它和 -r 一起用时偏移语义容易搞错。显式算 a+got 更稳。
      timeout 600 curl -sS -r "$((a+got))-$b" "$U" >> "chunks/p$i" 2>/dev/null
      sleep 3
      tries=$((tries+1)); [ $tries -gt 40 ] && { echo "!! p$i 放弃"; exit 1; }
    done
  ) &
done
wait

cat $(for i in $(seq 0 $((P-1))); do echo "chunks/p$i"; done) > "$OUT"
[ "$(stat -c%s "$OUT")" -eq "$N" ] || { echo "!! 拼接后大小不对"; exit 1; }
gzip -t "$OUT" || { echo "!! 拼接后 gzip 校验失败"; exit 1; }
mv "$OUT" Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
echo "GENOME OK $(stat -c%s Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz)"
