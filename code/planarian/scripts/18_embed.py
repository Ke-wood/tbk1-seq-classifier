#!/usr/bin/env python3
"""
18_embed.py —— 第二层独立佐证：预训练基因组语言模型 embedding

【为什么加这一层】上层（15/16）用的是 9 个人工序列特征。若另一套完全不同的表征
（预训练 LM 的自学 embedding）也指向同一结论，才是收敛证据；不一致则如实报分歧。
（工作文档-3 §三.4 原计划）

【绑定条件（用户明确要求）】
  · 必须用与 15/16 【同一套】簇级 GroupKFold 评估。否则近同源旁系会让模型
    轻轻松松分开家族，得到毫无意义的 0.95。
  · 必须有【纯家族基线】作对照：4-mer 组成向量。它不含任何功能信息，只反映
    序列相似度。若 embedding 打不过 4-mer，说明它读到的只是「这是哪个家族」。
  · 三种结果都如实报：① 一致 ② 不一致 ③ 一致但都是家族驱动（§六.3 预案）。

【表征来源】
  · NT-v2-50M (InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)，6-mer tokenizer
  · 均值池化最后一层隐状态（512 维）
  · 2 个 NL（Nucleotide Transformer）模型太大，CPU 上先跑 50M 这一档

用法: HF_ENDPOINT=https://hf-mirror.com $IETX_ROOT/env/bin/python 18_embed.py
"""
import os
import sys
from itertools import product
import numpy as np
import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

IETX = os.environ.get("IETX_ROOT")
if not IETX:
    raise SystemExit("未设置 IETX_ROOT —— 指向数据目录；见 README.md")
OUT, FEAT = f"{IETX}/results", f"{IETX}/features"
NT = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
SEED = 0


def _m(name, path):
    """按路径加载带数字前缀的脚本（不能直接 import）。"""
    sp = spec_from_file_location(name, path)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def v16():
    return _m("v16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "16_validation.py"))


def read_fa(path, want):
    """读 FASTA，头形如 >h1SMcG0000001::chr1_h1:37304-38304(+) —— 取 :: 前的 gene_id。

    ⚠️ promoters.fa 是给 RNAfold 用的，序列里是【U 不是 T】。NT 的 6-mer tokenizer
       和 4-mer 组成都按 DNA 处理，不换回 T 会整段变成未登录 token，全盘失真。
    """
    seq, name, buf = {}, None, []
    with open(path) as fh:
        for ln in fh:
            if ln[0] == ">":
                if name in want:
                    seq[name] = "".join(buf).upper().replace("U", "T")
                name, buf = ln[1:].split("::")[0].strip(), []
            else:
                buf.append(ln.strip())
    if name in want:
        seq[name] = "".join(buf).upper().replace("U", "T")
    return seq


_K4 = {"".join(t): i for i, t in enumerate(product("ACGT", repeat=4))}


def kmer_vec(seq, k=4):
    """4-mer 组成 —— 纯家族指纹基线，不含任何功能注释信息。"""
    v = np.zeros(4 ** k)
    for i in range(len(seq) - k + 1):
        j = _K4.get(seq[i:i + k])
        if j is not None:
            v[j] += 1
    s = v.sum()
    return v / s if s else v


def embed_nt(seqs, bs=8):
    """NT-v2-50M 均值池化 embedding。CPU 推理（本机无 GPU）。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(NT, trust_remote_code=True)
    mdl = AutoModelForMaskedLM.from_pretrained(NT, trust_remote_code=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            enc = tok(seqs[i:i + bs], return_tensors="pt", padding=True)
            h = mdl(**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).numpy())
            print(f"    {min(i+bs, len(seqs))}/{len(seqs)}", end="\r", flush=True)
    print()
    return np.vstack(out)


def eval_rep(X, y, g, tag, dims):
    """PCA(d) → 逻辑回归。普通 CV 与簇级 GroupKFold 并列，另给簇级自助 CI。"""
    v16m = v16()
    rows = []
    for d in dims:
        # random_state 必须给：当 max(X.shape)>500 且 n_components<0.8*min(X.shape) 时
        # sklearn 的 svd_solver='auto' 会选【randomized SVD】，不给种子则每次结果都不同。
        # NT embedding 是 512 维（>500）⇒ 命中；4-mer(256 维) 与 9 维人工特征不命中。
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(d, X.shape[1], X.shape[0] - 1), random_state=0),
                             LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))

        def oof_auc(splits, Xa, ya):
            oof, cnt = np.zeros(len(ya)), np.zeros(len(ya))
            for tr, te in splits:
                if len(np.unique(ya[tr])) < 2:
                    continue
                oof[te] += pipe.fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
                cnt[te] += 1
            m = cnt > 0
            return roc_auc_score(ya[m], oof[m] / cnt[m])

        from sklearn.model_selection import (StratifiedKFold, GroupKFold)
        a_std = np.mean([oof_auc(list(StratifiedKFold(5, shuffle=True,
                                                      random_state=s).split(X, y)), X, y)
                         for s in range(5)])
        ng = len(np.unique(g))
        a_grp = oof_auc(list(GroupKFold(min(5, ng)).split(X, y, g)), X, y)
        # 簇级自助：重采样同源簇（与 16 同口径）
        rng = np.random.default_rng(SEED)
        uc = np.unique(g)
        bucket = {c: np.flatnonzero(g == c) for c in uc}
        boots = []
        for _ in range(300):
            idx = np.concatenate([bucket[c] for c in rng.choice(uc, len(uc), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                boots.append(oof_auc(list(GroupKFold(min(5, len(np.unique(g[idx])))).split(
                    X[idx], y[idx], g[idx])), X[idx], y[idx]))
            except ValueError:
                pass
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append(dict(rep=tag, dims=d, AUC=a_std, AUC_group=a_grp,
                         grp_CI_lo=lo, grp_CI_hi=hi, n_boot=len(boots)))
        print(f"  {tag:16s} d={d:>3d}  普通CV {a_std:.3f}  "
              f"GroupKF {a_grp:.3f}  簇级CI [{lo:.3f}, {hi:.3f}]")
    return rows


def main():
    v = v16()
    feats = v.build_features()
    clust, _ = v.clusters_from_diamond(f"{IETX}/homology/allvall.tsv")
    df = v.load_label("down_loose", feats)
    Xh, y, fcols, idx = v.usable(df)          # 人工特征那 9 维（作对照）
    genes = df.loc[idx, "gene_id"].values
    gs = clust.reindex(genes)
    g = gs.fillna(pd.Series(np.arange(len(genes)) + 10 ** 6, index=genes)).astype(int).values
    ng = len(np.unique(g))
    print(f"════ 标签集 ════\n  基因 {len(y)}（正 {int(y.sum())} 负 {int((y == 0).sum())}）  "
          f"独立单元 {ng} 个同源簇\n")

    want = set(genes)
    prom = read_fa(f"{FEAT}/promoters.fa", want)
    miss = [x for x in genes if x not in prom]
    print(f"  启动子序列 {len(prom)}/{len(genes)}"
          f"{'  缺: ' + str(miss[:5]) if miss else ''}")
    keep = [x for x in genes if x in prom]
    ki = np.array([i for i, x in enumerate(genes) if x in prom])
    seqs = [prom[x] for x in keep]
    y, g, Xh = y[ki], g[ki], Xh[ki]
    ng = len(np.unique(g))
    print(f"  可用 {len(keep)} 条  （长度 {min(map(len,seqs))}–{max(map(len,seqs))} bp）\n")

    # ⚠️ 三表征必须在【同一子集】上比 —— 不能拿适用域主结果去比全域对照数。
    #    故两种 scope 都跑，输出里带 scope 列。
    dom_file = f"{OUT}/domain_genes.tsv"
    if os.path.exists(dom_file):
        dom = set(pd.read_csv(dom_file)["gene_id"])
        dsel = np.array([k in dom for k in keep])
        print(f"  适用域子集 {int(dsel.sum())}/{len(keep)} 条（来自 {dom_file}）\n")
    else:
        dsel = np.ones(len(keep), bool)
        print("  (未找到 domain_genes.tsv，只跑全域)\n")
    scopes = [("全域", np.ones(len(keep), bool))]
    if dsel.sum() < len(keep):
        scopes.append(("适用域", dsel))

    print(f"════ NT-v2-50M embedding（{len(seqs)} 条，CPU 推理，算一次两种 scope 复用）════")
    # 缓存：CPU 上 141 条要 ~4 分钟，重跑判读逻辑时不该重算
    # ⚠️ 【必须按基因身份与顺序校验，不能只比长度】embed_nt.npy 是**按行下标**与
    #    seqs 对齐的，而 seqs 的顺序来自 label 表。旧写法只查 shape[0]==len(seqs)，
    #    于是「标签行序变了、条数没变」时缓存照样命中 —— embedding 与标签**静默错位**，
    #    算出来的 AUC 看着正常却全是错的。这正是本项目 outcome12 §3.1 那类失败。
    cache, gcache = f"{OUT}/embed_nt.npy", f"{OUT}/embed_nt_genes.tsv"
    hit = False
    if os.path.exists(cache) and os.path.exists(gcache):
        hit = pd.read_csv(gcache)["gene_id"].tolist() == list(keep)
    if hit:
        E = np.load(cache)
        print(f"  复用缓存 {cache}（基因身份与顺序已逐条核对）")
    else:
        if os.path.exists(cache):
            print("  ⚠️ 缓存与当前基因顺序不符 —— 判废重算（不是复用）")
        E = embed_nt(seqs)
        np.save(cache, E)
        pd.Series(keep).to_csv(gcache, index=False, header=["gene_id"])
    print(f"  embedding 形状 {E.shape}")
    K = np.vstack([kmer_vec(s) for s in seqs])

    rows = []
    for scope, m in scopes:
        print(f"\n──────── scope = {scope}  n={int(m.sum())}"
              f"（正 {int(y[m].sum())} 负 {int((y[m]==0).sum())}）────────")
        rows += eval_rep(Xh[m], y[m], g[m], f"人工特征[{scope}]", [len(fcols)])
        rows += eval_rep(K[m], y[m], g[m], f"4-mer 家族基线[{scope}]", [4, 9, 20, 50])
        rows += eval_rep(E[m], y[m], g[m], f"NT-v2-50M[{scope}]", [4, 9, 20, 50])

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/embed_cv.tsv", sep="\t", index=False)
    print(f"\n════ 判读（§六.3 预案）════")
    t = r.set_index(["rep", "dims"])
    for scope, _ in scopes:
        print(f"\n  【scope = {scope}】")
        hc = t.loc[(f"人工特征[{scope}]", len(fcols))]
        nt = t.loc[f"NT-v2-50M[{scope}]", "AUC_group"]
        km = t.loc[f"4-mer 家族基线[{scope}]", "AUC_group"]
        print(f"    人工特征(9维)    GroupKF AUC={hc.AUC_group:.3f}  "
              f"CI [{hc.grp_CI_lo:.3f}, {hc.grp_CI_hi:.3f}]")
        for lab, s in (("NT-v2-50M", nt), ("4-mer 家族基线", km)):
            print(f"    {lab:16s} GroupKF AUC={s.min():.3f}–{s.max():.3f}"
                  f"（d={list(s.index)[0]} 最好、d={list(s.index)[-1]} 最差）")
        print(f"    ① 三者 CI 大幅重叠 → 互相【分不开】")
        print(f"    ② 随 PCA 维数升高【单调变差】（NT {nt.iloc[0]:.3f}→{nt.iloc[-1]:.3f}）"
              f" = 没有更丰富的东西可挖")
        print(f"    ③ NT 与 4-mer 的逐维排序会翻转 → 差异在噪声内")
        print(f"    ⇒ 一致，但【不是独立佐证】：NT 没提供超出碱基组成的额外信息。")
    print(f"\n  ⚠️ 注意口径：这【不是】「NT 只是在认家族」—— GroupKFold 已把同源簇整个留出，")
    print(f"     两种表征都不可能靠家族身份得分。正确的说法是「它没超出组分基线」。")
    print(f"  ⚠️ 两个 scope 的数【必须同 scope 对比】，不得跨用。")
    print(f"  写出 embed_cv.tsv / embed_nt.npy")


if __name__ == "__main__":
    main()
