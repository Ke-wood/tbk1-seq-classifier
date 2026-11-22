# tbk1-seq-classifier

Analysis code for the manuscript:

> **When does longest-CDS length add predictive value beyond sequence and gene-model features? A validation protocol for gene-program classifiers**

This repository contains **code only**. It does not redistribute the RNA-seq data, genome references, derived feature tables, manuscript, figures, or internal notes. All data sources are public; see `data/README.md`.

## What the code does

The pipeline:

1. builds responder and covariate-matched control labels from public perturbation transcriptomes;
2. computes nine promoter, coding-sequence and gene-model features per gene;
3. fits an L2-regularised logistic regression under homology-cluster GroupKFold;
4. estimates cluster-level bootstrap intervals, permutation nulls, leave-one-cluster jackknife and per-feature ablation;
5. compares three models (length only, nine features, joint) on one shared cluster partition to estimate the conditional value of longest-CDS length;
6. runs an expression-only baseline and cross-perturbation negative controls;
7. repeats the length comparison in human hepatocytes and macrophages, and tests five UTR features.

## Layout

```
code/
├── planarian/                 planarian TBK1 analysis
│   ├── run_all.sh             one-command rerun (~5.5 h)
│   ├── environment.yml        conda environment (Linux)
│   ├── requirements.txt       pip pins
│   ├── scripts/               main analysis scripts 00-26
│   └── posthoc/               supplementary analyses 27-34
├── human/                     human interferon-response replication
│   ├── run_all.sh
│   ├── run_models.sh
│   ├── run_utr.sh
│   └── scripts/               label, feature, model, UTR scripts
├── increment/                 three-model length increment
│   └── scripts/70_len_increment.py
│       scripts/71_human_phh.py
└── expression/                expression-only baseline
    └── scripts/72_expression_baseline.py
        scripts/70_len_increment.py   (helper, same file)
```

## Environment

The exact version of scikit-learn matters:

- Python 3.11.16
- scikit-learn **1.9.1**
- numpy 2.4.6, pandas 2.3.3, scipy 1.17.1
- matplotlib 3.11.2 (only if you run plotting code)
- ViennaRNA 2.7.2 for RNAfold features

With scikit-learn 1.9.0 the gene-level AUC changes from 0.7723 to 0.7697 because the `RepeatedStratifiedKFold` assignment changes; with 1.6.1 the nine-feature model has been observed at 0.7518. The cluster-level value 0.7254 is stable. Pin 1.9.1 if you want to reproduce the reported numbers.

Create the environment:

```bash
conda env create -f code/planarian/environment.yml
conda activate irs
# or
pip install -r code/planarian/requirements.txt
```

Set the thread limit **before** importing numpy:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Without this, small-matrix fits are much slower and less reproducible.

## Data

No data are included. See `data/README.md` for accessions and the expected directory layout.

The planarian pipeline expects `IETX_ROOT` to point at a working directory containing `data/`, `ref/` and `logs/`. The human pipeline expects `IETX_HUMAN` to point at its data root. The increment and expression scripts expect `data_t01/`, `data_human/`, `counts.tsv` and write to `out/` under their module directory.

## Running

### Planarian

```bash
export IETX_ROOT=/path/to/IE-Tx_analysis_data
cd code/planarian
bash run_all.sh              # full pipeline, about 5.5 h
bash run_all.sh --only 16    # one step
```

### Human replication

```bash
export IETX_HUMAN=/path/to/human_data
bash code/human/run_all.sh
```

The human launcher was written for the archived data layout. If your layout differs, run the individual scripts under `code/human/scripts/` in the order documented in `run_all.sh`.

### Length increment

```bash
cd code/increment
python scripts/70_len_increment.py   # planarian down + two-sided
python scripts/71_human_phh.py       # hepatocyte extension
```

`70_len_increment.py` hard-asserts the archived point estimates and exits on mismatch:

- down: length 0.7375, nine 0.7254, joint 0.8937
- two-sided: length 0.6571, nine 0.6731, joint 0.7563

### Expression baseline

```bash
cd code/expression
python scripts/72_expression_baseline.py \
  --labels data_t01 \
  --counts /path/to/counts.tsv \
  --out out_expression
```

The script hard-asserts the nine-feature AUCs (0.7254, 0.6731) and reports the expression-only and joint models with paired cluster-bootstrap intervals.

## Reproducibility notes

- All scripts set `OMP_NUM_THREADS=1` before importing numpy.
- The three-model comparison uses one deterministic five-fold homology-cluster partition per label set, with 2,000 paired cluster resamples and refitting in every draw.
- The two planarian label sets share genes; their intervals are separate within-label descriptions, not independent replication.
- Derived tables and the full archive are deposited on Zenodo under the concept DOI `10.5281/zenodo.23855543`. A version-specific DOI is being assigned.
- The code was developed on Linux. Shell scripts require bash and coreutils; the Python scripts are platform-independent if the pinned packages are available.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, cite the manuscript above and the Zenodo archive. A `CITATION.cff` is included for GitHub's citation button.
