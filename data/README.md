# Data

This repository does **not** contain data. The raw and derived data are public and are not redistributed here.

## Sources

| Resource | Accession / location |
|---|---|
| Planarian genome assembly `schMedS3_h1` and annotation | Zenodo record **13798866** |
| TBK1-knockdown RNA-seq, 41 samples | Zenodo record **17072252** (also 19045792) |
| Cross-perturbation 1: RNAi knockdown | BioProject **PRJNA319245** |
| Cross-perturbation 2: regeneration time course | BioProject **PRJNA319973** |
| Cross-perturbation 3: gamma irradiation | BioProject **PRJNA319241** |
| Human hepatocytes, interferon-stimulated | GEO **GSE351189** |
| Human iPSC-derived macrophages, interferon-stimulated | GEO **GSE309167** |
| Human genome and annotation | Ensembl release **110** (GENCODE v44) |
| Pretrained nucleotide language model | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (HuggingFace) |
| Analysis archive | Zenodo concept DOI **10.5281/zenodo.23855543** |

The planarian genome and annotation must come from the same Zenodo record. Mixing the NCBI annotation with this assembly fails silently because scaffold names differ.

## Expected layout

### Planarian (`IETX_ROOT`)

```
IETX_ROOT/
├── data/          derived feature, label and result tables (from Zenodo archive)
├── ref/           genome FASTA, annotation, indices
├── env/           optional conda/micromamba environment
└── logs/
```

`code/planarian/02_download_data.sh` downloads the public data. `code/planarian/scripts/` expects the derived tables under `IETX_ROOT/data/` in the layout of `IE-Tx_analysis_20261101/data/`.

### Human (`IETX_HUMAN`)

```
IETX_HUMAN/
├── ref/
├── GSE351189/
├── GSE309167/
└── logs/
```

`code/human/run_all.sh` documents the order. Derived tables are also in the `IE-Tx_human_cross_species_20261109` archive on the same Zenodo record.

### Increment (`code/increment/`)

```
code/increment/
├── data_t01/
│   ├── label_down_tp3_cpm.tsv
│   └── label_both_tp3_cpm.tsv
├── data_human/
│   ├── features.tsv
│   ├── clusters.pkl
│   └── label_both.tsv
└── out/            created by the scripts
```

### Expression baseline (`code/expression/`)

```
code/expression/
├── data_t01/
│   ├── label_down_tp3_cpm.tsv
│   └── label_both_tp3_cpm.tsv
├── counts.tsv
└── out_expression/    created by the script
```

## Rebuilding derived tables

If you start from raw reads, run the planarian and human pipelines in order. They perform quantification, differential expression, label construction, feature extraction, clustering and model evaluation. If you start from the archived derived tables, place them in the layouts above and run the increment/expression scripts directly.

## Checksums

Checksums for the archived tables are in the Zenodo record. The repository does not embed a checksum manifest because the data are versioned separately.
