# Protein Fitness Prediction on Segmented, Genotype-Split HIS3 Fitness Landscape

**Fortgeschrittenen-Praktikum Bioinformatik (IN5073) WS25/26 – Rostlab, TUM**
---

## Overview

This repository contains the code, splits, and analysis for a benchmark of sequence-based protein fitness predictors on the HIS3 protein under a segmented, genotype-based out-of-distribution (OOD) split. The goal is to evaluate whether multiple sequence alignments (MSAs), pairwise coevolutionary signals, or 3D/biophysical features improve OOD generalization over a standard protein language model (PLM) when only shallow DMS training data is available.

## Dataset

The curated splits are published on Hugging Face:
**[julkuhn/HIS3InterspeciesEpistasis](https://huggingface.co/datasets/julkuhn/HIS3InterspeciesEpistasis)**

- 483,609 multi-mutant HIS3 variants from ProteinGym (`HIS7_YEAST_Pokusaeva_2019`)
- Segmented, genotype-based train/val/test split across 8 segments (S02–S08, S12)
- Reference sequence: HIS7_ASHGO (*Ashbya gossypii*, UniProt Q75B47)
- Each variant annotated with segment, mutation count, and Hamming distance to the nearest orthologue


## Repository Structure

```
.
├── compute_all_metrics.py
├── .gitignore
├── metl_rosetta
│   ├── args
│   │   ├── energize_his3_S06.txt
│   │   ├── finetune_his3_S06_1D.txt
│   │   ├── finetune_his3_S06_linear_extract.txt
│   │   ├── finetune_his3_S06_local.txt
│   │   ├── finetune_his3_S06.txt
│   │   ├── finetune_his3_S08_3D.txt
│   │   ├── finetune_his3_S08.txt
│   │   ├── pretrain_his3_45k_1D.txt
│   │   ├── pretrain_his3_45k.txt
│   │   └── pretrain_his3_S06.txt
│   ├── compute_functional_metrics.py
│   ├── functional_metrics.csv
│   ├── metl_source_train.sbatch
│   ├── metl_target_train.sbatch
│   ├── plot_distance_analysis.py
│   ├── plot_extended_correlations.py
│   ├── plot_factor_analysis.py
│   ├── prepare_metl_splits.py
│   ├── rosetta_array.sbatch
│   ├── rosetta_energize_array.sbatch
│   ├── rosetta_energize.sbatch
│   ├── rosetta_relax.sbatch
│   ├── rosetta_single.sbatch
│   └── scripts
│       ├── create_rosetta_db.py
│       └── pairformer_mlp.py
├── preprocessing_yeast.ipynb
├── README.md
├── run_splitting.py
└── run_splitting.sh
```

External repositories (METL, METL-sim, ESM, MSA Transformer, Pairformer, ProteinNPT) and model checkpoints are **not tracked**.

### Data
Updated split is on Hugging Face. 
Original data sources:
- Pokusaeva et al. (2019). *PLoS Genet.* 15(4), e1008079.
- Notin et al. (2023). *NeurIPS 2023 Datasets & Benchmarks*.
