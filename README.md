# Protein Fitness Prediction on Segmented, Genotype-Split HIS3 Fitness Landscape

**Fortgeschrittenen-Praktikum Bioinformatik (IN5073) WS25/26 – Rostlab, TUM**
---

## Overview

This repository contains the code, splits, and analysis for a benchmark of sequence-based protein fitness predictors on the HIS3 protein under a segmented, genotype-based out-of-distribution (OOD) split. The goal is to evaluate whether multiple sequence alignments (MSAs), pairwise coevolutionary signals, or 3D/biophysical features improve OOD generalization over a standard protein language model (PLM) when only shallow DMS training data is available.

## Dataset

The final splits are published on Hugging Face:
**[julkuhn/HIS3InterspeciesEpistasis](https://huggingface.co/datasets/julkuhn/HIS3InterspeciesEpistasis)**
483,609 multi-mutant HIS3 variants from ProteinGym (`HIS7_YEAST_Pokusaeva_2019`), segmented, genotype-based train/val/test split across 8 segments (S02–S08, S12)


## Repository Structure

```
.
├── analysis
│   ├── compute_all_metrics.py
│   ├── compute_functional_metrics.py
│   ├── functional_metrics.csv
│   ├── plot_distance_analysis.py
│   ├── plot_extended_correlations.py
│   └── plot_factor_analysis.py
├── data
│   └── preprocessing_yeast.ipynb
├── .gitignore
├── models
│   ├── esm
│   │   ├── run_esm_deepen.sh
│   │   └── run_esm.sh
│   ├── metl_global
│   │   └── run_metl_global.sh
│   ├── metl_local
│   │   ├── args
│   │   │   ├── energize_his3_S06.txt
│   │   │   ├── finetune_his3_S06_1D.txt
│   │   │   ├── finetune_his3_S06_linear_extract.txt
│   │   │   ├── finetune_his3_S06_local.txt
│   │   │   ├── finetune_his3_S06.txt
│   │   │   ├── finetune_his3_S08_3D.txt
│   │   │   ├── finetune_his3_S08.txt
│   │   │   ├── pretrain_his3_45k_1D.txt
│   │   │   ├── pretrain_his3_45k.txt
│   │   │   └── pretrain_his3_S06.txt
│   │   ├── python
│   │   │   ├── create_rosetta_db.py
│   │   │   └── prepare_metl_splits.py
│   │   └── slurm
│   │       ├── all_combined_jobs.sh
│   │       ├── finetune
│   │       │   ├── 07_finetune_target6.sh
│   │       │   ├── 07_finetune_target.sh
│   │       │   ├── 08_finetune_S06_3D.sh
│   │       │   ├── 08_finetune_S08_3D.sh
│   │       │   ├── finetune_local_45k_1D_array.sh
│   │       │   ├── finetune_local_45k_array.sh
│   │       │   ├── finetune_local_45k_S05swap.sh
│   │       │   ├── finetune_local_45k_S06.sh
│   │       │   ├── finetune_local_45k_v2_array.sh
│   │       │   ├── finetune_local_45k_v2_S05swap.sh
│   │       │   ├── finetune_local_45k_v3_array.sh
│   │       │   ├── finetune_local_45k_v3_S05swap.sh
│   │       │   ├── finetune_local_linear_extract_S06.sh
│   │       │   └── metl_target_train.sbatch
│   │       ├── metl_local.sh
│   │       ├── pretrain
│   │       │   ├── 05_pretrain_source.sh
│   │       │   ├── metl_source_train.sbatch
│   │       │   ├── pretrain_local_45k_1D.sh
│   │       │   └── pretrain_local_45k.sh
│   │       └── rosetta
│   │           ├── 02_prepare_pdb.sh
│   │           ├── 03_energize_array.sh
│   │           ├── 04_process_results.sh
│   │           ├── rebuild_rosetta_db_45k.sh
│   │           ├── rosetta_array.sbatch
│   │           ├── rosetta_energize_array.sbatch
│   │           ├── rosetta_energize.sbatch
│   │           ├── rosetta_relax.sbatch
│   │           └── rosetta_single.sbatch
│   ├── msa_transformer
│   │   ├── msa_embed_checkpointed.py
│   │   ├── msa_transformer_baseline.py
│   │   ├── run_msa_all_chained.sh
│   │   └── run_msa_transformer.sh
│   ├── pairformer
│   │   ├── pairformer_mlp.py
│   │   └── run_pairformer.sh
│   └── proteinnpt
│       ├── protein_npt.py
│       ├── run_npt_inference.py
│       └── run_protein_npt.sh
├── README.md
└── splits
    ├── run_splitting.py
    └── run_splitting.sh

16 directories, 64 files
```

External repositories (METL, METL-sim, ESM, MSA Transformer, Pairformer, ProteinNPT) and model checkpoints are **not tracked**.

### Data
Updated split is on Hugging Face. 
Original data sources:
- Pokusaeva et al. (2019). *PLoS Genet.* 15(4), e1008079.
- Notin et al. (2023). *NeurIPS 2023 Datasets & Benchmarks*.
