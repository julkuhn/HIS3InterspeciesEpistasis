# Protein Fitness Prediction on Segmented, Genotype-Split HIS3 Fitness Landscape

**Fortgeschrittenen-Praktikum Bioinformatik (IN5073) WS25/26 – Rostlab, TUM**

---

## Overview

This project benchmarks sequence-based protein fitness predictors on the HIS3 protein (yeast imidazoleglycerol-phosphate dehydratase) under a segmented, genotype-based out-of-distribution (OOD) split. The central question is whether access to multiple sequence alignments (MSAs), pairwise coevolutionary signals, or 3D/biophysical features improves OOD generalization compared to a standard protein language model (ESM-2), given only shallow DMS training data.

**Models benchmarked:**
- ESM-2 (baseline PLM)
- MSA Transformer (coevolutionary signal)
- ProteinNPT (semi-supervised + MSA)
- METL-Global (biophysical pretraining, sequence-only)
- METL-Local (biophysical pretraining with Rosetta energies)
- Pairformer (structure-informed pairwise features)

---

## Dataset

The processed splits are published on Hugging Face:
**[julkuhn/HIS3InterspeciesEpistasis](https://huggingface.co/datasets/julkuhn/HIS3InterspeciesEpistasis)**

- 483,609 multi-mutant HIS3 variants from ProteinGym (`HIS7_YEAST_Pokusaeva_2019`)
- Segmented, genotype-based train/val/test split across 8 segments (S02–S08, S12)
- Each segment corresponds to a contiguous region of the protein; train/test splits are defined such that no genotype appears in both train and test (OOD by construction)

**Original data sources:**
- Pokusaeva et al. (2019). *PLoS Genet.* 15(4), e1008079.
- Notin et al. (2023). *NeurIPS 2023 Datasets & Benchmarks* (ProteinGym).

---

## Repository Structure

Only scripts, configs, and analysis code are tracked. External repos, model weights, Rosetta outputs, embeddings, and result files are gitignored (see `.gitignore`).

```
.
├── analysis/                          # Metric computation and plotting
│   ├── compute_all_metrics.py         # Aggregate Spearman/NDCG across all models and splits
│   ├── compute_functional_metrics.py  # Per-split functional performance metrics
│   ├── functional_metrics.csv         # Precomputed metric table
│   ├── plot_distance_analysis.py      # Performance vs. sequence distance to train set
│   ├── plot_extended_correlations.py  # Cross-model correlation plots
│   └── plot_factor_analysis.py        # Factor/PCA analysis of predictions
│
├── data/
│   └── preprocessing_yeast.ipynb      # Raw ProteinGym data → HIS3 splits
│
├── models/
│   ├── esm/
│   │   ├── run_esm.sh                 # ESM-2 embedding + linear probe (SLURM)
│   │   └── run_esm_deepen.sh          # ESM-2 with deeper MLP head
│   │
│   ├── metl_global/
│   │   └── run_metl_global.sh         # METL-Global fine-tuning (SLURM)
│   │
│   ├── metl_local/                    # METL-Local: Rosetta-pretrained, structure-aware
│   │   ├── args/                      # Argument files for METL pretraining/fine-tuning
│   │   │   ├── energize_his3_S06.txt
│   │   │   ├── finetune_his3_S06.txt
│   │   │   ├── finetune_his3_S06_1D.txt
│   │   │   ├── finetune_his3_S06_linear_extract.txt
│   │   │   ├── finetune_his3_S06_local.txt
│   │   │   ├── finetune_his3_S08_3D.txt
│   │   │   ├── finetune_his3_S08.txt
│   │   │   ├── pretrain_his3_45k.txt
│   │   │   ├── pretrain_his3_45k_1D.txt
│   │   │   └── pretrain_his3_S06.txt
│   │   ├── python/
│   │   │   ├── create_rosetta_db.py   # Build SQLite variant database from Rosetta outputs
│   │   │   └── prepare_metl_splits.py # Format splits for METL training
│   │   └── slurm/
│   │       ├── all_combined_jobs.sh
│   │       ├── metl_local.sh
│   │       ├── finetune/              # Fine-tuning job scripts (array + single)
│   │       ├── pretrain/              # Pretraining job scripts
│   │       └── rosetta/               # PDB prep, energize array, result processing
│   │
│   ├── msa_transformer/
│   │   ├── msa_embed_checkpointed.py  # Memory-efficient MSA embedding with grad checkpointing
│   │   ├── msa_transformer_baseline.py
│   │   ├── run_msa_transformer.sh     # Per-split MSA Transformer runs (SLURM)
│   │   └── run_msa_all_chained.sh     # Chain all segments sequentially
│   │
│   ├── pairformer/
│   │   ├── pairformer_mlp.py          # MLP head on top of Pairformer pairwise embeddings
│   │   └── run_pairformer.sh
│   │
│   └── proteinnpt/
│       ├── protein_npt.py             # ProteinNPT wrapper / inference script
│       ├── run_npt_inference.py
│       └── run_protein_npt.sh
│
├── splits/
│   ├── run_splitting.py               # Generate segmented genotype-split CSV files
│   └── run_splitting.sh
│
├── .gitignore
└── README.md
```

**Gitignored (not tracked):**

| Path | Contents |
|------|----------|
| `/metl/`, `/metl-sim/`, `/metl-pretrained/`, `/metl-pub/` | Cloned METL repositories |
| `/esm/`, `/msa_transformer/`, `/pairformer/`, `/proteinNPT/`, `/his3_metl/` | Cloned model repos + run outputs |
| `metl_rosetta/` (most subdirs) | Rosetta energize outputs, model checkpoints, variant databases |
| `plots/` | Generated figures |
| `*.pt`, `*.safetensors`, `*.ckpt`, `*.pdb`, `*.a3m` | Model weights, structures, MSAs |

---

## Workflow

### 1. Data Preprocessing

Run `data/preprocessing_yeast.ipynb` to download the raw ProteinGym HIS3 data and produce a cleaned variant table.

### 2. Split Generation

```bash
cd splits/
bash run_splitting.sh
```

Produces per-segment train/val/test CSV files. The final version is also available directly from Hugging Face (see above).

### 3. Model Training / Inference

Each model has a dedicated subdirectory under `models/` with SLURM job scripts. The general pattern is:

```bash
# Example: ESM-2 baseline
sbatch models/esm/run_esm.sh

# Example: MSA Transformer (all segments chained)
bash models/msa_transformer/run_msa_all_chained.sh

# Example: METL-Local (requires Rosetta precomputation first)
# Step 1: energize variants with Rosetta
sbatch models/metl_local/slurm/rosetta/rosetta_energize_array.sbatch
# Step 2: build the Rosetta DB
python models/metl_local/python/create_rosetta_db.py
# Step 3: pretrain source model
sbatch models/metl_local/slurm/pretrain/pretrain_local_45k.sh
# Step 4: fine-tune on target split
sbatch models/metl_local/slurm/finetune/finetune_local_45k_array.sh
```
---

## External Dependencies

The following repositories are cloned locally but not tracked in this repo:

| Repo | Purpose |
|------|---------|
| [METL](https://github.com/gitter-lab/metl) | METL training framework |
| [metl-pretrained](https://github.com/gitter-lab/metl-pretrained) | Pretrained METL checkpoints |
| [metl-sim](https://github.com/gitter-lab/metl-sim) | Rosetta simulation pipeline |
| [metl-pub](https://github.com/gitter-lab/metl-pub) | METL publication code |
| [ESM](https://github.com/facebookresearch/esm) | ESM-2|
| [ProteinNPT](https://github.com/OATML-Markslab/ProteinNPT) | ProteinNPT |
