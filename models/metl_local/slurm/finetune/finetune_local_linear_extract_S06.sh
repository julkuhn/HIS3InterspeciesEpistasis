#!/bin/bash
#SBATCH --job-name=metl_local_linear_extract_S06
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_linear_extract_S06_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_linear_extract_S06_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace

set -euo pipefail

export PYTHONPATH=/workspace/.pip_packages
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p "$PIP_CACHE_DIR"

echo "=== Finetune METL-Local (linear head, extract) on S06 ==="
echo "Node: $(hostname)"
nvidia-smi 2>&1 | head -5 || true

cd /workspace/metl
python3 code/train_target_model_pl2.py \
    --ds_name his3_S06 \
    --split_dir /workspace/metl_rosetta/metl_data/his3_S06_split \
    --train_name train \
    --val_name val \
    --test_name test \
    --target_names score \
    --pdb_fn /workspace/metl/data/pdb_files/HIS7_YEAST_p.pdb \
    --model_name transfer_model \
    --pretrained_ckpt_path /workspace/metl_rosetta/source_model_45k/maSKy8WY/checkpoints/epoch=99-step=12600-val_loss=18.58.ckpt \
    --encoding int_seqs \
    --backbone_cutoff -1 \
    --dropout_after_backbone \
    --dropout_after_backbone_rate 0.5 \
    --top_net_type linear \
    --finetuning \
    --finetuning_strategy extract \
    --optimizer adam \
    --weight_decay 0.0 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --gradient_clip_val 0.5 \
    --batch_size 64 \
    --no_use_wandb \
    --experiment his3_S06_local_linear_extract \
    --log_dir_base /workspace/metl_rosetta/target_model_45k

echo "Done: $(date)"
