#!/bin/bash
#SBATCH --job-name=finetune_local_45k
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_45k_%A_%a.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_45k_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace

set -eo pipefail

# Segment config arrays (index 0-7)
SEGS=(S02 S03 S04 S05 S06 S07 S08 S12)
DS_NAMES=(his3_S02_thr0.08 his3_S03_thr0.1 his3_S04_thr0.08 his3_S05_thr0.25 his3_S06_thr0.18 his3_S07_thr0.05 his3_S08_thr0.18 his3_S12_thr0.1)
SPLIT_DIRS=(
    /workspace/metl_rosetta/metl_splits/S02_thr0.08
    /workspace/metl_rosetta/metl_splits/S03_thr0.1
    /workspace/metl_rosetta/metl_splits/S04_thr0.08
    /workspace/metl_rosetta/metl_splits/S05_thr0.25
    /workspace/metl_rosetta/metl_splits/S06_thr0.18
    /workspace/metl_rosetta/metl_splits/S07_thr0.05
    /workspace/metl_rosetta/metl_splits/S08_thr0.18
    /workspace/metl_rosetta/metl_splits/S12_thr0.1
)

IDX=${SLURM_ARRAY_TASK_ID}
SEG=${SEGS[$IDX]}
DS_NAME=${DS_NAMES[$IDX]}
SPLIT_DIR=${SPLIT_DIRS[$IDX]}
PRETRAINED_CKPT=/workspace/metl_rosetta/source_model_45k/maSKy8WY/checkpoints/epoch=99-step=12600-val_loss=18.58.ckpt

export PYTHONPATH=/workspace/.pip_packages_pretrain:/workspace/.pip_packages
export PIP_CACHE_DIR=/workspace/.cache/pip
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" /workspace/.cache/huggingface

echo "=== Finetune METL-Local 45k: ${SEG} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Dataset: ${DS_NAME}"
echo "Split dir: ${SPLIT_DIR}"
echo "Checkpoint: ${PRETRAINED_CKPT}"
nvidia-smi 2>&1 | head -5 || true

# Install pinned packages into isolated dir (skip if already done by pretrain job)
if [ ! -f /workspace/.pip_packages_pretrain/torch/version.py ]; then
    pip3 install --target=/workspace/.pip_packages_pretrain \
        torch --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null
    pip3 install --target=/workspace/.pip_packages_pretrain \
        "pytorch-lightning==1.9.5" \
        "torchmetrics>=0.7.0,<1.0" \
        "lightning-utilities>=0.6.0,<0.9.0" \
        "transformers==4.44.2" \
        "numpy<2.0" matplotlib scikit-learn scipy pandas --quiet 2>/dev/null
fi

python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd /workspace/metl
python3 code/train_target_model_pl2.py \
    --ds_name "${DS_NAME}" \
    --split_dir "${SPLIT_DIR}" \
    --train_name train \
    --val_name val \
    --test_name test \
    --target_names DMS_score \
    --pdb_fn /workspace/metl/data/pdb_files/HIS7_YEAST_p.pdb \
    --model_name transfer_model \
    --pretrained_ckpt_path "${PRETRAINED_CKPT}" \
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
    --experiment "his3_${SEG}_local_45k" \
    --log_dir_base /workspace/metl_rosetta/target_model_45k

echo "Done: $(date)"
