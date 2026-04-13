#!/bin/bash
#SBATCH --job-name=ft45k_v2
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_45k_v2_%A_%a.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_local_45k_v2_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace

# Changes vs v1 (extract + linear):
#   - finetuning_strategy: backbone (gradual unfreeze at epoch 20, backbone LR = 0.1x head LR)
#   - top_net_type: nonlinear (256-node hidden layer)
#   - dropout_after_backbone_rate: 0.3 (down from 0.5)
#   - weight_decay: 0.01 (L2 regularization)
#   - lr_scheduler: warmup_cosine_decay
#   - max_epochs: 200 (more room for backbone to adapt)
#   - early_stopping with patience 20
#   - log_dir_base: target_model_45k_v2

set -eo pipefail

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
mkdir -p "$PIP_CACHE_DIR" /workspace/.cache/huggingface /workspace/metl_rosetta/target_model_45k_v2

echo "=== Finetune METL-Local 45k v2 (backbone): ${SEG} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Dataset: ${DS_NAME} | Split: ${SPLIT_DIR}"
nvidia-smi 2>&1 | head -5 || true

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
    --dropout_after_backbone_rate 0.3 \
    --top_net_type nonlinear \
    --top_net_hidden_nodes 256 \
    --top_net_use_dropout \
    --top_net_dropout_rate 0.2 \
    --finetuning \
    --finetuning_strategy backbone \
    --unfreeze_backbone_at_epoch 20 \
    --backbone_initial_ratio_lr 0.1 \
    --optimizer adamw \
    --weight_decay 0.01 \
    --learning_rate 0.001 \
    --lr_scheduler warmup_cosine_decay \
    --warmup_steps 0.05 \
    --max_epochs 200 \
    --gradient_clip_val 0.5 \
    --batch_size 64 \
    --early_stopping \
    --es_monitor val \
    --es_patience 20 \
    --no_use_wandb \
    --experiment "his3_${SEG}_local_45k_v2" \
    --log_dir_base /workspace/metl_rosetta/target_model_45k_v2

echo "Done: $(date)"
