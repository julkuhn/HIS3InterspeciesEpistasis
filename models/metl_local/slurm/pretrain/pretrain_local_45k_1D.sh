#!/bin/bash
#SBATCH --job-name=pretrain_local_45k_1D
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pretrain_45k_1D_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pretrain_45k_1D_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace

set -eo pipefail

export PYTHONPATH=/workspace/.pip_packages_pretrain:/workspace/.pip_packages
export PIP_CACHE_DIR=/workspace/.cache/pip
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
rm -rf /workspace/.pip_packages_pretrain
mkdir -p "$PIP_CACHE_DIR" /workspace/.pip_packages_pretrain /workspace/metl_rosetta/source_model_45k_1D

echo "=== Pretrain METL-Local Source Model 1D (45k Rosetta) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi 2>&1 | head -5 || true

pip3 install --target=/workspace/.pip_packages_pretrain \
    torch --index-url https://download.pytorch.org/whl/cu121
pip3 install --target=/workspace/.pip_packages_pretrain \
    "pytorch-lightning==1.9.5" \
    "torchmetrics>=0.7.0,<1.0" \
    "lightning-utilities>=0.6.0,<0.9.0" \
    "transformers==4.44.2" \
    "numpy<2.0" matplotlib scikit-learn scipy pandas

python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd /workspace/metl
python3 code/train_source_model.py \
    @/workspace/metl_rosetta/args/pretrain_his3_45k_1D.txt

echo "=== Finding best checkpoint ==="
SOURCE_DIR=/workspace/metl_rosetta/source_model_45k_1D
LATEST_RUN=$(ls -td ${SOURCE_DIR}/*/ 2>/dev/null | head -1)
CKPT_DIR="${LATEST_RUN}checkpoints"
BEST_CKPT=$(ls ${CKPT_DIR}/epoch=*.ckpt 2>/dev/null | sort -t= -k4 -n | head -1)
[ -z "$BEST_CKPT" ] && BEST_CKPT="${CKPT_DIR}/last.ckpt"
echo "Best checkpoint: $BEST_CKPT"
echo "$BEST_CKPT" > ${SOURCE_DIR}/best_ckpt.txt

echo "Done: $(date)"
