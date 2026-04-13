#!/bin/bash
#SBATCH --job-name=pretrain_local_45k
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pretrain_45k_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pretrain_45k_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace


# Step 5: Pretrain METL-Local source model on Rosetta energies.
#
# Prerequisites:
#   - Step 4 completed (DB + Splits + Standardisierungsparameter existieren)
#
# Usage: sbatch 05_pretrain_source.sh

export PYTHONPATH=/workspace/.pip_packages
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p "$PIP_CACHE_DIR"

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi 2>&1 | head -5

pip3 install --target=/workspace/.pip_packages torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip3 install --target=/workspace/.pip_packages "pytorch-lightning==1.9.5" "torchmetrics>=0.7.0,<1.0" "lightning-utilities>=0.6.0,<0.9.0" "transformers==4.44.2" fair-esm matplotlib scikit-learn scipy pandas numpy 2>/dev/null

python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

 cd /workspace/metl
        python3 code/train_source_model.py \
            @/workspace/metl_rosetta/args/pretrain_his3_45k.txt