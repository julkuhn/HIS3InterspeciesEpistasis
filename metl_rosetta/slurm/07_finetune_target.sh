#!/bin/bash
#SBATCH --job-name=finetune_S08
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_S08_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_S08_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/:/workspace

export PYTHONPATH=/workspace/.pip_packages
export PIP_CACHE_DIR=/workspace/.cache/pip
mkdir -p "$PIP_CACHE_DIR"

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi 2>&1 | head -5

pip3 install --target=/workspace/.pip_packages torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip3 install --target=/workspace/.pip_packages fair-esm transformers matplotlib scikit-learn scipy pandas numpy 2>/dev/null

python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd /workspace/metl
python3 code/train_target_model_pl2.py \
            @/workspace/metl_rosetta/args/finetune_his3_S08.txt