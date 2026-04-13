#!/bin/bash
#SBATCH --job-name=metl_target
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/metl_target_%j.out
#SBATCH --error=logs/metl_target_%j.err
#SBATCH --container-image=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra:/workspace

cd /workspace/metl
mkdir -p /workspace/logs output/training_logs

echo "=== METL Target Model Training ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Installiere Pakete
echo "Installing dependencies..."
pip3 install "numpy<2.0" --quiet 2>/dev/null
pip3 install pytorch-lightning==2.6.1 torchmetrics==1.8.2 tensorboard biopython biopandas shortuuid networkx rich wandb connectorx umap-learn seaborn matplotlib scipy scikit-learn pyyaml tqdm sqlalchemy h5py tables --quiet 2>/dev/null

echo ""
echo "=== Starting Target Model Training ==="
python3 code/train_target_model_pl2_compat.py @args/finetune_hisb_local.txt

echo ""
echo "=== Training Complete ==="
