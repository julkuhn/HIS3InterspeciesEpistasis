#!/bin/bash
#SBATCH --job-name=pairformer_mlp
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pairformer_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/pairformer_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -eo pipefail

BASE="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME="pairformer_${SLURM_JOB_ID}"

echo "=== Pairformer MLP VEP ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
enroot create --name "$CONTAINER_NAME" "$SQSH"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -lc "
        set -eo pipefail
        export PYTHONPATH=/workspace/.pip_packages:\$PYTHONPATH
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        nvidia-smi

        python3 /workspace/metl_rosetta/scripts/pairformer_mlp.py \
            --msa /workspace/metl_rosetta/metl_data/HIS7_YEAST_full_11-26-2021_b09.a2m \
            --dms /workspace/metl_rosetta/metl_data/his3_S08.tsv \
            --split_dir /workspace/metl_rosetta/metl_data/his3_S08_split \
            --weights_dir /workspace/pairformer_weights \
            --output_dir /workspace/metl_rosetta/pairformer_results/S08 \
            --hidden_dim 256 \
            --n_layers 2 \
            --dropout 0.2 \
            --lr 0.001 \
            --epochs 100 \
            --patience 15 \
            --batch_size 256
    "

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
echo "Done: $(date)"