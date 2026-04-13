#!/bin/bash
#SBATCH --job-name=local_finetune_metl_s05
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME="metl_local_${SLURM_JOB_ID}"

echo "=== Finetune METL Local Target Model ==="
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
        nvidia-smi

        cd /workspace/metl

        echo '=== Pretrained Local Finetune ==='
        python3 code/train_target_model_pl2.py \
            @/workspace/metl_rosetta/args/finetune_his3_S06_local.txt
    "

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
echo "Done: $(date)"