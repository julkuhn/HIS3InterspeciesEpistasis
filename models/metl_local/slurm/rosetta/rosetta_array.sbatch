#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=rosetta_s05
#SBATCH --output=logs/rosetta_s05_%A_%a.out
#SBATCH --error=logs/rosetta_s05_%A_%a.err

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
cd "$BASE"
mkdir -p logs

SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME=fopra_container
enroot create --name "$CONTAINER_NAME" "$SQSH" 2>/dev/null || true

CHUNK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CHUNK_FILE="variant_lists/chunks_s05/chunk_${CHUNK_ID}"

echo "Processing chunk ${CHUNK_ID} with $(wc -l < metl-sim/${CHUNK_FILE}) variants"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -c '
        export LD_LIBRARY_PATH=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/source/build/src/release/linux/5.15/64/x86/gcc/11/default:/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/source/build/external/release/linux/5.15/64/x86/gcc/11/default
        export PYTHONPATH=/workspace/.pip_packages
        cd /workspace/metl-sim
        python3 code/energize.py \
            --rosetta_main_dir=/workspace/rosetta \
            --pdb_dir=/workspace/metl-sim/output/prepare_outputs/HIS7_YEAST \
            --variants_fn='"${CHUNK_FILE}"' \
            --log_dir_base=output/energize_outputs/hisb_s05_full \
            --cluster local \
            --process local
    '
