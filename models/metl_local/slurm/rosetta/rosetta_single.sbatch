#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=rosetta_s05
#SBATCH --output=logs/rosetta_s05_%j.out
#SBATCH --error=logs/rosetta_s05_%j.err

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
cd "$BASE"
mkdir -p logs

SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME=fopra_container
enroot create --name "$CONTAINER_NAME" "$SQSH" 2>/dev/null || true

CHUNK_FILE="$1"
echo "Processing ${CHUNK_FILE} on $(hostname)"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -lc "
        export PYTHONPATH=/workspace/.pip_packages
        ${BASE}/rosetta/source/bin/relax.static.linuxgccrelease --help >/dev/null
        cd /workspace/metl-sim
        python3 code/energize.py \
            --rosetta_main_dir=/workspace/rosetta \
            --pdb_dir=/workspace/metl-sim/output/prepare_outputs/HIS7_YEAST \
            --variants_fn=${CHUNK_FILE} \
            --log_dir_base=output/energize_outputs/hisb_s05_full \
            --cluster local \
            --process local
    "
