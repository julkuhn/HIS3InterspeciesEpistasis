#!/bin/bash
#SBATCH --job-name=energize_S06
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/energize_%A_%a.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/energize_%A_%a.err
#SBATCH --time=23:40:00
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=501-510%9

set -eo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

CHUNK_DIR="${BASE}/metl_rosetta/variant_lists/chunks"
CHUNK_FN="${CHUNK_DIR}/chunk_$(printf '%04d' ${TASK_ID}).txt"

PDB_DIR="${BASE}/metl_rosetta/pdb_prepared/HIS7_YEAST_2026-02-14_15-47-02"

OUT_DIR="${BASE}/metl_rosetta/energize_outputs/chunk_$(printf '%04d' ${TASK_ID})"

if [ ! -f "${CHUNK_FN}" ]; then
    echo "Chunk file not found: ${CHUNK_FN} -- skipping"
    exit 0
fi

N_VARIANTS=$(wc -l < "${CHUNK_FN}")
echo "=== Energize: Task ${TASK_ID} ==="
echo "Chunk: ${CHUNK_FN} (${N_VARIANTS} variants)"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd "${BASE}"

SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME=fopra_container

# Eindeutigen Container-Name pro Task verwenden
CONTAINER_NAME="fopra_${SLURM_JOB_ID}_${TASK_ID}"

# Alten Container aufräumen falls vorhanden
enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
enroot create --name "$CONTAINER_NAME" "$SQSH"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -lc "
        set -eo pipefail
        export PYTHONPATH=/workspace/.pip_packages
        cd /workspace/metl-sim
        python3 code/energize.py \
            --rosetta_main_dir=/workspace/rosetta \
            --pdb_dir=${PDB_DIR} \
            --variants_fn=${CHUNK_FN} \
            --log_dir_base=${OUT_DIR} \
            --cluster local \
            --process local
    "

# Aufräumen nach Erfolg
enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Done: $(date)"
