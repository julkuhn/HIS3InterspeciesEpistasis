#!/bin/bash
# ============================================================
# run_esm_deepen.sh — Submits SLURM GPU job für ESM-2 auf
#                     dem DeePEn HuggingFace-Datensatz
#
# Usage:  bash run_esm_deepen.sh
# ============================================================

set -euo pipefail

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
RESULTS_DIR="${FOPRA}/esm_results/deepen"
CONTAINER="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"

mkdir -p "${RESULTS_DIR}" "${FOPRA}/logs"

JID=$(sbatch --parsable \
    --job-name="esm_deepen" \
    --time=20:00:00 \
    --no-requeue \
    --output="${FOPRA}/logs/%j_%x.out" \
    --error="${FOPRA}/logs/%j_%x.err" \
    --nodes=1 \
    --cpus-per-task=8 \
    --mem=80G \
    --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
    --gres=gpu:1 \
    --container-image="${CONTAINER}" \
    --container-mounts="${FOPRA}/:/workspace" \
    --wrap="
        export PYTHONPATH=/workspace/.pip_packages
        pip3 install --target=/workspace/.pip_packages \
            fair-esm torch transformers matplotlib scikit-learn scipy pandas numpy datasets \
            2>/dev/null
        python3 /workspace/esm_deepen.py \
            --output_dir /workspace/esm_results/deepen \
            --batch_size 8 \
            --cache_dir  /workspace/.hf_cache
    ")

echo "Submitted esm_deepen: job ${JID}"
echo "Ergebnisse: ${RESULTS_DIR}/"
echo "Log:        ${FOPRA}/logs/${JID}_esm_deepen.out"
echo ""
echo "Status verfolgen:"
echo "  squeue -j ${JID}"
echo "  tail -f ${FOPRA}/logs/${JID}_esm_deepen.out"
