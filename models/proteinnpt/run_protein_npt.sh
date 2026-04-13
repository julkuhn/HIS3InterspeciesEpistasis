#!/bin/bash
# ============================================================
# run_protein_npt.sh — Submits one SLURM GPU job per segment for ProteinNPT.
#
# Prerequisite: MSA Transformer embeddings must already exist in
#               /workspace/msa_results/<SEGMENT>/X_*.npy
#
# Usage:  bash run_protein_npt.sh [optional: single segment, e.g. S08]
# ============================================================

set -euo pipefail

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
MSA_RESULTS="${FOPRA}/msa_results"
NPT_RESULTS="${FOPRA}/npt_results"
CONTAINER="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"

mkdir -p "${NPT_RESULTS}" "${FOPRA}/logs"

# If a segment is passed as argument, run only that one; otherwise run all.
if [[ $# -ge 1 ]]; then
    SEGMENTS=("$1")
else
    # Discover segments from existing MSA embedding directories
    SEGMENTS=()
    for d in "${MSA_RESULTS}"/*/; do
        seg=$(basename "$d")
        if [[ -f "${d}/X_train.npy" ]]; then
            SEGMENTS+=("$seg")
        fi
    done
fi

echo "Segments to run: ${SEGMENTS[*]}"

for SEG in "${SEGMENTS[@]}"; do
    EMB_DIR="${MSA_RESULTS}/${SEG}"
    OUT_DIR="${NPT_RESULTS}/${SEG}"
    mkdir -p "$OUT_DIR"

    sbatch \
        --job-name="npt_${SEG}" \
        --time=06:00:00 \
        --no-requeue \
        --output="${FOPRA}/logs/%j_%x.out" \
        --error="${FOPRA}/logs/%j_%x.err" \
        --nodes=1 \
        --cpus-per-task=8 \
        --mem=60G \
        --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
        --gres=gpu:1 \
        --container-image="${CONTAINER}" \
        --container-mounts="${FOPRA}/:/workspace" \
        --wrap="
            export PYTHONPATH=/workspace/.pip_packages
            pip3 install --target=/workspace/.pip_packages \
                torch scipy scikit-learn pandas matplotlib numpy \
                2>/dev/null
            python3 /workspace/protein_npt.py \
                --segment       ${SEG} \
                --embeddings_dir /workspace/msa_results/${SEG} \
                --output_dir    /workspace/npt_results/${SEG} \
                --d_npt         256 \
                --n_heads       8 \
                --n_layers      4 \
                --support_size  256 \
                --query_size    64 \
                --batch_size    8 \
                --epochs        50 \
                --patience      10 \
                --n_support_sets 4
        "
    echo "  Submitted npt_${SEG}"
done

echo ""
echo "Alle NPT-Jobs submitted."
echo "Ergebnisse: ${NPT_RESULTS}/"
