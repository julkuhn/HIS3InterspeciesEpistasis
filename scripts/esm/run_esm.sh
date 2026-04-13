#!/bin/bash
# ============================================================
# run_esm.sh — Submits one SLURM GPU job per segment (+ ALL combined)
#              for ESM-2 650M + MLP, then plots results.
#
# Usage:  bash run_esm.sh
# ============================================================

set -euo pipefail

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
SPLIT_DIR="${FOPRA}/splits_segmentwise_species"
RESULTS_DIR="${FOPRA}/esm_results"
BEST_CSV="${SPLIT_DIR}/best_per_segment.csv"
CONTAINER="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"

mkdir -p "${RESULTS_DIR}" "${FOPRA}/logs"

# ── Fragment positions per segment (0-indexed, exclusive end) ─────────────────
declare -A FRAG_START FRAG_END
FRAG_START[S02]=135; FRAG_END[S02]=163
FRAG_START[S03]=144; FRAG_END[S03]=170 
FRAG_START[S04]=170; FRAG_END[S04]=200
FRAG_START[S05]=180; FRAG_END[S05]=211
FRAG_START[S06]=95;  FRAG_END[S06]=125
FRAG_START[S07]=65;  FRAG_END[S07]=95
FRAG_START[S08]=55;  FRAG_END[S08]=85
FRAG_START[S12]=5;   FRAG_END[S12]=24

# ── Read segments + thresholds from best_per_segment.csv (pure awk, no python) ─
mapfile -t SEGMENTS < <(awk -F',' 'NR>1 {print $1","$2}' "${BEST_CSV}")

JOB_IDS=()

# ── Submit one job per segment ────────────────────────────────────────────────
for entry in "${SEGMENTS[@]}"; do
    SEG=$(echo "$entry" | cut -d, -f1)
    THR=$(echo "$entry" | cut -d, -f2)
    OUT_DIR="${RESULTS_DIR}/${SEG}"
    mkdir -p "$OUT_DIR"

    # Build fragment args
    if [[ -v FRAG_START[$SEG] ]]; then
        FRAG_ARGS="--use_fragment --frag_start ${FRAG_START[$SEG]} --frag_end ${FRAG_END[$SEG]}"
    else
        FRAG_ARGS=""
    fi

    JID=$(sbatch --parsable \
        --job-name="esm_${SEG}" \
        --time=06:00:00 \
        --no-requeue \
        --output="${FOPRA}/logs/%j_%x.out" \
        --error="${FOPRA}/logs/%j_%x.err" \
        --nodes=1 \
        --cpus-per-task=6 \
        --mem=40G \
        --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
        --gres=gpu:1 \
        --container-image="${CONTAINER}" \
        --container-mounts="${FOPRA}/:/workspace" \
        --wrap="
            export PYTHONPATH=/workspace/.pip_packages
            pip3 install --target=/workspace/.pip_packages \
                fair-esm torch transformers matplotlib scikit-learn scipy pandas numpy \
                2>/dev/null
            python3 /workspace/esm_baseline.py \
                --segment      ${SEG} \
                --threshold    ${THR} \
                --split_dir    /workspace/splits_segmentwise_species \
                --output_dir   /workspace/esm_results/${SEG} \
                ${FRAG_ARGS}
        ")
    echo "Submitted ${SEG} (thr=${THR}): job ${JID}"
    JOB_IDS+=("$JID")
done

# ── ALL combined (full sequence, no fragment) ─────────────────────────────────
ALL_OUT="${RESULTS_DIR}/ALL_combined"
mkdir -p "$ALL_OUT"

JID_ALL=$(sbatch --parsable \
    --job-name="esm_ALL" \
    --time=08:00:00 \
    --no-requeue \
    --output="${FOPRA}/logs/%j_%x.out" \
    --error="${FOPRA}/logs/%j_%x.err" \
    --nodes=1 \
    --cpus-per-task=6 \
    --mem=60G \
    --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
    --gres=gpu:1 \
    --container-image="${CONTAINER}" \
    --container-mounts="${FOPRA}/:/workspace" \
    --wrap="
        export PYTHONPATH=/workspace/.pip_packages
        pip3 install --target=/workspace/.pip_packages \
            fair-esm torch transformers matplotlib scikit-learn scipy pandas numpy \
            2>/dev/null
        python3 /workspace/esm_baseline.py \
            --segment   ALL_combined \
            --threshold all \
            --split_dir /workspace/splits_segmentwise_species \
            --output_dir /workspace/esm_results/ALL_combined
    ")
echo "Submitted ALL_combined: job ${JID_ALL}"
JOB_IDS+=("$JID_ALL")

# ── Wait for all jobs, then plot ──────────────────────────────────────────────
DEP=$(IFS=:; echo "${JOB_IDS[*]}")

sbatch --parsable \
    --job-name="esm_plot" \
    --time=00:30:00 \
    --no-requeue \
    --output="${FOPRA}/logs/%j_%x.out" \
    --error="${FOPRA}/logs/%j_%x.err" \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --partition=lrz-cpu \
    --qos=cpu \
    --dependency="afterok:${DEP}" \
    --wrap="
        export PYTHONPATH=${FOPRA}/.pip_packages
        pip3 install --target=${FOPRA}/.pip_packages matplotlib pandas numpy 2>/dev/null
        python3 ${FOPRA}/plot_results.py \
            --model       esm \
            --results_dir ${RESULTS_DIR} \
            --out         ${RESULTS_DIR}/esm_results_plot.png
    "

echo ""
echo "Alle Jobs submitted. Plot wird nach Abschluss aller Jobs erzeugt."
echo "Ergebnisse: ${RESULTS_DIR}/"
echo "Plot:       ${RESULTS_DIR}/esm_results_plot.png"
