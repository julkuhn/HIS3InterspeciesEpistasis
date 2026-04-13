#!/bin/bash
# ============================================================
# run_metl_global.sh — Submits METL-Global (3D + 1D) finetune jobs
#                      for all segments from best_per_segment.csv
#
# Usage: bash run_metl_global.sh
# ============================================================

set -euo pipefail

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
SQSH="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
BEST_CSV="${FOPRA}/splits_segmentwise_species/best_per_segment.csv"
PDB="/workspace/metl/data/pdb_files/HIS7_YEAST_p.pdb"
CKPT_3D="/workspace/metl-pretrained/models/METL-G-50M-3D-6PSAzdfv.pt"
CKPT_1D="/workspace/metl-pretrained/models/METL-G-50M-1D-auKdzzwX.pt"

mkdir -p "${FOPRA}/metl_rosetta/logs" "${FOPRA}/metl_rosetta/results"

mapfile -t SEGMENTS < <(awk -F',' 'NR>1 {print $1","$2}' "${BEST_CSV}")

for entry in "${SEGMENTS[@]}"; do
    SEG=$(echo "$entry" | cut -d, -f1)
    THR=$(echo "$entry" | cut -d, -f2)
    DS_NAME="his3_${SEG}_thr${THR}"
    SPLIT_DIR="/workspace/metl_rosetta/metl_splits/${SEG}_thr${THR}"

    for MODEL_TYPE in 3D 1D; do
        if [[ "$MODEL_TYPE" == "3D" ]]; then
            CKPT="$CKPT_3D"
        else
            CKPT="$CKPT_1D"
        fi

        CONTAINER_NAME="metl_g_${MODEL_TYPE}_${SEG}_\${SLURM_JOB_ID}"
        OUT_DIR="/workspace/metl_rosetta/results/metl_g_${MODEL_TYPE}/${SEG}"

        JID=$(sbatch --parsable \
            --job-name="metl_${MODEL_TYPE}_${SEG}" \
            --output="${FOPRA}/metl_rosetta/logs/metl_${MODEL_TYPE}_${SEG}_%j.log" \
            --error="${FOPRA}/metl_rosetta/logs/metl_${MODEL_TYPE}_${SEG}_%j.err" \
            --time=24:00:00 \
            --no-requeue \
            --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4 \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=32G \
            --wrap="
                set -e
                CONTAINER_NAME=metl_g_${MODEL_TYPE}_${SEG}_\${SLURM_JOB_ID}
                enroot remove -f \"\$CONTAINER_NAME\" 2>/dev/null || true
                enroot create --name \"\$CONTAINER_NAME\" '${SQSH}'
                enroot start --root \
                    --mount '${FOPRA}:/workspace' \
                    --mount '${FOPRA}:${FOPRA}' \
                    \"\$CONTAINER_NAME\" \
                    bash -lc \"
                        set -e
                        pip3 install --quiet --target=/workspace/.pip_packages_metl \
                            'torch==2.4.0' \
                            --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
                        pip3 install --quiet --target=/workspace/.pip_packages_metl \
                            'fair-esm==2.0.0' \
                            'transformers>=4.17,<4.47' \
                            scipy scikit-learn pandas numpy matplotlib \
                            pytorch-lightning wandb biopandas biopython \
                            tqdm seaborn shortuuid pyyaml sqlalchemy \
                            networkx connectorx tensorboard torchinfo \
                            torchmetrics rich 2>/dev/null
                        export PYTHONPATH=/workspace/.pip_packages_metl:\\\$PYTHONPATH
                        cd /workspace/metl
                        mkdir -p ${OUT_DIR}
                        python3 code/train_target_model_pl2.py \
                            --ds_name        ${DS_NAME} \
                            --target_names   DMS_score \
                            --model_name     transfer_model \
                            --pretrained_ckpt_path ${CKPT} \
                            --pdb_fn         ${PDB} \
                            --backbone_cutoff -2 \
                            --batch_size     64 \
                            --max_epochs     200 \
                            --split_dir      ${SPLIT_DIR} \
                            --train_name     train \
                            --val_name       val \
                            --test_name      test \
                            --encoding       int_seqs \
                            --gradient_clip_val 0.5 \
                            --no_use_wandb \
                            --experiment     metl_g_${MODEL_TYPE}_${SEG} \
                            --log_dir_base   ${OUT_DIR}
                    \"
                enroot remove -f \"\$CONTAINER_NAME\" 2>/dev/null || true
            ")
        echo "Submitted metl_g_${MODEL_TYPE} ${SEG} (thr=${THR}): job ${JID}"
    done
done

echo ""
echo "All METL-Global jobs submitted."
