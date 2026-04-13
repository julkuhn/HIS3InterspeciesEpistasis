#!/bin/bash
# Submit ALL_combined jobs for: NPT, METL-Local-45k v1, METL-Local-45k v2, METL-G 3D, METL-G 1D

set -euo pipefail

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
SQSH="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
LOGS="${FOPRA}/metl_rosetta/logs"
CKPT_LOCAL="/workspace/metl_rosetta/source_model_45k/maSKy8WY/checkpoints/epoch=99-step=12600-val_loss=18.58.ckpt"
CKPT_G3D="/workspace/metl-pretrained/models/METL-G-50M-3D-6PSAzdfv.pt"
CKPT_G1D="/workspace/metl-pretrained/models/METL-G-50M-1D-auKdzzwX.pt"
SPLIT_DIR="/workspace/metl_rosetta/metl_splits/ALL_combined"
DS_NAME="his3_ALL_combined"
PDB="/workspace/metl/data/pdb_files/HIS7_YEAST_p.pdb"
mkdir -p "$LOGS"

COMMON_SBATCH="--partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=24:00:00 --no-requeue"
CONTAINER_ARGS="--container-image=${SQSH} --container-mounts=${FOPRA}/:/workspace"

# ── 1. NPT ALL_combined ──────────────────────────────────────────────────────
JID=$(sbatch --parsable \
    --job-name="npt_ALL" \
    --output="${LOGS}/npt_ALL_%j.log" \
    --error="${LOGS}/npt_ALL_%j.err" \
    --mem=128G \
    --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
    --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 --no-requeue \
    ${CONTAINER_ARGS} \
    --wrap="
        export PYTHONPATH=/workspace/.pip_packages
        pip3 install --target=/workspace/.pip_packages torch scipy scikit-learn pandas matplotlib numpy 2>/dev/null
        python3 /workspace/protein_npt.py \
            --segment       ALL_combined \
            --embeddings_dir /workspace/msa_results/ALL_combined \
            --output_dir    /workspace/npt_results/ALL_combined \
            --d_npt         256 \
            --n_heads       8 \
            --n_layers      4 \
            --support_size  256 \
            --query_size    64 \
            --batch_size    8 \
            --epochs        50 \
            --patience      10 \
            --n_support_sets 4
    ")
echo "Submitted NPT ALL_combined: job ${JID}"

# ── 2. METL-Local 45k v1 ALL_combined (extract + linear) ────────────────────
JID=$(sbatch --parsable \
    --job-name="ft45k_v1_ALL" \
    --output="${LOGS}/finetune_local_45k_v1_ALL_%j.log" \
    --error="${LOGS}/finetune_local_45k_v1_ALL_%j.err" \
    $COMMON_SBATCH $CONTAINER_ARGS \
    --wrap="
        set -e
        export PYTHONPATH=/workspace/.pip_packages_pretrain:/workspace/.pip_packages
        export PIP_CACHE_DIR=/workspace/.cache/pip
        export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
        export HF_HOME=/workspace/.cache/huggingface
        mkdir -p \$PIP_CACHE_DIR /workspace/.cache/huggingface
        nvidia-smi 2>&1 | head -3 || true
        if [ ! -f /workspace/.pip_packages_pretrain/torch/version.py ]; then
            pip3 install --target=/workspace/.pip_packages_pretrain torch --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null
            pip3 install --target=/workspace/.pip_packages_pretrain 'pytorch-lightning==1.9.5' 'torchmetrics>=0.7.0,<1.0' 'lightning-utilities>=0.6.0,<0.9.0' 'transformers==4.44.2' 'numpy<2.0' matplotlib scikit-learn scipy pandas --quiet 2>/dev/null
        fi
        python3 -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\"
        cd /workspace/metl
        python3 code/train_target_model_pl2.py \
            --ds_name ${DS_NAME} --split_dir ${SPLIT_DIR} \
            --train_name train --val_name val --test_name test \
            --target_names DMS_score --pdb_fn ${PDB} \
            --model_name transfer_model --pretrained_ckpt_path ${CKPT_LOCAL} \
            --encoding int_seqs --backbone_cutoff -1 \
            --dropout_after_backbone --dropout_after_backbone_rate 0.5 \
            --top_net_type linear \
            --finetuning --finetuning_strategy extract \
            --optimizer adam --weight_decay 0.0 --learning_rate 0.001 \
            --max_epochs 100 --gradient_clip_val 0.5 --batch_size 64 \
            --no_use_wandb \
            --experiment his3_ALL_combined_local_45k \
            --log_dir_base /workspace/metl_rosetta/target_model_45k
    ")
echo "Submitted METL-Local-45k v1 ALL_combined: job ${JID}"

# ── 3. METL-Local 45k v2 ALL_combined (backbone + nonlinear) ────────────────
JID=$(sbatch --parsable \
    --job-name="ft45k_v2_ALL" \
    --output="${LOGS}/finetune_local_45k_v2_ALL_%j.log" \
    --error="${LOGS}/finetune_local_45k_v2_ALL_%j.err" \
    $COMMON_SBATCH $CONTAINER_ARGS \
    --wrap="
        set -e
        export PYTHONPATH=/workspace/.pip_packages_pretrain:/workspace/.pip_packages
        export PIP_CACHE_DIR=/workspace/.cache/pip
        export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
        export HF_HOME=/workspace/.cache/huggingface
        mkdir -p \$PIP_CACHE_DIR /workspace/.cache/huggingface /workspace/metl_rosetta/target_model_45k_v2
        nvidia-smi 2>&1 | head -3 || true
        if [ ! -f /workspace/.pip_packages_pretrain/torch/version.py ]; then
            pip3 install --target=/workspace/.pip_packages_pretrain torch --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null
            pip3 install --target=/workspace/.pip_packages_pretrain 'pytorch-lightning==1.9.5' 'torchmetrics>=0.7.0,<1.0' 'lightning-utilities>=0.6.0,<0.9.0' 'transformers==4.44.2' 'numpy<2.0' matplotlib scikit-learn scipy pandas --quiet 2>/dev/null
        fi
        python3 -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\"
        cd /workspace/metl
        python3 code/train_target_model_pl2.py \
            --ds_name ${DS_NAME} --split_dir ${SPLIT_DIR} \
            --train_name train --val_name val --test_name test \
            --target_names DMS_score --pdb_fn ${PDB} \
            --model_name transfer_model --pretrained_ckpt_path ${CKPT_LOCAL} \
            --encoding int_seqs --backbone_cutoff -1 \
            --dropout_after_backbone --dropout_after_backbone_rate 0.3 \
            --top_net_type nonlinear --top_net_hidden_nodes 256 \
            --top_net_use_dropout --top_net_dropout_rate 0.2 \
            --finetuning --finetuning_strategy backbone \
            --unfreeze_backbone_at_epoch 20 --backbone_initial_ratio_lr 0.1 \
            --optimizer adamw --weight_decay 0.01 --learning_rate 0.001 \
            --lr_scheduler warmup_cosine_decay --warmup_steps 0.05 \
            --max_epochs 200 --gradient_clip_val 0.5 --batch_size 64 \
            --early_stopping --es_monitor val --es_patience 20 \
            --no_use_wandb \
            --experiment his3_ALL_combined_local_45k_v2 \
            --log_dir_base /workspace/metl_rosetta/target_model_45k_v2
    ")
echo "Submitted METL-Local-45k v2 ALL_combined: job ${JID}"

# ── 4+5. METL-G 3D and 1D ALL_combined ──────────────────────────────────────
for MODEL_TYPE in 3D 1D; do
    if [[ "$MODEL_TYPE" == "3D" ]]; then
        CKPT="$CKPT_G3D"
    else
        CKPT="$CKPT_G1D"
    fi
    OUT_DIR="/workspace/metl_rosetta/results/metl_g_${MODEL_TYPE}/ALL_combined"

    JID=$(sbatch --parsable \
        --job-name="metl_${MODEL_TYPE}_ALL" \
        --output="${LOGS}/metl_${MODEL_TYPE}_ALL_%j.log" \
        --error="${LOGS}/metl_${MODEL_TYPE}_ALL_%j.err" \
        --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 \
        --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --no-requeue \
        --wrap="
            set -e
            CNAME=metl_g_${MODEL_TYPE}_ALL_\${SLURM_JOB_ID}
            enroot remove -f \"\$CNAME\" 2>/dev/null || true
            enroot create --name \"\$CNAME\" '${SQSH}'
            enroot start --root \
                --mount '${FOPRA}:/workspace' \
                --mount '${FOPRA}:${FOPRA}' \
                \"\$CNAME\" \
                bash -lc \"
                    set -e
                    pip3 install --quiet --target=/workspace/.pip_packages_metl \
                        'torch==2.4.0' --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
                    pip3 install --quiet --target=/workspace/.pip_packages_metl \
                        'fair-esm==2.0.0' 'transformers>=4.17,<4.47' \
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
                        --experiment     metl_g_${MODEL_TYPE}_ALL_combined \
                        --log_dir_base   ${OUT_DIR}
                \"
            enroot remove -f \"\$CNAME\" 2>/dev/null || true
        ")
    echo "Submitted METL-G ${MODEL_TYPE} ALL_combined: job ${JID}"
done

echo ""
echo "All ALL_combined jobs submitted."
