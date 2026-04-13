#!/bin/bash
# ============================================================
# run_msa_all_chained.sh — Job chain for MSA Transformer ALL_combined
#
# Structure:
#   train_1 → train_2          (254k seqs, ~44h → 2x 24h jobs)
#   val_1                      (63k seqs,  ~11h → 1x 24h job)
#   test_1 → test_2            (166k seqs, ~29h → 2x 24h jobs)
#   mlp_job (after all above)  trains MLP + evaluates
#
# Each embed job auto-resumes from last checkpoint — safe to resubmit.
# ============================================================

set -euo pipefail

FOPRA="${FOPRA:-/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra}"
CONTAINER="${FOPRA}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
PARTS="lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4"
LOGS="${FOPRA}/logs"
mkdir -p "$LOGS"

EMBED_SCRIPT="/workspace/msa_embed_checkpointed.py"
BASELINE_SCRIPT="/workspace/msa_transformer_baseline.py"
OUT_DIR="/workspace/msa_results/ALL_combined"
SPLIT_DIR="/workspace/splits_segmentwise_species"
MSA_FILE="/workspace/data/pgen.1008079.s010.fas"

COMMON_EMBED="
    export PYTHONPATH=/workspace/.pip_packages
    pip3 install --quiet --target=/workspace/.pip_packages \
        fair-esm torch scipy scikit-learn pandas numpy matplotlib 2>/dev/null
    python3 -u ${EMBED_SCRIPT} \
        --segment    ALL_combined \
        --threshold  all \
        --split_dir  ${SPLIT_DIR} \
        --output_dir ${OUT_DIR} \
        --msa_file   ${MSA_FILE} \
        --chunk_size 8000 \
        --batch_size 32 \
        --n_seqs     63 \\
"

submit_embed() {
    local SPLIT=$1
    local DEP=$2   # empty = no dependency
    local DEP_ARG=""
    [[ -n "$DEP" ]] && DEP_ARG="--dependency=afterok:${DEP}"

    sbatch --parsable \
        --job-name="msa_ALL_${SPLIT}" \
        --time=24:00:00 \
        --no-requeue \
        --output="${LOGS}/%j_msa_ALL_${SPLIT}.out" \
        --error="${LOGS}/%j_msa_ALL_${SPLIT}.err" \
        --nodes=1 --cpus-per-task=8 --mem=80G \
        --partition=${PARTS} --gres=gpu:1 \
        --container-image="${CONTAINER}" \
        --container-mounts="${FOPRA}/:/workspace" \
        $DEP_ARG \
        --wrap="${COMMON_EMBED} --split_name ${SPLIT}"
}

echo "=== Submitting embed jobs ==="

# Train: 2 sequential jobs (each resumes from checkpoint)
JID_TRAIN1=$(submit_embed train "")
echo "  train job 1: ${JID_TRAIN1}"

JID_TRAIN2=$(submit_embed train "${JID_TRAIN1}")
echo "  train job 2: ${JID_TRAIN2} (after ${JID_TRAIN1})"

# Val: 1 job (independent)
JID_VAL=$(submit_embed val "")
echo "  val   job:   ${JID_VAL}"

# Test: 2 sequential jobs
JID_TEST1=$(submit_embed test "")
echo "  test  job 1: ${JID_TEST1}"

JID_TEST2=$(submit_embed test "${JID_TEST1}")
echo "  test  job 2: ${JID_TEST2} (after ${JID_TEST1})"

# MLP job: runs after all embeddings are done
# Needs X_train.npy, X_val.npy, X_test.npy to exist (written by embed jobs)
JID_MLP=$(sbatch --parsable \
    --job-name="msa_ALL_mlp" \
    --time=04:00:00 \
    --no-requeue \
    --output="${LOGS}/%j_msa_ALL_mlp.out" \
    --error="${LOGS}/%j_msa_ALL_mlp.err" \
    --nodes=1 --cpus-per-task=8 --mem=80G \
    --partition=${PARTS} --gres=gpu:1 \
    --container-image="${CONTAINER}" \
    --container-mounts="${FOPRA}/:/workspace" \
    --dependency=afterok:${JID_TRAIN2}:${JID_VAL}:${JID_TEST2} \
    --wrap="
        export PYTHONPATH=/workspace/.pip_packages
        pip3 install --quiet --target=/workspace/.pip_packages \
            fair-esm torch scipy scikit-learn pandas numpy matplotlib 2>/dev/null
        python3 -u -c \"
import numpy as np, pandas as pd, pickle, os
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/workspace/msa_results/ALL_combined'
SPLIT_DIR = '/workspace/splits_segmentwise_species'

print('Loading embeddings...')
X_train = np.load(f'{OUT}/X_train.npy')
y_train = np.load(f'{OUT}/y_train.npy')
X_val   = np.load(f'{OUT}/X_val.npy')
y_val   = np.load(f'{OUT}/y_val.npy')
X_test  = np.load(f'{OUT}/X_test.npy')
y_test  = np.load(f'{OUT}/y_test.npy')
print(f'train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_va_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_test)

print('Training MLP...')
mlp = MLPRegressor(hidden_layer_sizes=(512,256,128), activation='relu',
                   solver='adam', max_iter=500, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20,
                   random_state=42, verbose=True)
mlp.fit(X_tr_s, y_train)

def ev(X, y, name):
    pred = mlp.predict(X)
    sp   = spearmanr(y, pred)[0]
    from sklearn.metrics import roc_auc_score
    y_bin = (y > 0.5).astype(int)
    auc = roc_auc_score(y_bin, pred) if 0 < y_bin.sum() < len(y_bin) else float('nan')
    print(f'{name:10s}  Spearman={sp:.4f}  AUC={auc:.4f}')
    return sp, pred, auc

sp_tr, p_tr, auc_tr = ev(X_tr_s, y_train, 'train')
sp_va, p_va, auc_va = ev(X_va_s, y_val,   'val')
sp_te, p_te, auc_te = ev(X_te_s, y_test,  'test')

# save summary
import pandas as pd
pd.DataFrame([{'segment':'ALL_combined','threshold':'all','use_fragment':False,
    'n_train':len(y_train),'n_val':len(y_val),'n_test':len(y_test),
    'spearman_train':sp_tr,'spearman_val':sp_va,'spearman_test_ood':sp_te,
    'auc_train':auc_tr,'auc_val':auc_va,'auc_test_ood':auc_te}]
).to_csv(f'{OUT}/summary.csv', index=False)
print('summary.csv saved')

# save model
with open(f'{OUT}/mlp_model.pkl','wb') as f: pickle.dump(mlp, f)
with open(f'{OUT}/scaler.pkl','wb') as f:    pickle.dump(scaler, f)

# plot
fig, axes = plt.subplots(1,3, figsize=(15,5))
for ax, (ytr, ypr, nm) in zip(axes, [(y_train,p_tr,'Train'),(y_val,p_va,'Val'),(y_test,p_te,'Test OOD')]):
    ax.scatter(ytr, ypr, s=1, alpha=0.2)
    ax.set_xlabel('True DMS'); ax.set_ylabel('Predicted')
    ax.set_title(f'{nm}  ρ={spearmanr(ytr,ypr)[0]:.3f}')
plt.suptitle('MSA Transformer 100M | ALL_combined', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/msa_transformer_ALL_combined_results.png', dpi=120)
print('plot saved')
\"
    ")
echo "  mlp  job:    ${JID_MLP} (after ${JID_TRAIN2}, ${JID_VAL}, ${JID_TEST2})"

echo ""
echo "Chain submitted:"
echo "  train: ${JID_TRAIN1} → ${JID_TRAIN2}"
echo "  val:   ${JID_VAL}"
echo "  test:  ${JID_TEST1} → ${JID_TEST2}"
echo "  mlp:   ${JID_MLP}  (after all of the above)"
echo ""
echo "Expected total wall time: ~4 days (jobs run in parallel where possible)"
