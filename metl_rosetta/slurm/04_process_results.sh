#!/bin/bash
#SBATCH --job-name=process_rosetta_S06
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/process_rosetta_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/process_rosetta_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Step 4: Process Rosetta energize results and create SQLite database + splits.
#
# Prerequisites:
#   - Step 3 completed (all energize array jobs finished)
#
# Usage: sbatch 04_process_results.sh

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME="process_rosetta_${SLURM_JOB_ID}"

echo "=== Process Rosetta Results (S06) ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p "${BASE}/metl_rosetta/logs"

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
enroot create --name "$CONTAINER_NAME" "$SQSH"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -lc "
        set -euo pipefail
        export PYTHONPATH=/workspace/.pip_packages:\${PYTHONPATH:-}

        echo 'Installing dependencies...'
        pip3 install scikit-learn --quiet 2>/dev/null

        echo ''
        echo '=== Erstelle SQLite-DB, Splits und Standardisierungsparameter ==='
        python3 /workspace/metl_rosetta/scripts/create_rosetta_db.py \
            --energize_outputs_dir /workspace/metl_rosetta/energize_outputs \
            --db_fn /workspace/metl/data/rosetta_data/his3_s06/his3_s06.db \
            --split_dir /workspace/metl/data/rosetta_data/his3_s06/splits/standard_tr0.8_tu0.1_te0.1 \
            --ct_fn /workspace/metl-sim/variant_database/create_tables.sql \
            --train_size 0.8 \
            --val_size 0.1 \
            --seed 42
    "

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "=== Ergebnis ==="
echo "DB:     ${BASE}/metl/data/rosetta_data/his3_s06/his3_s06.db"
echo "Splits: ${BASE}/metl/data/rosetta_data/his3_s06/splits/standard_tr0.8_tu0.1_te0.1/"
echo "Done: $(date)"
