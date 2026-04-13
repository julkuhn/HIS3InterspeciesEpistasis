#!/bin/bash
#SBATCH --job-name=rebuild_rosetta_db_45k
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/rebuild_db_45k_%j.log
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl_rosetta/logs/rebuild_db_45k_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME="rebuild_db_45k_${SLURM_JOB_ID}"

DB_DIR="${BASE}/metl/data/rosetta_data/his3_s06_45k"
DB_FN="${DB_DIR}/his3_s06_45k.db"
SPLIT_DIR="${DB_DIR}/splits/standard_tr0.8_tu0.1_te0.1"

echo "=== Rebuild Rosetta DB (45k) ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

mkdir -p "${DB_DIR}" "${SPLIT_DIR}"

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true
enroot create --name "$CONTAINER_NAME" "$SQSH"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    "$CONTAINER_NAME" \
    bash -lc "
        set -euo pipefail
        export PYTHONPATH=/workspace/.pip_packages:\${PYTHONPATH:-}
        pip3 install scikit-learn --quiet 2>/dev/null

        mkdir -p /workspace/metl/data/rosetta_data/his3_s06_45k/splits/standard_tr0.8_tu0.1_te0.1

        python3 /workspace/metl_rosetta/scripts/create_rosetta_db.py \
            --energize_outputs_dir /workspace/metl_rosetta/energize_outputs \
            --db_fn /workspace/metl/data/rosetta_data/his3_s06_45k/his3_s06_45k.db \
            --split_dir /workspace/metl/data/rosetta_data/his3_s06_45k/splits/standard_tr0.8_tu0.1_te0.1 \
            --ct_fn /workspace/metl-sim/variant_database/create_tables.sql \
            --train_size 0.8 \
            --val_size 0.1 \
            --seed 42
    "

enroot remove -f "$CONTAINER_NAME" 2>/dev/null || true

# Count variants in new DB
PYTHONPATH="${BASE}/.pip_packages" python3 -c "
import sqlite3
conn = sqlite3.connect('${DB_FN}')
n = conn.execute('SELECT COUNT(*) FROM variant').fetchone()[0]
print(f'Variants in new DB: {n}')
" 2>/dev/null || true

echo "Done: $(date)"
echo "DB: ${DB_FN}"
