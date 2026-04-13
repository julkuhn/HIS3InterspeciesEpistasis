#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=prepare_pdb
#SBATCH --output=logs/prepare_pdb_%j.out
#SBATCH --error=logs/prepare_pdb_%j.err
#SBATCH --no-requeue

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
cd "$BASE"
mkdir -p logs .pip_packages .cache/pip metl_rosetta/logs

SQSH="${BASE}/nvidia+ai-workbench+python-cuda122+1.0.6.sqsh"
CONTAINER_NAME=fopra_container
enroot create --name "$CONTAINER_NAME" "$SQSH" 2>/dev/null || true

echo "=== Prepare PDB for Rosetta ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

enroot start --root \
  --mount "${BASE}:/workspace" \
  --mount "${BASE}:${BASE}" \
  "$CONTAINER_NAME" \
  bash -lc '
    set -euo pipefail

    export PYTHONPATH=/workspace/.pip_packages
    export PIP_CACHE_DIR=/workspace/.cache/pip
    mkdir -p "$PIP_CACHE_DIR"

    # Install deps into mounted target (persists across jobs)
    python3 -m pip install --upgrade --target=/workspace/.pip_packages \
      --root-user-action=ignore \
      biopython >/dev/null

    # If conda is not available, provide a minimal shim for "conda run ..."
    if ! command -v conda >/dev/null 2>&1; then
      mkdir -p /workspace/.local/bin
      cat > /workspace/.local/bin/conda <<'"'"'EOF'"'"'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "run" ]]; then
  echo "conda shim: only supports '\''conda run ...'\''" >&2
  exit 2
fi
shift

# drop common options: -n/--name ENV, -p/--prefix PATH, --no-capture-output, etc.
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -n|--name) shift 2 ;;
    -p|--prefix) shift 2 ;;
    --no-capture-output) shift 1 ;;
    *) shift 1 ;;
  esac
done

# Remaining args are the real command
exec "$@"
EOF
      chmod +x /workspace/.local/bin/conda
    fi
    export PATH=/workspace/.local/bin:$PATH

    # If Rosetta runtime libs are needed
    export LD_LIBRARY_PATH=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/source/build/src/release/linux/5.15/64/x86/gcc/11/default:/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/source/build/external/release/linux/5.15/64/x86/gcc/11/default:${LD_LIBRARY_PATH:-}

    cd /workspace/metl-sim

    python3 code/prepare.py \
      --rosetta_main_dir=/workspace/rosetta \
      --pdb_fn=/workspace/data/HIS7_YEAST.pdb \
      --relax_nstruct=10 \
      --out_dir_base=/workspace/metl_rosetta/pdb_prepared
  '

echo "Done: $(date)"
echo "Prepared PDB should be in: ${BASE}/metl_rosetta/pdb_prepared/"
echo "Look for the *_p.pdb file (lowest energy relaxed structure)."

