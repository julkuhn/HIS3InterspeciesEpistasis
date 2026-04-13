#!/bin/bash
#SBATCH --job-name=his3_split
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu

FOPRA="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
PKGS="${FOPRA}/.pip_packages"
mkdir -p "${FOPRA}/logs" "$PKGS"

export PYTHONPATH="$PKGS"
export PIP_CACHE_DIR="${FOPRA}/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

echo "Node: $(hostname)"
echo "Python: $(python3 --version)"

pip3 install --target="$PKGS" biopython scipy pandas numpy matplotlib 2>/dev/null

python3 "${FOPRA}/run_splitting.py" \
    --csv_path   "${FOPRA}/data/HIS7_YEAST_Pokusaeva_2019_with_segments.csv" \
    --fasta_path "${FOPRA}/data/pgen.1008079.s010.fas" \
    --out_dir    "${FOPRA}/splits_segmentwise_species" \
    --plot
