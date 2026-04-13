#!/bin/bash
#SBATCH --job-name=rosetta_test
#SBATCH --time=20:00:00
#SBATCH --output=rosetta_test_%j.out
#SBATCH --error=rosetta_test_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu

set -euo pipefail

BASE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra
cd "$BASE"

enroot start --root \
    --mount "${BASE}:/workspace" \
    --mount "${BASE}:${BASE}" \
    fopra_container \
    bash -lc "
        set -euo pipefail

        # Optional: schneller Fail-Fast Check, ob Rosetta im Container starten kann
        ${BASE}/rosetta/source/bin/relax.static.linuxgccrelease --help >/dev/null

        cd /workspace/metl-sim
        python3 code/energize.py \
            --rosetta_main_dir=/workspace/rosetta \
            --pdb_dir=/workspace/metl-sim/output/prepare_outputs/HIS7_YEAST \
            --variants_fn=variant_lists/hisb_s05_test100_formatted.txt \
            --log_dir_base=output/energize_outputs/hisb_s05_test
    "

echo "Test finished!"
