#!/bin/bash
#SBATCH --job-name rosetta_energize
#SBATCH --time=24:00:00
#SBATCH --output logs/rosetta_%A_%a.out
#SBATCH --error logs/rosetta_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu

cd /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl-sim

PART=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python3 code/energize.py \
    --rosetta_main_dir=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta \
    --pdb_fn=output/prepare_outputs/HIS7_YEAST/HIS7_YEAST_p.pdb \
    --variants_fn=variant_lists/hisb_s05_part_${PART}.txt \
    --out_dir=output/energize_outputs/hisb_s05_part_${PART}

echo "Part ${PART} finished!"
