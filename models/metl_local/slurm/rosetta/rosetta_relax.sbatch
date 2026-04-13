#!/bin/bash
#SBATCH --job-name rosetta_relax
#SBATCH --time=01:00:00
#SBATCH --output rosetta_relax_%j.out
#SBATCH --error rosetta_relax_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/metl-sim

/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/source/bin/relax.default.linuxgccrelease \
    -s output/prepare_outputs/HIS7_YEAST/working_dir/HIS7_YEAST_A.pdb \
    -relax:constrain_relax_to_start_coords \
    -relax:coord_constrain_sidechains \
    -relax:ramp_constraints false \
    -ex1 -ex2 \
    -use_input_sc \
    -flip_HNQ \
    -no_optH false \
    -nstruct 10 \
    -out:path:pdb output/prepare_outputs/HIS7_YEAST/working_dir/ \
    -database /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/rosetta/database

echo "Relax finished!"
