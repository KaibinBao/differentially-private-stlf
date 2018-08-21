#!/bin/bash
#SBATCH -A kaba
#SBATCH --job-name=gen_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-10

SEEDS=(\
	"3569688430" \
	"1144090884" \
	"3975072639" \
	"3380223895" \
	"3284302308" \
	"685220624" \
	"1078070129" \
	"1384693566" \
	"3058439984" \
	"3839378342"\
	)

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
srun python generate_noised_data.py --seed 42 0
else
SEED=${SEEDS[(($SLURM_ARRAY_TASK_ID - 1))]}
srun python generate_noised_data.py --seed $SEED 1778 17783 177828 5623 56234 562341 1000 10000 100000 1000000 3162 31623 316228
fi
