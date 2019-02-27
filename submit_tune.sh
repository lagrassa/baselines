#!/bin/sh
  
#SBATCH -o tune_fp.out-%j
#SBATCH -a 1-1
#SBATCH -c 12
#SBATCH --constraint=opteron


# run with: sbatch jobArray.sh
# or run with: LLsub jobArray.sh

# Initialize Modules
source /etc/profile
module load cuda-9.0
export CUDA_VISIBLE_DEVICES=""


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

python tune.py naf_fetchpush
