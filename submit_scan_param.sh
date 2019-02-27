#!/bin/sh
  
#SBATCH -o tunelogs/tune_al25.out-%j
#SBATCH -a 1-3
#SBATCH -c 8
#SBATCH --constraint=opteron


# run with: sbatch jobArray.sh
# or run with: LLsub jobArray.sh

# Initialize Modules
source /etc/profile
module load cuda-9.0
export CUDA_VISIBLE_DEVICES=""


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
if [ $SLURM_ARRAY_TASK_ID == 1 ]
then
python scan_params.py naf
fi

if [ $SLURM_ARRAY_TASK_ID == 2 ]
then
python scan_params.py ddpg
fi
if [ $SLURM_ARRAY_TASK_ID == 3 ]
then
python scan_params.py ppo2
fi
~      

