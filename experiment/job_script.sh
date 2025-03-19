#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=5120
#SBATCH --time=01:00:00   # walltime
#SBATCH --error=err.txt

module load gcc/8.2.0 python_gpu

echo "Starting run at: `date`"

python train_model.py

echo "Program finished with exit code $? at: `date`"
