#!/bin/bash
#SBATCH --job-name=everything
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=J
#SBATCH --time=4:00:00

module load conda
conda activate 541
python -m new.run_experiments