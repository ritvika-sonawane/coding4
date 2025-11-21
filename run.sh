#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-16:1
#SBATCH --mem=8G
#SBATCH --time=12:00:00


# Change this based on your activating function
source .venv/bin/activate

# Run experiment
python train.py \
    --tag exp \
    --mode monolingual
