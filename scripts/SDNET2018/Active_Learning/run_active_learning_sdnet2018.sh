#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:turing:1
#SBATCH -p res-gpu-small
#SBATCH --job-name=AL_sdnet2018
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00

source /etc/profile
source /home3/vzcl68/Code/MedCAL-Bench/MedCAL/bin/activate

echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
nvidia-smi || true

python Active_learning_result_comparison_SDNET2018.py

echo "End time: $(date)"