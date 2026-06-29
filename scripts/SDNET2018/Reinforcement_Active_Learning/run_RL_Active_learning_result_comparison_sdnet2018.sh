#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH -p res-gpu-small
#SBATCH --job-name=RAL_sdnet2018
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00

source /etc/profile
source /home3/vzcl68/Code/MedCAL-Bench/MedCAL/bin/activate

cd /home3/vzcl68/Code/Active_Learning_Benchmarking_Classification/scripts/SDNET2018/Reinforcement_Active_Learning

echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "CUDA:"
nvidia-smi || true

python RL_Active_learning_result_comparison_SDNET2018.py

echo "End time: $(date)"