#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:turing:1
#SBATCH -p res-gpu-small
#SBATCH --job-name=DynRAL_sdnet2018_patched
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00

source /etc/profile
source /home3/vzcl68/Code/MedCAL-Bench/MedCAL/bin/activate

cd /home3/vzcl68/Code/Active_Learning_Benchmarking_Classification/scripts/SDNET2018/Dynamic_Reinforcement_Active_Learning_Patched

echo "Running on node: $HOSTNAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "CUDA:"
nvidia-smi || true

python Dynamic_RL_Active_learning_result_patched_SDNET2018.py

echo "End time: $(date)"
