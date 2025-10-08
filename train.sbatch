#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bezc-delta-gpu
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=4


source /u/wchen11/anaconda3/bin/activate 
conda activate verl
cd /u/wchen11/verl

./examples/sglang_multiturn/run_qwen2.5-0.5b_gsm8k_multiturn_w_interaction.sh