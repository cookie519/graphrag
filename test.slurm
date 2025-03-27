#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%A_%a
#SBATCH --job-name=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=35G
#SBATCH --gres=gpu:2
#SBATCH --time=0:5:00
#SBATCH --array=0
#SBATCH -p pli-lc
#SBATCH --account=llmkg
echo “Executing on the machine:” $(hostname)