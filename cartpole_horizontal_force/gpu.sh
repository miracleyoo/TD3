#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=00-04:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
g_ratio=${2:-4}
env_timestep=${3:-0.02}
echo $seed $g_ratio $env_timestep

python main.py --seed $seed --g_ratio $g_ratio --jit --env_timestep $env_timestep
exit