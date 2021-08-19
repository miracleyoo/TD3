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
echo $seed $g_ratio

python main.py --seed $seed --g_ratio $g_ratio --jit
exit