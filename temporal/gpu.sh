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
response_rate=${3:-0.02}
neurons=${4:-256}
echo $seed $g_ratio $response_rate $neurons

python main.py --seed $seed --g_ratio $g_ratio --jit_duration 0.02 --response_rate $response_rate
exit