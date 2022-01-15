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
reflex_threshold=${4:-0.15}
echo $seed $g_ratio $response_rate $reflex_threshold

python train_reflex.py --seed $seed --g_ratio $g_ratio --jit_duration 0.02 --response_rate $response_rate --reflex_response_rate 0.02 --max_timesteps 400000 --save_model --eval_freq 10000 --reflex_threshold $reflex_threshold
exit