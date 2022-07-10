#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=00-17:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
g_ratio=${2:-4}
response_rate=${3:-0.02}
parent_response_rate=${4:-0.04}
reflex_threshold=${5:-0.15}
reflex_force_scale=${6:-0.5}

echo $seed $g_ratio $response_rate $parent_response_rate $reflex_threshold $reflex_force_scale

python hand_crafted_reflex.py --seed $seed --g_ratio $g_ratio --jit_duration 0.02 --response_rate $response_rate --parent_response_rate $parent_response_rate --save_model --delayed_env --reflex_threshold $reflex_threshold --reflex_force_scale $reflex_force_scale
exit

