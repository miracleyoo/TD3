#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
g_ratio=${2:-4}
response_rate=${3:-0.02}
echo $seed $g_ratio $response_rate

python train_reflex.py --seed $seed --g_ratio $g_ratio --jit_duration 0.02 --response_rate $response_rate --save_model --delayed_env
exit

