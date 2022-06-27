#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=00-03:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8


g_ratio=${1:-4}
response_rate=${2:-0.02}
echo $g_ratio $response_rate

python eval_reflex_after_training.py --g_force $g_ratio --response_rate $response_rate
exit

