#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-1080ti-phd
#SBATCH --gres=gpu:1
#SBATCH --time=00-07:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
response_rate=${2-0.02}
env_name=${3-InvertedPendulum-v2}
jit_duration=${3-0.02}
echo $seed $g_ratio $response_rate $env_name $jit_duration

python normal_perturbation.py --seed $seed --response_rate $response_rate --delayed_env --env_name $env_name --jit_duration $jit_duration
exit