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
response_rate=${2-0.02}
env_name=${3-InvertedPendulum-v2}
echo $seed $g_ratio $response_rate $env_name

python normal.py --seed $seed --response_rate $response_rate --delayed_env --save_model -env_name $env_name
exit