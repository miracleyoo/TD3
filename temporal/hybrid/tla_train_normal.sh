#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00:00
#SBATCH --mem=160000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=24

seed=${1:-0}
response_rate=${2:-0.02}
parent_response_rate=${3:-0.04}
env_name=${4:-InvertedPendulum-v2}

echo $seed $response_rate $parent_response_rate $env_name

python tla_train_normal.py --seed $seed --response_rate $response_rate --parent_response_rate $parent_response_rate --env_name $env_name --double_action --with_parent_action --penalty
exit

