#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00:00
#SBATCH --mem=1600000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=24

seed=${1:-0}
g_ratio=${2:-4}
response_rate=${3:-0.02}
population=${4:-20}
parent_response_rate=${5:-0.04}
jit_duration=${6:-0.02}
env_name=${6:-InvertedPendulum-v2}

echo $seed $g_ratio $response_rate $population $parent_response_rate $jit_duration $env_name

python reflex_search.py --seed $seed --g_ratio $g_ratio --response_rate $response_rate --population $population --parent_response_rate $parent_response_rate --jit_duration $jit_duration --env_name $env_name
exit

