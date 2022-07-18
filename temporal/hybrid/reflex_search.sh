#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-titanx-phd
#SBATCH --gres=gpu:1
#SBATCH --time=00-10:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
g_ratio=${2:-4}
response_rate=${3:-0.02}
population=${4:-20}
echo $seed $g_ratio $response_rate $population

python reflex_search.py --seed $seed --g_ratio $g_ratio --response_rate $response_rate --population $population
exit

