#!/usr/bin/env bash

for seed in {0..4}
do
    for g_force in {0..10}
    do
        sbatch gpu.sh $seed $g_force
    done
done