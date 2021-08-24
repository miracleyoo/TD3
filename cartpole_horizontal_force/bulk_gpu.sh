#!/usr/bin/env bash

for seed in {0..4}
do
    for g_force in {0..10}
    do
        for env_timestep in 0.02
        do
            sbatch gpu.sh $seed $g_force $env_timestep
        done
    done
done