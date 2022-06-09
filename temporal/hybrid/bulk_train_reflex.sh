#!/usr/bin/env bash

count=0
for seed in 0
do
    for g_force in 8
    do
        for response_rate in 0.02
        do
            if ! [ -f "reflex_network_TD3_InvertedPendulum-v2_${seed}_0.02_${g_force}.0_${response_rate}_1.0_True" ];
            then
              sbatch train_reflex_gpu.sh $seed $g_force $response_rate
              count=$((count + 1))
            fi
        done
    done
done

echo Launched $count jobs