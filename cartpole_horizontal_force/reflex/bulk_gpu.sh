#!/usr/bin/env bash

for seed in {0..4}
do
    for g_force in {0..10}
    do
        for response_rate in 0.04
        do
            for reflex_threhold in 0.15
            do
                sbatch gpu.sh $seed $g_force $response_rate $reflex_threhold
            done
        done
    done
done