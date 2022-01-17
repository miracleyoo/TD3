#!/usr/bin/env bash

for seed in {0..4}
do
    for g_force in {0..10}
    do
        for response_rate in 0.04
        do
            for reflex_threhold in 0.15
            do
                for reflex_response_rate in 0.02:
                do
                  for reflex_force_scale in 1:
                  do
                    sbatch gpu.sh $seed $g_force $response_rate $reflex_threhold $reflex_response_rate $reflex_force_scale
                  done
                done
            done
        done
    done
done