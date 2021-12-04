#!/usr/bin/env bash

for seed in {0..4}
do
    for g_force in {0..10}
    do
        for neurons in 256
        do
          for response_rate in 0.04
          do
              sbatch gpu.sh $seed $g_force $response_rate $neurons
          done
        done
    done
done