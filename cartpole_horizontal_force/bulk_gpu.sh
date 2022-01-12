#!/usr/bin/env bash

count=0
for seed in {0..4}
do
    for g_force in {0..21}
    do
        for neurons in 256
        do
          for response_rate in 0.01 0.02 0.04 0.08 0.16 0.32 0.64
          do
              if ! [ -f "models/TD3_InvertedPendulum-v2_${seed}_0.02_${g_force}.0_${response_rate}_1.0_False_256_final_actor"];
              then
                sbatch gpu.sh $seed $g_force $response_rate $neurons
                count=$((count + 1))
              fi
          done
        done
    done
done

echo Launched $count jobs