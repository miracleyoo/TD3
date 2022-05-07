#!/usr/bin/env bash

count=0
for seed in 0
do
    for g_force in 8
    do
        for reward_factor in 0
        do
          for response_rate in 0.02
          do
              if ! [ -f "models/fast_TD3_InvertedPendulum-v2_${seed}_0.02_${g_force}.0_${response_rate}_1.0_True_${$reward_factor}_final_actor" ];
              then
                sbatch gpu.sh $seed $g_force $response_rate $reward_factor
                count=$((count + 1))
              fi
          done
        done
    done
done

echo Launched $count jobs