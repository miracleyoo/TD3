#!/usr/bin/env bash

count=0
for seed in 0
do
  for g_force in 8
  do
    for parent_response_rate in 0.04
    do
      for response_rate in 0.02
      do
        for reflex_threshold in 0.15
        do
          for reflex_force_scale in 0.5
          do
            if ! [ -f "hand_crafted_reflex_TD3_InvertedPendulum-v2_${seed}_0.02_${g_force}.0_${response_rate}_1.0_True_${parent_response_rate}_${reflex_threshold}_${reflex_force_scale}_final_actor" ];
            then
              sbatch hand_crafted_reflex.sh $seed $g_force $response_rate $parent_response_rate $reflex_threshold $reflex_force_scale
              count=$((count + 1))
            fi
          done
        done
      done
    done
  done
done

echo Launched $count jobs