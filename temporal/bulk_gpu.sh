#!/usr/bin/env bash

count=0
for seed in {0..4}
do
  for env_name in InvertedPendulum-v2
      for response_rate in 0.01 0.02 0.04 0.08 0.16 0.32 0.64
      do
          if ! [ -f "models/TD3_${env_name}_${seed}_${response_rate}_True_final_actor" ];
          then
            sbatch gpu.sh $seed $response_rate $env_name
            count=$((count + 1))
          fi
      done
  done
done

echo Launched $count jobs