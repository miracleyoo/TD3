#!/usr/bin/env bash

for seed in {0..4}
do
  for env_name in InvertedPendulum-v2
  do
      for response_rate in 0.04 0.08 0.12
      do
          for jit_duration in 0.02
          do
            sbatch normal_perturbation.sh $seed $response_rate $env_name $jit_duration
          done
      done
  done
done

