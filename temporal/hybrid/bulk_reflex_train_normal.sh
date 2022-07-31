#!/usr/bin/env bash

for seed in {0..4}; do
  for response_rate in 0.04; do
    for parent_response_rate in 0.08 0.12; do

      for env_name in InvertedPendulum-v2; do
        sbatch reflex_train_normal.sh $seed $response_rate $parent_response_rate $env_name
      done
    done
  done
done
