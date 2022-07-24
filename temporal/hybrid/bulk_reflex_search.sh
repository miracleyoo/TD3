#!/usr/bin/env bash

for seed in 0
do
  for g_force in 0
  do
    for response_rate in 0.02
    do
      for parent_response_rate in 0.04
      do
        for population in 20
        do
          for jit_duration in 0.02
          do
            for env_name in InvertedPendulum-v2
            do
              sbatch reflex_search.sh $seed $g_force $response_rate $population $parent_response_rate $jit_duration $env_name
            done
          done
        done
      done
    done
  done
done
