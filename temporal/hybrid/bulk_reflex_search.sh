#!/usr/bin/env bash

for seed in 0
do
  for g_force in 0
  do
      for response_rate in 0.02
      do
        for population in 20
        do
            sbatch reflex_search.sh $seed $g_force $response_rate $population
        done
      done
  done
done
