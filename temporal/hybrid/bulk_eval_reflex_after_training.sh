#!/usr/bin/env bash


for g_force in 5
do
  for response_rate in 0.02
  do
    for reflex_force_scale in 0.5
      do
        for angle in 0.15
        do
          sbatch eval_reflex_after_training.sh $g_force $response_rate $angle $reflex_force_scale
        done
      done
  done
done

