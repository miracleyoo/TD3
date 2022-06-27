#!/usr/bin/env bash


for g_force in 8
do
  for response_rate in 0.02
  do
        sbatch eval_reflex_after_training.sh $g_force $response_rate
  done
done

