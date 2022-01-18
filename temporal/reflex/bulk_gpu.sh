#!/usr/bin/env bash

count=0
for seed in {0..4}
do
    for g_force in {0..10}
    do
        for response_rate in 0.04
        do
            for reflex_threhold in 0.15
            do
                for reflex_response_rate in 0.02:
                do
                  for reflex_force_scale in 1:
                  do
                    if ! [ -f "models/TD3_reflex_InvertedPendulum-v2_${seed}_0.02_${g_force}.0_${response_rate}_1.0_${reflex_response_rate}_${reflex_threhold}_${reflex_force_scale}_final_actor" ];
                    then
                      sbatch gpu.sh $seed $g_force $response_rate $reflex_threhold $reflex_response_rate $reflex_force_scale
                      count=$((count + 1))
                    fi
                  done
                done
            done
        done
    done
done

echo Launched $count jobs