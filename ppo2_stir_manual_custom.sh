#!/bin/bash
for i in 1 2 3 4
do
    python trainable.py  StirEnv-v0 AL58 0 0 74 ppo2 hyperparams/ppo2StirEnv-v0AL57bobs_0act_0rw_0.05best_hyperparams.npy $i manual &
    sleep 1s
done


