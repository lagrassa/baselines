#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 21 22 23 24 25 26 27 28 29 30 
do
    python trainable.py  StirEnv-v0 AL59b 0 0 85 ppo2 hyperparams/ppo2StirEnv-v0AL59obs_0act_0rw_85best_hyperparams.npy $i manual &
    sleep 1s
done


