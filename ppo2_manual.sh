#!/bin/bash
for i in 1 2 3 4
do
    python trainable.py  FetchReach-v1 AL69 0 0 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL57bobs_0act_0rw_0.05best_hyperparams.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL69 0 0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL57bobs_0act_0rw_0.05best_hyperparams.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL69 0 0 0.1 ppo2 hyperparams/ppo2FetchReach-v1AL57bobs_0act_0rw_0.05best_hyperparams.npy $i 0.0 manual &
done


