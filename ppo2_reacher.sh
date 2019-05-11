#!/bin/bash
for i in 1 2 3 4 5 6
do
    python trainable.py  Reacher-v2 AL60 0 0 85 ppo2 hyperparams/ppo2Reacher-v1AL60obs_0act_0rw_30best_hyperparams.npy $i manual &
sleep 1s
done
