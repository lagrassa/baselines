#!/bin/bash
for i in 1 2 3 4 5
do
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.3 ddpg hyperparams/ddpgStirEnv-v0AL59cobs_0act_0rw_85best_hyperparams.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.4 ddpg hyperparams/ddpgStirEnv-v0AL59cobs_0act_0rw_85best_hyperparams.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.5 ddpg hyperparams/ddpgStirEnv-v0AL59cobs_0act_0rw_85best_hyperparams.npy $i 0.0 manual &
    sleep 1s
done





