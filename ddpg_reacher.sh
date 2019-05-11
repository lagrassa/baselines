#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9
do
    python trainable.py  Reacher-v2 AL60 0 0 85 ddpg hyperparams/ddpgReacher-v0AL60obs_0act_0rw_85best_hyperparams.npy $i manual  & 
    sleep 1s
done
