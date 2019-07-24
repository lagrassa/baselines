#!/bin/bash
for i in 1 2 3 4 5 6 7 8
do
	python trainable.py  StirEnv-v0 AL59d 0 0 85 ddpg hyperparams/ddpgStirEnv-v0AL59cobs_0act_0rw_85best_hyperparams.npy $i manual &
	    sleep 1s
done


