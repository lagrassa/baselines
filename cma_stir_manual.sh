#!/bin/bash
for i in 1 2 3 4 5 6 7 8 
do
	python trainable.py  StirEnv-v0 AL74 0 0 81 cma hyperparams/cmaStirEnv-v0AL74obs_0act_0rw_81rew_noise_std_0.0best_hyperparams.npy $i 0 manual &
	    sleep 1s
done



