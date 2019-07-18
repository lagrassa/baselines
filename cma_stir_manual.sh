#!/bin/bash
do
	python trainable.py  StirEnv-v0 AL74 0 0 81 cma hyperparams/cmaStirEnv-v0AL74obs_0act_0rw_81rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual 0 &
	    sleep 1s
done



