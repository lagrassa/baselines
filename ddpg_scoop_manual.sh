#!/bin/bash
for i in 1 2
do
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.3 ddpg hyperparams/ddpgScoopEnv-v0AL74ctestobs_0act_0rw_81rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.4 ddpg hyperparams/ddpgScoopEnv-v0AL74ctestobs_0act_0rw_81rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.5 ddpg hyperparams/ddpgScoopEnv-v0AL74ctestobs_0act_0rw_81rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
done





