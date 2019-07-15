#!/bin/bash
for i in 1 2 3 4 5 6
do
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.3 ppo2 hyperparams/ppo2ScoopEnv-v0AL74aobs_0act_0rw_0rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s    
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.4 ppo2 hyperparams/ppo2ScoopEnv-v0AL74aobs_0act_0rw_0rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s    
    python trainable.py  ScoopEnv-v0 AL74 0.0 0.0 0.5 ppo2 hyperparams/ppo2ScoopEnv-v0AL74aobs_0act_0rw_0rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s    
done



