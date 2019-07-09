#!/bin/bash
for i in 1 2 3 4 5 6 7 
do
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.1 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #actuation noise
    python trainable.py  FetchReach-v1 AL74 0.0 0.01 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.0 0.1 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.0 0.2 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.0 0.5 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    #obs noise

    python trainable.py  FetchReach-v1 AL74 0.001 0.0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.1 0.0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.2 0.0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.5 0.0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s



done


