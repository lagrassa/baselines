#!/bin/bash
for i in 6 7 8 9 10 
do
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.05 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74b 0.0 0.0 0.1 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #actuation noise
    #python trainable.py  FetchReach-v1 AL74b 0.0 0.1 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74b 0.0 0.5 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 1.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 2.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #obs noise

    python trainable.py  FetchReach-v1 AL74b 0.1 0.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74b 0.5 0.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74b 1.0 0.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74b 2.0 0.0 0.08 cma hyperparams/cmaFetchReach-v1AL74bobs_0act_0rw_0.05rew_noise_std_0.0best_params_so_far.npy $i 0.0 manual &
    sleep 1s



done


