#!/bin/bash
for i in 1 2 3 4 5 6 7
do
    #python trainable.py  FetchPush-v1 AL74 0.0 0.0 0.05 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 0.0 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 0.0 0.1 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #actuation noise
    #python trainable.py  FetchPush-v1 AL74 0.0 0.01 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 0.1 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 0.2 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 0.5 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 2 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchPush-v1 AL74 0.0 1 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    #sleep 1s
    #obs noise

    python trainable.py  FetchPush-v1 AL74 0.01 0.0 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchPush-v1 AL74 0.1 0.0 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchPush-v1 AL74 0.2 0.0 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchPush-v1 AL74 0.5 0.0 0.08 her hyperparams/herFetchPush-v1AL51obs_0act_0rw_0.05best_params_so_far.npy $i 0.0 manual &
    sleep 1s


done


