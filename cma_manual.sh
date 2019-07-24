#!/bin/bash
for i in 1 2 3 4 5 6 7 8
do
    #python trainable.py  FetchReach-v1 AL47 0 0.1 0.05 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
    #python trainable.py  FetchReach-v1 AL47 0 0.2 0.05 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
    #python trainable.py  FetchReach-v1 AL47 0.1 0 0.05 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0 0.1 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0 0.08 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0 0.05 cma hyperparams/cmaFetchReach-v1AL47obs_0act_0rw_0.08best_params_so_far.npy $i manual &
done


