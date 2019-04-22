#!/bin/bash
for i in 1 2 3
do
    python trainable.py  FetchReach-v1 AL47 0 0 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0 0.08 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0 0.1 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0.01 0 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0.1 0 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0.01 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
    python trainable.py  FetchReach-v1 AL47 0 0.1 0.05 ppo2 hyperparams/ppo2FetchReach-v1AL47obs_0act_0rw_0.05best_hyperparams.npy $i manual &
done






