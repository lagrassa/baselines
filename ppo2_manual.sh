#!/bin/bash
for i in 6 8 9 10 
do
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.05 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.0 0.1 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #actuation noise
    #python trainable.py  FetchReach-v1 AL74 0.0 0.01 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.1 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.2 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.0 0.5 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.0 2 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    sleep 1s
    python trainable.py  FetchReach-v1 AL74 0.0 1 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    sleep 1s
    #obs noise

    python trainable.py  FetchReach-v1 AL74 0.01 0.0 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.1 0.0 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.2 0.0 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s
    #python trainable.py  FetchReach-v1 AL74 0.5 0.0 0.08 ppo2 hyperparams/test_ppo2frparams.npy $i 0.0 manual &
    #sleep 1s



done


