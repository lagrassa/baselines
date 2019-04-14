from trainable import best_hyperparams_for_config
import numpy as np
import os
from helper import get_formatted_name
env_name = "FetchReach-v1"
algs = ["naf"]
def optimize_hyperparams(params, smoke_test = False):
    for alg in algs:
        exp_name = get_formatted_name(params)
        best_hyperparams_for_config(params, exp_name, smoke_test = smoke_test)

"""
given a set of parameters, runs all algorithms with the optimal hyperparameters
runs them multiple times, preferably on different clusters (so with LLsub then)
saves the exp_name. might want to do using LLsub
"""
def run_action_noise_experiment(num_samples, param_set, exp_name, env_name, LLcluster=True, smoke_test = False):
    #create experiment file based on params
    #run experiment using LLsub, probably alg by alg and params by params
    default_params = {'env_name':env_name, 'exp_name':exp_name, 'obs_noise_std':0, 'action_noise_std':0}
    for alg in algs:
        default_params['alg'] = alg
        sample_space = {0}
        #sample_space = {0.05,0.2}
        for action_noise_std in sample_space:
            params = default_params.copy()
            params['action_noise_std'] = action_noise_std
            hyperparam_file = get_formatted_name(params)+"best_hyperparams.npy"
            if hyperparam_file not in os.listdir("hyperparams"):
                optimize_hyperparams(params, smoke_test = smoke_test)
            else:
                print("Already found the hyperparams")
            run_batch_job(params, LLcluster=LLcluster)

""" precondition: hyperparameter optimization has already happened"""
def run_batch_job(params, LLcluster=True):
    import os
    exp_name_with_params = get_formatted_name(params)
    np.save("params/"+exp_name_with_params+"_params.npy", params)
    filename = write_batch_job(exp_name_with_params, LLcluster=LLcluster)
    if LLcluster:
        os.system("LLsub "+filename)
    else:
        os.system("./"+filename)

def write_batch_job(name, num_processes=8, LLcluster = True):
    filename = "batch_scripts/batch_job"+name+".sh"
    batch_file = open(filename, "w")
    batch_file.write("#!/bin/sh\n")
    if LLcluster:
        batch_file.write("#SBATCH -o "+ "batchlogs/"+name +".out-%j\n")
        batch_file.write("#SBATCH -C "+ "opteron"+"\n")
        batch_file.write("#SBATCH -a "+ "1-"+str(num_processes)+"\n")
        batch_file.write("#SBATCH -s "+ " 2"+"\n")
        batch_file.write("source /etc/profile\nmodule load cuda-9.0 \n")
        batch_file.write("python trainable.py "+name)
    else:
        for i in range(num_processes):
            batch_file.write("python trainable.py "+name +" "+str(i)+" & \n")
    batch_file.close()
    return filename
    
            
def test_write_batch_job():
    default_params = {'env_name':"Pendulum-v0", 'exp_name':"test", 'obs_noise_std':0, 'action_noise_std':0, 'alg':'naf'}
    write_batch_job(default_params)
    f = open("batch_scripts/batch_job"+get_formatted_name(default_params)+".sh")
    print(f.read())
    f.close()

#optimize_hyperparams({'env_name':"FetchPush-v1", 'exp_name':"test", 'obs_noise_std':0, 'action_noise_std':0, 'alg':'naf'})
#env_name = "FetchPush-v1"
exp_name="AL47"
param_set = {'env_name':env_name, 'exp_name':exp_name, 'obs_noise_std':0, 'action_noise_std':0}

#optimize_hyperparams(param_set, smoke_test = True)
if __name__=="__main__":
    import sys
    algs = [sys.argv[1]]
    LLcluster="nocluster" not in sys.argv
    print("LLcluster", LLcluster)
    if 'smoke' in sys.argv:
        param_set['alg'] =algs[0]
        #optimize_hyperparams(param_set, smoke_test = True)
        run_action_noise_experiment(10, param_set, exp_name, env_name, LLcluster=False, smoke_test = True)
    else:
        run_action_noise_experiment(10, param_set, exp_name, env_name, LLcluster=LLcluster)
