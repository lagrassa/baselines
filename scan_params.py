from tune import best_hyperparams_for_config

algs = ["ppo2", "naf", "ddpg", "cma"]
env_name = "Pendulum-v0"

def get_formatted_name(params):
    return params['alg']+params['exp_name']+"obs_"+str(params['obs_noise_std'])+"act_"+str(params['action_noise_std'])

def optimize_hyperparams(env_name,params):
    for alg in algs:
        exp_name = get_formatted_name(params)
        best_hyperparams_for_config(params, exp_name)

"""
given a set of parameters, runs all algorithms with the optimal hyperparameters
runs them multiple times, preferably on different clusters (so with LLsub then)
saves the exp_name. might want to do using LLsub
"""
def run_action_space_experiment(num_samples, param_set, exp_name, env_name):
    #create experiment file based on params
    #run experiment using LLsub, probably alg by alg and params by params
    default_params = {'env_name':env_name, 'exp_name':exp_name, 'obs_noise_std':0, 'action_noise_std':0}
    for alg in algs:
        default_params['alg'] = alg
        sample_space = {0.1,0.2}
        for action_noise_std in sample_space:
            params = default_params.copy()
            params['action_noise_std'] = action_noise_std
            run_batch_job(params)

def run_batch_job(params):
    import os
    exp_name_with_params = get_formatted_name(params)
    np.save(exp_name_with_params, params)
    filename = write_batch_job(exp_name_with_params)
    os.system("LLsub "+filename)

def write_batch_job(params, num_processes=8):
    name = get_formatted_name(params)
    batch_file = open("batch_scripts/batch_job"+name+".sh", "w")
    batch_file.write("#!/bin/sh\n")
    batch_file.write("#SBATCH -o "+ name +".out-%j\n")
    batch_file.write("#SBATCH -a "+ "1-"+str(num_processes)+"\n")
    batch_file.write("#SBATCH --constraint=opteron\n")
    batch_file.write("source /etc/profile\nmodule load cuda-9.0 \n")
    batch_file.close()
    
            
def test_write_batch_job():
    default_params = {'env_name':"Pendulum-v0", 'exp_name':"test", 'obs_noise_std':0, 'action_noise_std':0, 'alg':'ddpg'}
    write_batch_job(default_params)
    f = open("batch_scripts/batch_job"+get_formatted_name(default_params)+".sh")
    print(f.read())
    f.close()

test_write_batch_job()
