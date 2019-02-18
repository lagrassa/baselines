import ray
import threading
import ipdb
import json
import baselines.ppo2.ppo2 as ppo2
import ray.tune as tune
import numpy as np
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize
#from gym.envs.registration import register
GIT_DIR = "/home/gridsan/alagrassa/git/"

#register(
#    id='StirEnv-v0',
#    entry_point='cup_skills.floating_stirrer.World'
#)


def alg_to_module(alg):
    import sys
    sys.path.append(GIT_DIR+"handful-of-trials/scripts")
    sys.path.append(GIT_DIR+"handful-of-trials")
    sys.path.append(GIT_DIR+"NAF-tensorflow")
    import naf_tf_main
    alg_to_module = {'ppo2':ppo2,  'naf':naf_tf_main}
    return alg_to_module[alg]

import ipdb
ray.init()
def make_class(params):
    class TrainableClass(tune.Trainable):
        def _setup(self, arg):
            #self.alg_module = arg["alg_module"]
            env_name = params['env_name']
            self.exp_name = params['exp_name']
            self.alg = params['alg']
            self.lock = threading.Lock()
            self.best_success_rates = []
            self.best_success_rate = -np.inf
            self.alg_module = alg_to_module(self.alg)
            config = tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
            get_session(config=config)
            flatten_dict_observations = self.alg not in {'her'}
            sample_config, fixed_config, env_config = alg_to_config(params['alg'], env_name)
            sample_config_sample = {ky:arg[ky] for ky in sample_config.keys() }
            learn_params = {**sample_config_sample, **fixed_config}
            action_noise_std = params['action_noise_std']
            obs_noise_std = params['obs_noise_std']
            
            reward_scale = 1.0
            if "reward_scale" in arg.keys():
                reward_scale = arg["reward_scale"]
            env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std)
            if self.alg == "ppo2":
                env = VecNormalize(env)
            if 'env' not in learn_params.keys():
                learn_params['env'] = env 
             
            load_path = None
            self.local_variables = self.alg_module.learn_setup(**learn_params)

            self.nupdates_total = env_config["total_learning_iters"]
            self.nupdates = 1

        def _train(self):
            self.local_variables['update'] = self.nupdates
            _, success_rate = self.alg_module.learn_iter(**self.local_variables)
            self.lock.acquire()
            if success_rate > self.best_success_rate:
                self.best_success_rates.append(success_rate)
                self.best_success_rate = success_rate
            np.save(self.exp_name+"_best_success_rates.npy", self.best_success_rates)
            self.lock.release()

            self.nupdates += 1 
            return {'done':self.nupdates > self.nupdates_total or success_rate > 0.95, 'success_rate':success_rate}

        def _save(self, state):
            return {}

        def _restore(self):
            pass
    return TrainableClass

def pick_params(trial_list):
    best_trial = max(trial_list, key=lambda trial: trial.last_result["success_rate"])
    print("Best success rate was", best_trial.last_result["success_rate"])
    print(best_trial.config)
    return best_trial.config

def alg_to_config(alg, env_name=None):
    num_env = 1
    if alg == "ppo2":
        total_timesteps = int(4.6e6/3.0)
        nsteps = 2048
        nbatch = nsteps*num_env
        sample_config =  {"lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.00001, 0.1)),
                    "vf_coef": tune.sample_from(
                        lambda spec: np.random.uniform(0.4, 0.6)),
                    "max_grad_norm": tune.sample_from(
                        lambda spec: np.random.uniform(0.4, 0.6)),
                    "cliprange": tune.sample_from(
                        lambda spec: np.random.uniform(0.1, 0.3)),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.01, 10)),
                    "lam": tune.sample_from(
                        lambda spec: np.random.uniform(0.90, 0.99))}
        
        fixed_config = {'network':'mlp', 
                        'nsteps':nsteps, 
                        'total_timesteps':total_timesteps, 
                        'ent_coef':0.0,
                        'gamma':0.95,
                        'log_interval':10,
                        'nminibatches':4,
                        'noptepochs':4,
                        'save_interval':10}
        env_config = {'num_env':1,
                      'total_learning_iters' : total_timesteps//nbatch,}


    elif alg == "mbrl":
        sample_config =  {}
        fixed_config = {"ctrl_type":"MPC",
                        "ctrl_args":[],
                        "env":"pusher", 
                        "overrides":[],
                        "logdir":"log"}

        
        env_config = {'num_env':1,
                      'total_learning_iters' : 10,
        }
    elif alg=="naf":
        sample_config =  {"learning_rate": tune.sample_from(
                        lambda spec: np.random.choice([1e-2, 1e-3, 1e-4])),
                    "noise_scale": tune.sample_from(
                        lambda spec: np.random.choice([0.1, 0.3, 0.8, 1.0])),
                    "use_batch_norm": tune.sample_from(
                        lambda spec: np.random.choice([True, False]))}
        fixed_config = {'env_name':env_name}
        env_config = {'num_env':1,
                      'total_learning_iters' : 5,#1e2,
        }
    else:
        raise ValueError("Algorithm not supported")
    return sample_config, fixed_config, env_config

def run_async_hyperband(smoke_test = False, expname = "test", obs_noise_std=0, action_noise_std=0, params={}):
    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="success_rate",
        grace_period=50,#int(4.67e5/10.0),
        max_t=500) #int(4.67e5/4.0))
    #params = {'env_name':"FetchPush-v1", 'alg' : "ppo2", 'exp_name' : expname}
    if smoke_test:
        num_samples = 1
        num_cpu = 1
    else:
        num_samples = 150
        num_cpu = 12
    return tune.run_experiments(
        {
            "asynchyperband_test": {
                "run": make_class(params),
                "stop": {
                    "done": True
                },
                "num_samples": num_samples,
                "resources_per_trial": {
                    "cpu": num_cpu,
                    "gpu": 0
                },
                "config": alg_to_config(params['alg'])[0], #just the tuneable ones
            }
        },
scheduler=ahb)

def run_alg(exp_name, params):
    exp_name = sys.argv[1] 
    hyperparams_file = np.load(exp_name+"best_params.npy")
    params = np.load(exp_name+"params.npy")
    trainable = make_class(params)()
    #TODO call of the functions
    #get the env_id for logging

def best_hyperparams_for_config(params, exp_name):
    res = run_async_hyperband(expname=exp_name, smoke_test = True, params=params)
    best_params = pick_params(res)
    np.save(exp_name+"best_hyperparams.npy", best_params)
    return best_params



if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    import sys
    exp_name = sys.argv[1] 
    hyperparams_file = np.load(exp_name+"best_params.npy")
    params = np.load(exp_name+"params.npy")
    ray.shutdown()

