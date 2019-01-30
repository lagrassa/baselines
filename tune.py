import ray
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
GIT_DIR = "/home/lagrassa/git/"

#register(
#    id='StirEnv-v0',
#    entry_point='cup_skills.floating_stirrer.World'
#)


def alg_to_module(alg):
    import sys
    sys.path.append(GIT_DIR+"handful-of-trials/scripts")
    sys.path.append(GIT_DIR+"handful-of-trials")
    try:
        import mbexp
    except:
        ipdb.set_trace()
        print(sys.path)
    alg_to_module = {'ppo2':ppo2, 'mbrl':mbexp}
    return alg_to_module[alg]

import ipdb
ray.init()
def make_class(params):
    class TrainableClass(tune.Trainable):
        def _setup(self, arg):
            #self.alg_module = arg["alg_module"]
            env_name = params['env_name']
            self.alg = params['alg']
            self.alg_module = alg_to_module(self.alg)
            config = tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
            get_session(config=config)
            flatten_dict_observations = self.alg not in {'her'}
            sample_config, fixed_config, env_config = alg_to_config(params['alg'])
            sample_config_sample = {ky:arg[ky] for ky in sample_config.keys() }
            learn_params = {**sample_config_sample, **fixed_config}
            
            reward_scale = 1.0
            if "reward_scale" in arg.keys():
                reward_scale = arg["reward_scale"]
            env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations)
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
    return best_trial.config

def alg_to_config(alg):
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
    else:
        raise ValueError("Algorithm not supported")
    return sample_config, fixed_config, env_config

def run_async_hyperband(smoke_test = False, expname = "test"):
    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="success_rate",
        grace_period=int(4.67e5/10.0),
        max_t=int(4.67e5/4.0))
    params = {'env_name':"StirEnv-v0", 'alg' : "ppo2", 'exp_name' : expname}
    #params = {'env_name':"FetchPush-v1", 'alg' : "ppo2", 'exp_name' : expname}
    if smoke_test:
        num_samples = 1
        num_cpu = 1
    else:
        num_samples = 150
        num_cpu = 20
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


if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    import sys
    exp_name = sys.argv[1]
    res = run_async_hyperband(expname=exp_name, smoke_test = True)
    print("Finding best parameters...")
    best_params = pick_params(res)
    with open("params/"+exp_name, "w") as fp:
        json.dump(best_params, fp)
    ray.shutdown()

