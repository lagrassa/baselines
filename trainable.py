import ray
import threading
from helper import get_formatted_name
import json
import ray.tune as tune
import numpy as np
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize
#from gym.envs.registration import register
GIT_DIR = "/home/lagrassa/git/"
SAVE_DIR = "/home/lagrassa/git/baselines/"
NBATCH_STANDARD = 50

#register(
#    id='StirEnv-v0',
#    entry_point='cup_skills.floating_stirrer.World'
#)


def alg_to_module(alg):
    import sys
    sys.path.append(GIT_DIR+"NAF-tensorflow")
    if alg == 'ppo2':
        import baselines.ppo2.ppo2 as ppo2
        return  ppo2
    elif alg == 'naf':
        import naf_tf_main
        return naf_tf_main
    elif alg == 'pets':
        sys.path.append(GIT_DIR+"handful-of-trials/scripts")
        sys.path.append(GIT_DIR+"handful-of-trials")
        print ("error, pets not supported")

    elif alg == 'cma':
        import cma_agent.run_experiment as cma
        return cma
       
    elif alg == "ddpg":
        import baselines.ddpg.ddpg as ddpg
        return ddpg



def make_class(params):
    class TrainableClass(tune.Trainable):
        def _setup(self, arg):
            #self.alg_module = arg["alg_module"]
            env_name = params['env_name']
            self.exp_name = params['exp_name']
            self.params = params
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
            if 'sample_config_best' not in arg.keys():
                sample_config_sample = {ky:arg[ky] for ky in sample_config.keys() }
                sample_config_bound = sample_config_sample
                learn_params = {**sample_config_sample, **fixed_config}
            else:
                learn_params = {**arg['sample_config_best'], **fixed_config}
                sample_config_bound = arg['sample_config_best']
            self.sample_config_bound = sample_config_bound 
            action_noise_std = params['action_noise_std']
            obs_noise_std = params['obs_noise_std']
            
            reward_scale = 1.0
            if "reward_scale" in arg.keys():
                reward_scale = arg["reward_scale"]
            total_iters = 1e5
            self.nupdates_total = int(round(total_iters/NBATCH_STANDARD))
            self.nupdates = 1
            env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std)
            if self.alg == "ppo2":
                env = VecNormalize(env)
                learn_params["nupdates"] = self.nupdates_total
            if 'env' not in learn_params.keys():
                learn_params['env'] = env 
             
            load_path = None
            self.local_variables = self.alg_module.learn_setup(**learn_params)

        def _train(self):
            self.local_variables['update'] = self.nupdates
            if self.nupdates % 10 == 0:
                print("nupdates", self.alg, self.nupdates)
            _, success_rate = self.alg_module.learn_iter(**self.local_variables)
            print("success_rate", success_rate)
            self.lock.acquire()
            if success_rate > self.best_success_rate:
                self.best_success_rates.append(success_rate)
                self.best_success_rate = success_rate
                np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"best_params_so_far.npy", self.sample_config_bound)
                np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"_best_success_rates.npy", self.best_success_rates)
            self.lock.release()

            self.nupdates += 1 
            return {'done':self.nupdates > self.nupdates_total or success_rate > 0.4, 'success_rate':success_rate}

        def _save(self, state):
            return {}

        def _restore(self):
            pass
    return TrainableClass

def pick_params(trial_list, exp_name="noexpname"):
    best_trial = max(trial_list, key=lambda trial: trial.last_result["success_rate"])
    print("Best success rate was", best_trial.last_result["success_rate"])
    np.save("hyperparams/"+exp_name+"best_value.npy", best_trial.last_result["success_rate"])
    threshold = 0.2
    #assert(best_trial.last_result["success_rate"] > threshold)
    print(best_trial.config)
    return best_trial.config

def alg_to_config(alg, env_name=None):
    num_env = 1
    if alg == "ppo2":
        nsteps = NBATCH_STANDARD
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
                        'ent_coef':0.0,
                        'gamma':0.95,
                        'log_interval':10,
                        'nminibatches':5,
                        'noptepochs':4,
                        'save_interval':10}
        env_config = {'num_env':1}
    elif alg == "ddpg":
        sample_config =  {"actor_lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.00001, 0.1)),
                    "critic_lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.00001, 0.1)),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.01, 10)) }
        
        fixed_config = {'network':'mlp', 
                        'nb_epoch_cycles':NBATCH_STANDARD,
                        'nb_rollout_steps':40,
                        'gamma':0.95}
        env_config = {'num_env':1}

    elif alg == "mbrl":
        sample_config =  {}
        fixed_config = {"ctrl_type":"MPC",
                        "ctrl_args":[],
                        "env":"pusher", 
                        "overrides":[],
                        "logdir":"log"}

        
        env_config = {'num_env':1
        }
    elif alg == "cma":
        sample_config =  {"CMA_mu": tune.sample_from(
                        lambda spec: int(np.random.uniform(3,NBATCH_STANDARD//2))),
                    "CMA_cmean": tune.sample_from(
                        lambda spec: np.random.uniform(0.8, 1.2)),
                    "CMA_rankmu": tune.sample_from(
                        lambda spec: np.random.uniform(0.8, 1.2)),
                    "CMA_rankone": tune.sample_from(
                        lambda spec: np.random.uniform(0.8, 1.2))}
        
        fixed_config = {'env_name':env_name, 
                        'nbatch_standard':NBATCH_STANDARD}
        env_config = {'num_env':1}
    elif alg=="naf":
        sample_config =  {"learning_rate": tune.sample_from(
                        lambda spec: np.random.choice([1e-2, 1e-3, 1e-4])),
                    "noise_scale": tune.sample_from(
                        lambda spec: np.random.choice([0.1, 0.3, 0.8, 1.0])),
                    "use_batch_norm": tune.sample_from(
                        lambda spec: np.random.choice([True, False]))}
        fixed_config = {'env_name':env_name,
                'nbatch_standard':50} #NBATCH_STANDARD}
        env_config = {'num_env':1
        }
    else:
        raise ValueError("Algorithm not supported")
    return sample_config, fixed_config, env_config

def run_async_hyperband(smoke_test = False, expname = "test", obs_noise_std=0, action_noise_std=0, params={}):
    if smoke_test:
        grace_period = 1
        max_t = 5
        num_samples = 1
        num_cpu = 1
        NBATCH_STANDARD = 10
    else:
        grace_period = 4
        max_t = 1e5//50#NBATCH_STANDARD
        num_samples = 30
        num_cpu = 8

    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="success_rate",
        grace_period=grace_period,#int(4.67e5/10.0),
        max_t=max_t) #int(4.67e5/4.0))
    #params = {'env_name':"FetchPush-v1", 'alg' : "ppo2", 'exp_name' : expname}
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
scheduler=ahb, queue_trials=True, verbose=0, resume=False)
"""
Precondition: hyperparameter optimization happened
"""
def run_alg(params, iters=2):
    exp_name = get_formatted_name(params)
    hyperparams = np.load("hyperparams/"+exp_name+"best_hyperparams.npy").all()
    args = {"sample_config_best":hyperparams}
    trainable = make_class(params)(args)
    success_rates = []
    SAVE_INTERVAL=1
    import os
    exp_number = os.environ["SLURM_ARRAY_TASK_ID"]
    for i in range(iters):
      res = trainable._train()
      success_rates.append(res['success_rate'])
      if i % SAVE_INTERVAL == 0:
          np.save("run_results/"+exp_name+"success_rates_"+str(exp_number)+".npy", success_rates)

    #TODO call of the functions
    #get the env_id for logging

def best_hyperparams_for_config(params, exp_name, smoke_test = False):
    ray.init()
    res = run_async_hyperband(expname=exp_name, smoke_test = smoke_test, params=params)
    best_params = pick_params(res, exp_name)
    np.save("hyperparams/"+get_formatted_name(params)+"best_hyperparams.npy", best_params)
    ray.shutdown()
    return best_params

def test_run_params():
    for alg in ['naf']:
        params = {'env_name':"Pendulum-v0", 'exp_name':"test", 'obs_noise_std':0, 'action_noise_std':0, 'alg':alg}
        run_alg(params)

if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    import sys
    exp_name = sys.argv[1] 
    params= np.load("params/"+exp_name+"_params.npy").all()
    run_alg(params, iters=int(1e6))
    #hyperparams_file = np.load(exp_name+"best_params.npy")
    ray.shutdown()

