import ray
import baselines.ppo2.ppo2 as ppo2
import ray.tune as tune
import numpy as np
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize

import ipdb
ray.init()

class TrainableClass(tune.Trainable):
    def _setup(self, arg):
        lr = arg["lr"]
        vf_coef = arg["vf_coef"]
        max_grad_norm = arg["max_grad_norm"]
        reward_scale = arg["reward_scale"]
        cliprange = arg["cliprange"]
        lam = arg["lam"]
        #self.alg_module = arg["alg_module"]
        self.alg_module = ppo2
        num_env = 1
        nsteps = 2048
        total_timesteps = 3000
        self.alg = "ppo2" #TODO link these and make them go into a config
        self.total_timesteps = 2055
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)
        flatten_dict_observations = self.alg not in {'her'}
        env = make_vec_env("FetchReach-v1", "mujoco", num_env or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations)

        env = VecNormalize(env)

        load_path = None
        self.local_variables = self.alg_module.learn_setup(network='mlp', env=env, total_timesteps=self.total_timesteps,  nsteps=nsteps, ent_coef=0.0, lr=lr, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gamma=0.99, lam=lam, log_interval=10, nminibatches=4, noptepochs=4, cliprange=cliprange, save_interval=10,load_path=load_path)
        self.goal_val = 3
        self.max_iter = 10
        self.val = 4
        nbatch = num_env * nsteps
        self.nupdates_total = total_timesteps//nbatch
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

def pick_params(trial_list):
    best_trial = max(trial_list, key=lambda trial: trial.last_result["success_rate"])
    return best_trial.config

def run_async_hyperband(smoke_test = False):
    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="success_rate",
        grace_period=5,
        max_t=100)
    return tune.run_experiments(
        {
            "asynchyperband_test": {
                "run": TrainableClass,
                "stop": {
                    "done": True
                },
                "num_samples": 1,
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 0
                },
                "config": {
                    "lr": tune.sample_from(
                        lambda spec: np.random.uniform(0.001, 0.1)),
                    "vf_coef": tune.sample_from(
                        lambda spec: np.random.uniform(0.4, 0.6)),
                    "max_grad_norm": tune.sample_from(
                        lambda spec: np.random.uniform(0.4, 0.6)),
                    "cliprange": tune.sample_from(
                        lambda spec: np.random.uniform(0.1, 0.3)),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.01, 10)),
                    "lam": tune.sample_from(
                        lambda spec: np.random.uniform(0.92, 0.99)),
                },
            }
        },
scheduler=ahb)

experiment_spec = tune.Experiment(
    "experiment_name2",
    TrainableClass,
    config = {"lr":tune.grid_search([0.0003]),
              "vf_coef":tune.grid_search([0.5]),
              "max_grad_norm":tune.grid_search([0.5]),
              "cliprange":tune.grid_search([0.2]),
              "reward_scale":tune.grid_search([1.0]),
              "lam":tune.grid_search([0.95]),
              }, 
    resources_per_trial={
        "cpu":1,
        "gpu":0
    }, 
    num_samples=1,
    stop = {"done":True},
    local_dir = "~/models",
    checkpoint_freq = 5)

if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    res = run_async_hyperband()
    pick_params(res)

