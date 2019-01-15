import ray
import baselines.ppo2.ppo2 as ppo2
import ray.tune as tune
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize

import ipdb
ray.init()

class TrainableClass(tune.Trainable):
    def _setup(self, arg):
        self.alpha = arg["alpha"]
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
        self.alg_module.learn_iter(**self.local_variables)
        self.nupdates += 1 
        return {'done':self.nupdates > self.nupdates_total, 'mean_accuracy':4}

    def _save(self, state):
        return {}

    def _restore(self):
        pass

def pick_params(trial_list):
    best_trial = max(trial_list, key=lambda trial: trial.last_result["mean_accuracy"])
    return best_trial.config

experiment_spec = tune.Experiment(
    "experiment_name2",
    TrainableClass,
    config = {"alpha": tune.grid_search([0.1, 0.001]),
              "lr":tune.grid_search([0.0003]),
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
    res = tune.run_experiments(experiments=experiment_spec)
    pick_params(res)

