import threading
import os
from helper import get_formatted_name
import json
import numpy as np
import ray.tune as tune
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize
#from gym.envs.registration import register
GIT_DIR = os.environ["HOME"]+"/git/"
SAVE_DIR = os.environ["HOME"]+"/git/baselines/"
alg_to_iters = {'ppo2':1e6, 'ddpg':1e7, 'naf':1e6, 'cma':1e6, 'her':90000}
#alg_to_iters = {'ppo2':800, 'ddpg':1e7, 'naf':80, 'cma':1e5, 'her':5000}

#register(
#    id='StirEnv-v0',
#    entry_point='cup_skills.floating_stirrer.World'
#)


def alg_to_module(alg):
    import sys
    sys.path.append(GIT_DIR+"NAF-tensorflow/src/")
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
    elif alg == "her":
        import baselines.her.her as her
        return her



def make_class(params):
    import ray.tune as tune
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
            print("setting up tf stuff")
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
            goal_radius = params['goal_radius']
            
            reward_scale = 1.0
            if "reward_scale" in arg.keys():
                reward_scale = arg["reward_scale"]
            total_iters = alg_to_iters[self.alg]
            self.nupdates_total = total_iters//(learn_params['n_episodes']*learn_params["n_steps_per_episode"])
            self.nupdates = 1
            env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std, distance_threshold=goal_radius)
            #env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std)
            if self.alg == "ppo2":
                #env = VecNormalize(env)
                learn_params["nupdates"] = self.nupdates_total
            if 'env' not in learn_params.keys():
                learn_params['env'] = env 
             
            load_path = None
            self.local_variables = self.alg_module.learn_setup(**learn_params)

        def _train(self):
            self.local_variables['update'] = self.nupdates
            if self.nupdates % 3 == 0:
                print("nupdates", self.alg, self.nupdates)
            if self.alg == "ppo2":
                num_iters = 1#20#self.local_variables['n_episodes']
            else:
                num_iters = 1
            for i in range(num_iters):
                self.alg_module.learn_iter(**self.local_variables)
            #test_success_rate = self._test(n_test_rollouts=15)['success_rate']
            test_success_rate = -0.17
            self.lock.acquire()
            if test_success_rate > self.best_success_rate:
                self.best_success_rates.append(test_success_rate)
                self.best_success_rate = test_success_rate
                print("New best success rate of ", test_success_rate)
                if test_success_rate > 0.30:
                    np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"best_params_so_far.npy", self.sample_config_bound)
                    np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"_best_success_rates.npy", self.best_success_rates)
            self.lock.release()

            self.nupdates += 1 
            return {'done':self.nupdates > self.nupdates_total or test_success_rate > 0.6, 'success_rate':test_success_rate}

        def _save(self, state):
            return {}

        def _restore(self):
            pass
        '''
        use current model in evaluation for n_test_rollouts
        '''
        def _test(self, n_test_rollouts=15):
            self.test_local_variables = self.local_variables.copy()
            self.test_local_variables['n_episodes'] = 1
            if self.alg == "cma":
                self.test_local_variables['n_episodes'] = 2
            self.test_local_variables['n_steps_per_iter'] = 50
            self.test_local_variables['n_episodes'] = n_test_rollouts
            success_rate = self.alg_module.learn_test(**self.test_local_variables)
            return {'success_rate':success_rate}
            
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
        sample_config =  {"lr": tune.sample_from(
            lambda spec: np.random.choice([3e-5, 3e-4, 3e-3])),
                    "vf_coef": tune.sample_from(
                        lambda spec: np.random.uniform(0.5, 0.505)),
                    "ent_coef":tune.sample_from(
                        lambda spec: np.random.uniform(0, 0.01)),
                    "max_grad_norm": tune.sample_from(
                        lambda spec: np.random.uniform(0.495, 0.505)),
                    "cliprange": tune.sample_from(
                        lambda spec: np.random.uniform(0.1, 0.3)),
                    "seed": tune.sample_from(
                        lambda spec: np.random.choice([51])),
                    "network": tune.sample_from(
                        lambda spec: np.random.choice(['mlp'])),
                    #"n_steps_per_episode": tune.sample_from(
                    #    lambda spec: np.random.choice([64, 512, 1024, 2048])),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.001, 1)),
                    "lam": tune.sample_from(
                        lambda spec: np.random.uniform(0.90, 0.99))}
        
        fixed_config = {'n_episodes':40, 
                        'n_steps_per_episode':50,#50, 
                        'gamma':0.99,
                        'log_interval':10,
                        'nminibatches':5,
                        'total_timesteps':1e6,
                        'noptepochs':10,
                        'save_interval':10}
        env_config = {'num_env':1}
    elif alg == "ddpg":
        sample_config =  {"actor_lr": tune.sample_from(
                        lambda spec: np.random.uniform(1e-5, 0.01)),
                    "batch_size": tune.sample_from(
                        lambda spec: np.random.choice([64, 512, 1024, 2048])),
                    "critic_lr": tune.sample_from(
                        lambda spec: np.random.uniform(1e-5, 0.01)),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.01, 100)) }
        
        fixed_config = {'network':'mlp', 
                        'n_episodes':200,
                        'n_steps_per_episode':50,
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
                        lambda spec: int(np.random.uniform(3,20))),
                    "CMA_cmean": tune.sample_from(
                        lambda spec: np.random.uniform(0.99, 1.01)),
                    "CMA_rankmu": tune.sample_from(
                        lambda spec: np.random.uniform(0.99, 1.01)),
                    "CMA_rankone": tune.sample_from(
                        lambda spec: np.random.uniform(0.99, 1.01))}
        
        fixed_config = {'env_name':env_name,
                        'n_episodes':17,
                        'n_steps_per_episode':50}
        env_config = {'num_env':1}
    elif alg=="naf":
        sample_config =  {"learning_rate": tune.sample_from(
                        lambda spec: np.random.choice([1e-2, 1e-3, 1e-4])),
                    "noise_scale": tune.sample_from(
                        lambda spec: np.random.choice([0.1, 0.3, 0.8, 1.0])),
                    "batch_size": tune.sample_from(
                        lambda spec: np.random.choice([100, 200])),
                    "seed": tune.sample_from(
                        lambda spec: np.random.choice([1,5,17,47,51])),
                    "im_rollouts": tune.sample_from(
                        lambda spec: np.random.choice([True, False])),
                    "use_batch_norm": tune.sample_from(
                        lambda spec: np.random.choice([True, False]))}
        fixed_config = {'env_name':env_name,
                'n_episodes':40,
                'n_steps_per_episode':50}  #NBATCH_STANDARD}
        env_config = {'num_env':1
        }
    elif alg=="her":
        sample_config =  {"seed": tune.sample_from(
                        lambda spec: np.random.choice([1, 5, 17, 24, 25, 14, 47]))}
        fixed_config = {'network':'mlp',
                'policy_save_interval':20,
                'n_cycles':10,#10,
                'n_batches':40,
                'n_test_rollouts':5,
                'n_steps_per_episode':50}  # better at 100 for some reason NBATCH_STANDARD}
        fixed_config['n_episodes'] = fixed_config['n_cycles']
        env_config = {'num_env':1
        }
    else:
        raise ValueError("Algorithm not supported")
    return sample_config, fixed_config, env_config

def run_async_hyperband(smoke_test = False, expname = "test", obs_noise_std=0, action_noise_std=0, params={}):
    if smoke_test:
        grace_period = 1
        max_t = 10
        num_samples = 1
        num_cpu = 1
        NBATCH_STANDARD = 10
    else:
        grace_period = 4
        max_t = 2e6 #this doesn't actually mean anything. Trainable takes care of killing processes when they go on for too long
        num_samples = 10
        num_cpu = 10

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
    scheduler=ahb, queue_trials=True, verbose=0)
"""
Precondition: hyperparameter optimization happened
"""
def run_alg(params, iters=2,hyperparam_file = None, LLcluster=True, exp_number=None):
    exp_name = get_formatted_name(params)
    if hyperparam_file is None:
        hyperparam_file = "hyperparams/"+exp_name+"best_hyperparams.npy"
    hyperparams = np.load(hyperparam_file).all()

    args = {"sample_config_best":hyperparams}
    #overwrite the old ones
    args["obs_noise_std"] = params["obs_noise_std"]
    args["action_noise_std"] = params["action_noise_std"]
    args["goal_radius"] = params["goal_radius"]
    trainable = make_class(params)(args)
    print("made class")
    test_success_rates = []
    SAVE_INTERVAL=10
    if LLcluster and exp_number is None:
        exp_number = os.environ["SLURM_ARRAY_TASK_ID"]
    for i in range(int(iters)):
        test_res = trainable._train()
        test_success_rates.append(test_res['success_rate'])

        if i % SAVE_INTERVAL == 0:
            np.save("run_results/"+exp_name+"test_success_rates_"+str(exp_number)+".npy", test_success_rates)

    #TODO call of the functions
    #get the env_id for logging

def best_hyperparams_for_config(params, exp_name, smoke_test = False):
    import ray
    ray.init()
    res = run_async_hyperband(expname=exp_name, smoke_test = smoke_test, params=params)
    best_params = pick_params(res, exp_name)
    if not smoke_test:
        np.save("hyperparams/"+get_formatted_name(params)+"best_hyperparams.npy", best_params)
    ray.shutdown()
    return best_params

def test_run_params():
    for alg in ['naf']:
        params = {'env_name':"Pendulum-v0", 'exp_name':"test", 'obs_noise_std':0, 'action_noise_std':0, 'alg':alg}
        run_alg(params)

if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    print("running trainable.py")
    import sys
    if "manual" in sys.argv:
        params = {'env_name':sys.argv[1], 'exp_name':sys.argv[2], 'obs_noise_std':float(sys.argv[3]), 'action_noise_std':float(sys.argv[4]), 'goal_radius':float(sys.argv[5]),'alg':sys.argv[6]}
        #params['lr'] = 1e-3
        print(params)
        n_episodes=alg_to_config(params["alg"])[1]['n_episodes']
        num_iters = alg_to_iters[params["alg"]]//n_episodes
        run_alg(params, iters=int(num_iters), hyperparam_file = sys.argv[7], LLcluster=False, exp_number=int(sys.argv[8]))

    else:
        exp_name = sys.argv[1] 
        params= np.load("params/"+exp_name+"_params.npy").all()
        params["nsteps"] = 2048
        alg = params['alg']
        params['seed']=1
        n_episodes=alg_to_config(params["alg"])[1]['n_episodes']
        num_iters = alg_to_iters[alg]//n_episodes
        if len(sys.argv) > 2:
            run_alg(params, iters=num_iters, exp_number = int(sys.argv[2]))
        else:
            run_alg(params, iters=num_iters)
    #hyperparams_file = np.load(exp_name+"best_params.npy")
    import ray
    ray.shutdown()

