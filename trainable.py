import threading
from datetime import datetime
import os
from helper import get_formatted_name, get_short_form_name
import json
import numpy as np
import ray.tune as tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import tensorflow as tf
import baselines
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize
#from gym.envs.registration import register
GIT_DIR = os.environ["HOME"]+"/git/"
SAVE_DIR = os.environ["HOME"]+"/git/baselines/"
train_alg_to_iters = {'ppo2':1e5, 'ddpg':1e5, 'naf':1e6, 'cma':1e5, 'her':90000}
tune_alg_to_iters = {'ppo2':300, 'ddpg':100, 'naf':80, 'cma':300, 'her':5000//50}
tune_alg_to_iters = {'ppo2':30, 'ddpg':30, 'naf':80, 'cma':30, 'her':5000//50}
#n_steps_per_iter_per_env = {'StirEnv-v0':18, 'Reacher-v2':50, 'FetchPush-v1':50, 'FetchReach-v1':50, 'ScoopEnv-v0':40}
n_steps_per_iter_per_env = {'StirEnv-v0':18, 'Reacher-v2':50, 'FetchPush-v1':50, 'FetchReach-v1':50, 'ScoopEnv-v0':42}
n_episodes_per_env = {'StirEnv-v0':8, 'Reacher-v2':40, 'FetchPush-v1':40, 'FetchReach-v1':40, 'ScoopEnv-v0':4} #was 10 for a while.... 
#tune_alg_to_iters = {'ppo2':800, 'ddpg':80, 'naf':80, 'cma':20, 'her':5000}

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
            self.env_name = env_name
            self.exp_name = params['exp_name']
            self.save_dir = SAVE_DIR+"tune_run_results/"+get_short_form_name(params)
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
            self.params = params
            self.alg = params['alg']
            self.lock = threading.Lock()
            self.best_success_rates = []
            date_object = datetime.now()
            self.exp_start_time = date_object.strftime('%Y-%m-%d')
            self.best_success_rate = -np.inf
            self.alg_module = alg_to_module(self.alg)
            config = tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
            get_session(config=config)
            force_flat = False
            if self.alg =="her":
                force_flat = False

            sample_config, fixed_config, env_config, cont_space = alg_to_config(params['alg'], env_name, force_flat = force_flat)
            if 'sample_config_best' not in arg.keys():
                sample_config_sample = {ky:arg[ky] for ky in sample_config.keys() }
                sample_config_bound = sample_config_sample
                learn_params = {**sample_config_sample, **fixed_config}
                total_iters = tune_alg_to_iters[self.alg]
            else:
                learn_params = {**arg['sample_config_best'], **fixed_config}
                sample_config_bound = arg['sample_config_best']
                total_iters = train_alg_to_iters[self.alg]
            self.sample_config_bound = sample_config_bound 
            action_noise_std = params['action_noise_std']
            obs_noise_std = params['obs_noise_std']
            rew_noise_std = params['rew_noise_std']
            goal_radius = params['goal_radius']
            
            reward_scale = 1.0
            if "reward_scale" in arg.keys():
                reward_scale = arg["reward_scale"]
            self.nupdates_total = total_iters
            print("total num updates", self.nupdates_total)
            self.nupdates = 1
            encoder_model = None #"encoder.h5"
            env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale,flatten_dict_observations = force_flat, rew_noise_std=rew_noise_std, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std, distance_threshold=goal_radius, encoder=encoder_model)
            #env = make_vec_env(env_name, "mujoco", env_config['num_env'] or 1, None, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations, action_noise_std=action_noise_std, obs_noise_std=obs_noise_std)
            if self.alg == "ppo2":
                #env = VecNormalize(env)
                learn_params["nupdates"] = self.nupdates_total
            if 'env' not in learn_params.keys():
                learn_params['env'] = env 
            learn_params["exp_name"] = get_formatted_name(self.params)
             
            load_path = None
            #learn_params["load_file"] = "ppo2ScoopEnv-v0AL83mlpobs_0.0act_0.0rw_0.3rew_noise_std_0.0" 
            self.local_variables = self.alg_module.learn_setup(**learn_params)
            self.mean_reward_over_samples = []
            if env_name in ["FetchPush-v1", "FetchReach-v1"]:
                self.local_variables["success_only"] = True
            else:
                print(env_name)
                self.local_variables["success_only"] = False


        def _train(self):
            self.local_variables['update'] = self.nupdates
            print("nupdates", self.alg, self.nupdates, " of ", self.nupdates_total)
            _, tmp_var, infos = self.alg_module.learn_iter(**self.local_variables)
            #test_success_rate = tmp_var
            if self.env_name == "StirEnv-v0" or self.env_name == "ScoopEnv-v0":
                num_tests = 5
            else:
                num_tests = 17#25
            test_success_rate = self._test(n_test_rollouts=num_tests)['success_rate']
            if np.isnan(test_success_rate):
                import ipdb; ipdb.set_trace()
            self.lock.acquire()
            if test_success_rate > self.best_success_rate:
                self.best_success_rates.append(test_success_rate)
                self.best_success_rate = test_success_rate
                if True or test_success_rate > 0:
                    np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"best_params_so_far.npy", self.sample_config_bound)
                    np.save(SAVE_DIR+"hyperparams/"+get_formatted_name(self.params)+"_best_success_rates.npy", self.best_success_rates)
            self.lock.release()

            self.nupdates += 1 
            self.mean_reward_over_samples.append(test_success_rate)
            return {'done':self.nupdates > self.nupdates_total or test_success_rate > 0.95, 'success_rate':test_success_rate, "infos":infos, 'episode_reward_mean':test_success_rate}

        def _save(self, checkpoint_dir):
            guid = extract_guid(checkpoint_dir, self.exp_start_time)
            np.save(self.save_dir+"/"+guid, self.mean_reward_over_samples) 
            return {}

        def _restore(self, checkpoint_dir):
            pass
        '''
        use current model in evaluation for n_test_rollouts
        '''
        def _test(self, n_test_rollouts=8):
            self.test_local_variables = self.local_variables.copy()
            self.test_local_variables['n_episodes'] = 1
            if self.alg == "cma":
                self.test_local_variables['n_episodes'] = 2
            self.test_local_variables['n_steps_per_iter'] = n_steps_per_iter_per_env[self.env_name]
            self.test_local_variables['n_episodes'] = n_test_rollouts
            success_rate = self.alg_module.learn_test(**self.test_local_variables)
            return {'success_rate':success_rate}
            
    return TrainableClass

def extract_guid(checkpoint_dir, exp_start_time):
    return checkpoint_dir[checkpoint_dir.index(exp_start_time)+len(exp_start_time):checkpoint_dir.index("/checkpoint")]

def pick_params(trial_list, exp_name="noexpname"):
    best_trial = max(trial_list, key=lambda trial: trial.last_result["success_rate"])
    print("Best success rate was", best_trial.last_result["success_rate"])
    np.save("hyperparams/"+exp_name+"best_value.npy", best_trial.last_result["success_rate"])
    threshold = 0.2
    #assert(best_trial.last_result["success_rate"] > threshold)
    print(best_trial.config)
    return best_trial.config

def alg_to_config(alg, env_name=None, force_flat = False):
    num_env = 1
    if env_name == "StirEnv-v0" or env_name == 'ScoopEnv-v0' :
        thresh = 0
    else:
        thresh=-50
    if env_name == "ScoopEnv-v0" and not force_flat:
        network = "mlp_combine"
    else:
        network = "mlp"
    if alg == "ppo2":
        sample_config =  {"lr": tune.sample_from(
            lambda spec: np.random.choice([2,3,4,5])),
                    "vf_coef": tune.sample_from(
                        lambda spec: np.random.uniform(0,1,2,3,4,5)),
                    "ent_coef":tune.sample_from(
                        lambda spec: np.random.uniform(0, 0.01)),
                    "max_grad_norm": tune.sample_from(
                        lambda spec: np.random.uniform(0.495, 0.505)),
                    "cliprange": tune.sample_from(
                        lambda spec: np.random.uniform(0.1, 0.3)),
                    "seed": tune.sample_from(
                        lambda spec: np.random.choice([51])),
                    #"n_steps_per_episode": tune.sample_from(
                    #    lambda spec: np.random.choice([64, 512, 1024, 2048])),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.uniform(0.001, 1)),
                    "lam": tune.sample_from(
                        lambda spec: np.random.uniform(0.90, 0.99))}
        cont_space =  {"lr":[2,5],
                    "vf_coef": [0,5],
                    "ent_coef":[1,5],
                    "max_grad_norm": [0.4,0.6],
                    "cliprange": [0.1,0.3],
                    "seed": [1,50],
                    "reward_scale": [0.01,10],
                    "lam": [0.8,1]}
        fixed_config = {'n_episodes':n_episodes_per_env[env_name],#80 
                        'n_steps_per_episode':n_steps_per_iter_per_env[env_name],
                        'gamma':1.0,
                        'network':network,
                        'log_interval':10,
                        'nminibatches':4,
                        'total_timesteps':1e6,
                        'noptepochs':10,
                        'save_interval':1000}
        env_config = {'num_env':1}
    elif alg == "ddpg":
        sample_config =  {"actor_lr": tune.sample_from(
                        lambda spec: np.random.choice([3,4,5])),
                    "batch_size": tune.sample_from(
                        lambda spec: np.random.choice([2,3,4,5, 6])),
                    "critic_lr": tune.sample_from(
                        lambda spec: np.random.choice([3,4,5])),
                    "seed": tune.sample_from(
                        lambda spec: np.random.choice([1,2,10,15,20])),
                    "noise_level": tune.sample_from(
                        lambda spec: np.random.random([0.05,0.3])),
                    "tau": tune.sample_from(
                        lambda spec: np.random.random([2,3])),
                    "reward_scale": tune.sample_from(
                        lambda spec: np.random.choice([0.01,10])) }
        cont_space =  {"actor_lr": [2,5],
                    "batch_size": [2,11],
                    "critic_lr": [2,5],
                    "seed":[1,20],
                    "tau":[2,3],
                    "noise_level":[0.05,0.3],
                    "reward_scale": [0.01, 10]}
        
        fixed_config = {'network':'mlp', 
                        'n_episodes':n_episodes_per_env[env_name],#200,
                        'n_steps_per_episode':n_steps_per_iter_per_env[env_name],
                        'reward_threshold' : thresh,
                        'critic_l2_reg':0.0,
                        'noise_type':'adaptive-param_0.2',
                        'gamma':1.0}
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
        cont_space =  {"CMA_mu":[3,10],
                    "CMA_cmean": [0.9,1.1],
                    "seed": [1,20],
                    "CMA_rankmu": [0.9, 1.1],
                    "CMA_rankone": [0.9,1.1]}

        fixed_config = {'env_name':env_name,
                        'n_episodes':17,
                        "reward_threshold":thresh,
                        'n_steps_per_episode':n_steps_per_iter_per_env[env_name]}

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

        cont_space =  {"seed": [1,10]}
        fixed_config = {'network':'mlp',
                'policy_save_interval':200,
                'n_cycles':10,#10,
                'n_batches':40,#40,
                'n_test_rollouts':5,#5
                'n_steps_per_episode':50}  # better at 100 for some reason NBATCH_STANDARD}
        fixed_config['n_episodes'] = fixed_config['n_cycles']
        env_config = {'num_env':1
        }
    else:
        raise ValueError("Algorithm not supported")
    return sample_config, fixed_config, env_config, cont_space

def run_async_hyperband(smoke_test = False, expname = "test", obs_noise_std=0, action_noise_std=0, params={}):
    if smoke_test:
        grace_period = 1
        max_t = 5
        num_samples = 3
        num_cpu = 1
        num_gpu = 0
        num_total_cpu = 1
        NBATCH_STANDARD = 10
    else:
        grace_period = 5
        max_t = 1e6//40 #this doesn't actually mean anything. Trainable takes care of killing processes when they go on for too long
        num_samples = 20 #30
        num_cpu = 1#10
        num_total_cpu = 10
        num_gpu = 0
    space = alg_to_config(params['alg'],params['env_name'])[3]
    bayes_opt = BayesOptSearch(
        space,
        max_concurrent=num_total_cpu,
        #metric="mean_loss",
        #mode="min",
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        })

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
                "checkpoint_freq":1,
                "resources_per_trial": {
                    "cpu": num_cpu,
                    "gpu": num_gpu,
                },
                "config": {'iterations':1e6//40}#alg_to_config(params['alg'], params['env_name'])[0], #just the tuneable ones
            }
        },
    scheduler=ahb, search_alg = bayes_opt, queue_trials=False, verbose=0)
"""
Precondition: hyperparameter optimization happened
"""
def run_alg(params, iters=2,hyperparam_file = None, LLcluster=True, exp_number=None):
    exp_name = get_formatted_name(params)
    if hyperparam_file is None:
        hyperparam_file = "hyperparams/"+exp_name+"best_hyperparams.npy"
    hyperparams = np.load(hyperparam_file, allow_pickle=True).all()

    args = {"sample_config_best":hyperparams}
    #overwrite the old ones
    args["obs_noise_std"] = params["obs_noise_std"]
    args["rew_noise_std"] = params["rew_noise_std"]
    args["action_noise_std"] = params["action_noise_std"]
    args["goal_radius"] = params["goal_radius"]
    trainable = make_class(params)(args)
    #print("made class")
    info_data = {}
    test_success_rates = []
    SAVE_INTERVAL=2
    if LLcluster and exp_number is None:
        exp_number = os.environ["SLURM_ARRAY_TASK_ID"]
    for i in range(int(iters)):
        print ("on iter #", i)
        test_res = trainable._train()
        test_success_rates.append(test_res['success_rate'])
        infos = test_res['infos']
        for info in infos.keys():
            if info in info_data.keys():
               info_data[info].append(infos[info])
            else:
               info_data[info] = [infos[info]]

        if i % SAVE_INTERVAL == 0:
            print(" last success rate", test_res["success_rate"])
            for info in info_data.keys():
                np.save("run_results/"+exp_name+info+"_"+str(exp_number)+".npy", info_data[info])
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


if __name__=="__main__":
    #res = tune.run_experiments(experiments=experiment_spec)
    print("running trainable.py")
    import sys
    if "manual" in sys.argv:
        params = {'env_name':sys.argv[1], 'exp_name':sys.argv[2], 'obs_noise_std':float(sys.argv[3]), 'action_noise_std':float(sys.argv[4]), 'goal_radius':float(sys.argv[5]),'alg':sys.argv[6]}
        if len(sys.argv) > 7:
            params['rew_noise_std'] = float(sys.argv[9])
        params['rew_noise_std'] = 0.0 #this wasn't a useful parameter but you can bring it back if you really want
        #params['lr'] = 1e-3
        print(params)
        n_episodes=alg_to_config(params["alg"], params['env_name'])[1]['n_episodes']
        num_iters = train_alg_to_iters[params["alg"]]//n_episodes
        run_alg(params, iters=int(num_iters), hyperparam_file = sys.argv[7], LLcluster=False, exp_number=int(sys.argv[8]))

    else:
        exp_name = sys.argv[1] 
        params= np.load("params/"+exp_name+"_params.npy").all()
        alg = params['alg']
        params['seed']=1
        n_episodes=alg_to_config(params["alg"], params['env_name'])[1]['n_episodes']
        num_iters = train_alg_to_iters[alg]//n_episodes
        if len(sys.argv) > 2:
            run_alg(params, iters=num_iters, exp_number = int(sys.argv[2]))
        else:
            run_alg(params, iters=num_iters)
    #hyperparams_file = np.load(exp_name+"best_params.npy")
    import ray
    ray.shutdown()

