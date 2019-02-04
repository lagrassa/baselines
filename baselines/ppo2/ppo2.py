import os
import ipdb
import inspect
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
from cup_skills.local_setup import path as PATH
PATH = PATH+"data/"

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=32, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, exp_name="test", **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    kwargs = {} #I'm sorry 6.031
    frame = inspect.currentframe()
    import os
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        exp_name = exp_name + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    separate_list = ["network_kwargs"]
    args, _, _, values = inspect.getargvalues(frame)
    for x, p in inspect.signature(learn).parameters.items():
        if x not in separate_list:
            kwargs[x] = values[x]
    local_variables = learn_setup(**kwargs, **network_kwargs )
        #vf_coef=vf_coef, max_grad_norm=max_grad_norm, gamma=gamma, lam=lam,
        #log_interval=log_interval, nminibatches=nminibatches, noptepochs=noptepochs, cliprange=cliprange,
        #save_interval=save_interval, load_path=load_path, model_fn=model_fn, **network_kwargs):
        
    nenvs = env.num_envs
    nbatch = nenvs * nsteps
    nupdates = total_timesteps//nbatch
    mean_rewards = []
    success_rates = []
    for update in range(1, nupdates+1):
        local_variables['update'] = update
        model, success_rate, mean_reward = learn_iter(**local_variables)
        mean_rewards.append(mean_reward)
        success_rates.append(success_rate)
        np.save(PATH+"mean_rewards_"+exp_name+".npy", mean_rewards)
        np.save(PATH+"success_rates_"+exp_name+".npy", success_rates)

    return model

def learn_setup(*, network=None, env=None, total_timesteps=None, eval_env = None, seed=None, nsteps=64, ent_coef=0.0, lr=3e-4, reward_scale = None, exp_name=None,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, **network_kwargs):

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    eval_runner = None
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    local_variables = {
                       'nbatch':nbatch,
                       'nminibatches':nminibatches,
                       'nbatch_train':nbatch_train,
                       'save_interval':save_interval,
                       'model':model,
                       'runner':runner,
                       'lr':lr,
                       'nsteps':nsteps,
                       'nenvs':nenvs,
                       'log_interval':log_interval,
                       'cliprange':cliprange,
                       'eval_runner' : eval_runner,
                       'eval_env' : eval_env,
                       'epinfobuf':epinfobuf,
                       'noptepochs':noptepochs,
                       'tfirststart':tfirststart,
                       'nupdates':nupdates          
                      }
    return local_variables

def learn_iter(nbatch=None, nminibatches=None, nbatch_train=None, model=None, runner=None, epinfobuf=None, tfirststart=None, nupdates=None, update=None, lr=None, eval_runner=None, cliprange=None, eval_env=None, noptepochs=None, log_interval=None, nsteps=None, nenvs=None, save_interval=None, exp_name=None):

    assert nbatch % nminibatches == 0
    # Start timer
    tstart = time.time()
    frac = 1.0 - (update - 1.0) / nupdates
    # Calculate the learning rate
    lrnow = lr(frac)
    # Calculate the cliprange
    cliprangenow = cliprange(frac)
    # Get minibatch
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
    if eval_env is not None:
        eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

    epinfobuf.extend(epinfos)
    if eval_env is not None:
        eval_epinfobuf.extend(eval_epinfos)

    # Here what we're going to do is for each minibatch calculate the loss and append it.
    mblossvals = []
    if states is None: # nonrecurrent version
        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
    else: # recurrent version
        assert nenvs % nminibatches == 0
        envsperbatch = nenvs // nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        envsperbatch = nbatch_train // nsteps
        for _ in range(noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mbstates = states[mbenvinds]
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

    # Feedforward --> get losses --> update
    lossvals = np.mean(mblossvals, axis=0)
    # End timer
    tnow = time.time()
    # Calculate the fps (frame per second)
    fps = int(nbatch / (tnow - tstart))
    try:
        success_rate = safemean([epinfo['is_success'] for epinfo in epinfobuf])
        mean_reward = safemean([epinfo['r'] for epinfo in epinfobuf])
    except:
        import ipdb; ipdb.set_trace()
    if update % log_interval == 0 or update == 1:
        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        ev = explained_variance(values, returns)
        logger.logkv("serial_timesteps", update*nsteps)
        logger.logkv("nupdates", update)
        logger.logkv("total_timesteps", update*nbatch)
        logger.logkv("fps", fps)
        logger.logkv("explained_variance", float(ev))
        logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        logger.logkv('success_rate', success_rate)
        
        if eval_env is not None:
            logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
            logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
        logger.logkv('time_elapsed', tnow - tfirststart)
        for (lossval, lossname) in zip(lossvals, model.loss_names):
            logger.logkv(lossname, lossval)
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            logger.dumpkvs()
    if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i'%update)
        print('Saving to', savepath)
        model.save(savepath)
    return model, success_rate, mean_reward
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
    import ipdb; ipdb.set_trace()


