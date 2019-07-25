import os
from itertools import chain
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


def constfn(val):
    def f(_):
        return val
    return f

def learn_test(nbatch=None, nminibatches=None, nbatch_train=None, model=None, runner=None, epinfobuf=None, tfirststart=None, nupdates=None, update=None, lr=None, eval_runner=None, cliprange=None, eval_env=None, noptepochs=None, log_interval=None, nsteps=None, nenvs=None, save_interval=None, exp_name=None, n_steps_per_iter=None, n_episodes=None, success_only = True):
    assert(nsteps > n_episodes)
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
    if success_only:
        success_rate = safemean([epinfo['is_success'] for epinfo in epinfos])
    else:
        success_rate =  safemean(returns)
    return success_rate


def learn_iter(nbatch=None, nminibatches=None,  nbatch_train=None, model=None, runner=None, epinfobuf=None, tfirststart=None, nupdates=None, update=None, lr=None, eval_runner=None, cliprange=None, eval_env=None, noptepochs=None, log_interval=None, nsteps=None, nenvs=None, save_interval=None, exp_name=None, n_steps_per_iter=None, n_episodes=None, success_only=True):

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
    #variables = model.act_model.pd.neglogp(actions)
    #feed_dict = {}
    #sess.run(variables, feed_dict)

    epinfobuf.extend(epinfos)
    if eval_env is not None:
        eval_epinfobuf.extend(eval_epinfos)

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
                if isinstance(obs, list):
                    slices = (arr[mbinds] for arr in (returns, masks, actions, values, neglogpacs))
                    obs_slices = ()
                    for subspace_i in range(len(obs[0])):
                        subspace_obs = np.zeros(((len(mbinds),)+obs[0][subspace_i].shape))
                        for ind in mbinds:
                            subspace_obs = subspace_obs + (obs[ind][subspace_i],)
                        subspace_obs = np.array(subspace_obs)
                        obs_slices = obs_slices + (subspace_obs,)
                    obs_slices = (obs_slices,)
                    slices = list(chain(obs_slices, slices))

                else:
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
    #30th batch
    #one_slice = np.array(mblossvals[-1][5:])[:,30]
    #need to edit below to fix that, but w.e. for now 
    #print(mblossvals[-1][5])
    lossvals = np.mean(mblossvals, axis=0)
    # End timer
    tnow = time.time()
    # Calculate the fps (frame per second)
    fps = int(nbatch / (tnow - tstart))
    mean_reward = safemean([epinfo['r'] for epinfo in epinfobuf])
    if success_only:
        success_rate = safemean([epinfo['is_success'] for epinfo in epinfos])
    else:
        success_rate =  safemean(returns)

    #print("mean reward", mean_reward)
    #import ipdb; ipdb.set_trace()
    infos = {}
    ev = explained_variance(values, returns)
    infos["exp_variance"] = float(ev)
    for (lossval, lossname) in zip(lossvals, model.loss_names):
        infos[lossname] = lossval
    if False and (update % log_interval == 0 or update == 1):
        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        logger.logkv("serial_timesteps", update*nsteps)
        logger.logkv("nupdates", update)
        #logger.logkv("total_timesteps", update*nbatch)
        #logger.logkv("fps", fps)
        #logger.logkv("explained_variance", float(ev))
        #logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        #logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        #logger.logkv('success_rate', success_rate)

        if eval_env is not None:
            logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
            logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
        #logger.logkv('time_elapsed', tnow - tfirststart)
        #for (lossval, lossname) in zip(lossvals, model.loss_names):
        #    logger.logkv(lossname, lossval)
        #if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        #    logger.dumpkvs()
    model.save("models/"+exp_name)
    """
    if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i'%update)
        print('Saving to', savepath)
        model.save(savepath)
    """
    print('success rate', success_rate)
    return model, success_rate, infos


def learn_setup(*, network=None, env=None, total_timesteps=None, eval_env = None, seed=None, nsteps=None, ent_coef=0.0, lr=3e-4, reward_scale = None, exp_name=None,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            n_steps_per_episode=None,
            n_episodes = 1,
            nupdates=1,
            batch_size=None,
            save_interval=0, load_path=None, model_fn=None, **network_kwargs):

    lr = 10**(-1*lr)
    vf_coef = 10**(-1*vf_coef)
    seed = int(seed)
    ent_coef = 10**(-1*ent_coef)

    if network == "lstm":
        nminibatches=1
    if nsteps is None:
        nsteps = n_steps_per_episode*n_episodes

    #set_global_seeds(seed)
    #np.random.seed(None)
    #np.random.seed(seed)
    if nsteps is None:
        nsteps = n_steps_per_episode
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    policy = build_policy(env, network, **network_kwargs)
# Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    nupdates = total_timesteps//nbatch


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

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    local_variables = {
                       'nbatch':nbatch,
                       'nminibatches':nminibatches,
                       'nbatch_train':nbatch_train,
                       'save_interval':save_interval,
                       'model':model,
                       'runner':runner,
                       'lr':lr,
                       'exp_name':exp_name,
                       'nsteps':nsteps,
                       'nenvs':nenvs,
                       'log_interval':log_interval,
                       'cliprange':cliprange,
                       'eval_runner' : eval_runner,
                       'n_episodes':n_episodes,
                       'eval_env' : eval_env,
                       'epinfobuf':epinfobuf,
                       'noptepochs':noptepochs,
                       'tfirststart':tfirststart,
                       'nupdates':nupdates
                      }
    return local_variables


def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=100, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=500, load_path=None, model_fn=None, **network_kwargs):
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

    assert(seed is None)
    #set_global_seeds(seed)

    kwargs = {} #I'm sorry 6.031
    frame = inspect.currentframe()
    #nminibatches = 1
    import os
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        exp_name = exp_name + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    separate_list = ["network_kwargs"]
    args, _, _, values = inspect.getargvalues(frame)
    for x, p in inspect.signature(learn).parameters.items():
        if x not in separate_list:
            kwargs[x] = values[x]

    local_variables = learn_setup(**kwargs, **network_kwargs )
    nbatch=local_variables["nbatch"]
    nminibatches=local_variables["nminibatches"]
    nbatch_train=local_variables["nbatch_train"]
    save_interval=local_variables["save_interval"]
    model=local_variables["model"]
    runner=local_variables["runner"]
    lr=local_variables["lr"]
    nsteps=local_variables["nsteps"]
    nenvs=local_variables["nenvs"]
    log_interval=local_variables["log_interval"]
    cliprange=local_variables["cliprange"]
    eval_runner =local_variables["eval_runner"]
    n_episodes=local_variables["n_episodes"]
    eval_env =local_variables["eval_env"]
    epinfobuf=local_variables["epinfobuf"]
    noptepochs=local_variables["noptepochs"]
    tfirststart=local_variables["tfirststart"]
    nupdates=local_variables["nupdates"]
    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    eprewmeans = []
    for update in range(1, nupdates+1):
        local_variables['update'] = update
        learn_iter(**local_variables)
        local_variables['model'] = model
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



