import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

from baselines import logger
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          exp_name="test",
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):
    local_variables = learn_setup(network, env, seed=seed, total_timesteps=total_timesteps, nb_epochs=nb_epochs, nb_epoch_cycles=nb_epoch_cycles, nb_rollout_steps=100,reward_scale=1.0,render=render,render_eval=render_eval,noise_type='adaptive-param_0.2',normalize_returns=False,normalize_observations=True,critic_l2_reg=1e-2,
          exp_name=exp_name,
          eval_env=eval_env,
          actor_lr=actor_lr,
          critic_lr=critic_lr,
          popart=popart,
          gamma=gamma,
          clip_norm=clip_norm,
          nb_train_steps=nb_train_steps, # per epoch cycle and MPI worker,
          nb_eval_steps=nb_eval_steps,
          batch_size=batch_size, # per MPI worker
          tau=tau,
          param_noise_adaption_interval=param_noise_adaption_interval,
          **network_kwargs)
    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500
    for epoch in range(nb_epochs):
        _, success_rate, _ = learn_iter(**local_variables)
    return local_variables["agent"]

def learn_setup(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=None,
          nb_rollout_steps=100,
          n_episodes=None,
          n_steps_per_episode=None,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          exp_name="test",
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    set_global_seeds(seed)
    if nb_epoch_cycles is None:
        nb_epoch_cycles = n_episodes 
        nb_rollout_steps = n_steps_per_episode
    else:
        input("Not using automated interface? ")

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    nb_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()
    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()
    agent.reset()
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]
    episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar
    epoch = 0
    start_time = time.time()
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    local_variables = {
        "epoch_episode_rewards":epoch_episode_rewards,
        "epoch_episode_steps":epoch_episode_steps,
        "batch_size":batch_size,
        "eval_env":eval_env,
        "epoch_actions" : epoch_actions,
        "nb_train_steps":nb_train_steps,
        "epoch_qs" : epoch_qs,
        "start_time":start_time,
        "epoch_episodes" : [epoch_episodes],
        "nb_epoch_cycles" : nb_epoch_cycles,
        "nb_rollout_steps" : nb_rollout_steps,
        "agent" : agent,
        "memory" : memory,
        "max_action":max_action,
        "env" : env,
        "nenvs" : nenvs,
        "obs" : [obs], #Forgive me 6.031
        "t":[t],
        "episode_reward":episode_reward,
        "episode_rewards_history":episode_rewards_history,
        "episode_step":episode_step,
        "episodes":[episodes],
        "rank" : rank,
        "param_noise_adaption_interval":param_noise_adaption_interval,
        "render" : render}
    return local_variables

def learn_test(epoch_episode_rewards=[],
               epoch_episode_steps=[],
               episode_rewards_history=None,
               update = None,
               epoch_actions = [],
               param_noise_adaption_interval=None,
               eval_env=None,
               start_time=None,
               batch_size=None,
               memory=None,
               epoch_qs = [],
               nb_train_steps = 0,
               max_action=None,
               mean_rewards = [],
               epoch_episodes = [0],
               nb_epoch_cycles = None,
               env=None,
               nb_rollout_steps = None,
               agent = None,
               t = None,
               n_episodes=None,
               n_steps_per_iter=None,
               episode_reward = None,
               episode_step = None,
               episodes=None,
               nenvs = None,
               obs = None,
               rank = None,
               render = None):

    # Evaluate.
    
    eval_obs = env.reset()
    for i in range(n_steps_per_iter):
        eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
        eval_obs, eval_r, eval_done, eval_info = env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
    if isinstance(eval_info, list):
        eval_info = eval_info[0]
    return eval_info['is_success']

def learn_test(**kwargs):
    return learn_iter(**kwargs, test=True)[1]


def learn_iter(epoch_episode_rewards=[], 
               epoch_episode_steps=[],
               episode_rewards_history=None,
               update = None,
               epoch_actions = [],
               param_noise_adaption_interval=None,
               eval_env=None,
               start_time=None,
               batch_size=None,
               memory=None,
               epoch_qs = [],
               nb_train_steps = 0,
               max_action=None,
               mean_rewards = [],
               epoch_episodes = [0],
               nb_epoch_cycles = None,
               n_episodes=None,
               env=None,
               nb_rollout_steps = None,
               n_steps_per_iter=None,
               agent = None,
               t = None,
               test=False,
               episode_reward = None,
               episode_step = None,
               episodes=None,
               nenvs = None,
               obs = None,
               rank = None,
               render = None):
    successes = []
    if n_steps_per_iter is not None:
        nb_rollout_steps = n_steps_per_iter
    if n_episodes is not None:
        nb_epoch_cycles = n_episodes
    for cycle in range(nb_epoch_cycles):
        # Perform rollouts.
        if nenvs > 1:
            # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
            # of the environments, so resetting here instead
            agent.reset()
        for t_rollout in range(nb_rollout_steps):
            # Predict next action.
            action, q, _, _ = agent.step(obs[0], apply_noise=not test, compute_Q=True)

            # Execute next action.
            if rank == 0 and render:
                env.render()

            # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
            new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            # note these outputs are batched from vecenv

            t[0] += 1
            if rank == 0 and render:
                env.render()
            episode_reward += r
            episode_step += 1

            # Book-keeping.
            epoch_actions.append(action)
            epoch_qs.append(q)
            agent.store_transition(obs[0], action, r, new_obs, done) #the batched data will be unrolled in memory.py's append.

            obs[0] = new_obs

            for d in range(len(done)):
                if done[d]:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward[d])
                    episode_rewards_history.append(episode_reward[d])
                    epoch_episode_steps.append(episode_step[d])
                    #successes.append(int(episode_reward[d] >= 0))
                    successes.append(episode_reward[d])
                    episode_reward[d] = 0.
                    episode_step[d] = 0
                    epoch_episodes[0] += 1
                    episodes[0] += 1
                    if nenvs == 1:
                        agent.reset()

        if not test:
            #train
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()


    if MPI is not None:
        mpi_size = MPI.COMM_WORLD.Get_size()
    else:
        mpi_size = 1
    infos = {}
    if not test:
        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration

        combined_stats['rollout/success_rate'] = np.mean(successes)
        print("successes", np.mean(successes))
        combined_stats['total/steps_per_second'] = float(t[0]) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes[0]
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        infos["action_mean"] = combined_stats["rollout/actions_mean"]
        infos["loss_actor"] = combined_stats["train/loss_actor"]
        infos["loss_critic"] = combined_stats["train/loss_critic"]
        infos["param_noise_distance"] = combined_stats["train/param_noise_distance"]
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/steps'] = t[0]

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        #if rank == 0:
        #    logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)
    return None, np.mean(successes), infos

