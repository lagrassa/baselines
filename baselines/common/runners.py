import numpy as np
from gym.spaces import Dict
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = {}
        if isinstance(env.observation_space, Dict):
            for obs_subspace in env.observation_space.spaces.keys():
                self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.spaces[obs_subspace].shape
                self.obs[obs_subspace] = np.zeros((nenv,) + env.observation_space.spaces[obs_subspace].shape, dtype=env.observation_space.spaces[obs_subspace].dtype.name)
            obs_parts = env.reset()
            for obs_subspace, obs_part in zip(env.observation_space.spaces.keys(), obs_parts):
                self.obs[obs_subspace][:] = obs_parts[obs_part]

        else:
            self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
            self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

