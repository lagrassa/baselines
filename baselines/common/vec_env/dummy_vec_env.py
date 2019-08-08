import numpy as np
from collections import deque
from keras.models import load_model
from gym import spaces
from . import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns, action_noise_std=0, obs_noise_std=0, distance_threshold=0.05, rew_noise_std=0.0, encoder=None):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        self.action_noise_std = action_noise_std
        self.obs_noise_std = obs_noise_std
        self.rew_noise_std = rew_noise_std
        env = self.envs[0]
        if encoder is not None:
            encoder_dict = {}
            if 'im' in encoder.keys():
                encoder_dict['im'] = [load_model(model_fn) for model_fn in encoder["im"]]
            if 'forces' in encoder.keys():
                encoder_dict['forces'] = encoder["forces"]
                self.obs_buffer = deque(maxlen=encoder["forces"].flow[1].output_dim//encoder["forces"].flow[1].input_dim)
            else:
                self.obs_buffer = deque(maxlen=1)
                try:
                   self.obs_space = self.envs[0].env.env.env.observation_space
                except  AttributeError: #sorry, I know thisis bad
                   self.obs_space = self.envs[0].env.env.observation_space

            print("Loading autoencoder")
            
            try:
                self.envs[0].env.env.gen_obs_space(encoder_dict)
                self.obs_space = self.envs[0].env.env.observation_space
            except  AttributeError: #sorry, I know thisis bad
                try:
                    self.envs[0].env.env.env.gen_obs_space(encoder_dict)
                    self.obs_space = self.envs[0].env.env.env.observation_space
                except  AttributeError: #sorry, I know thisis bad
                    self.envs[0].env.env.gen_obs_space(encoder_dict)
                    self.obs_space = self.envs[0].env.env.observation_space


            self.encoder=encoder_dict
        else:
            self.encoder=None
            self.obs_buffer = deque(maxlen=1)
            self.obs_space = env.env.observation_space
   
        self.distance_threshold = distance_threshold
        VecEnv.__init__(self, len(env_fns), self.obs_space, env.action_space)
        self.keys, shapes, dtypes = obs_space_info(self.obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.specs = [e.spec for e in self.envs]

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            if isinstance(self.envs[e].action_space, spaces.Discrete):
                action = int(action)
            if self.action_noise_std > 0:
                noisy_action = action + np.random.normal(np.zeros(action.shape),self.action_noise_std)
            else:
                noisy_action = action
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(noisy_action)
            self.buf_rews[e] = self.buf_rews[e] + np.random.normal(0,self.rew_noise_std)
            if isinstance(obs, dict):
                if 'observation' in obs.keys():
                    obs['observation'] = obs['observation'] + obs['observation']*np.random.normal(np.zeros(obs['observation'].shape),self.obs_noise_std)
                else:
                    for subspace in obs.keys():
                        obs[subspace] = obs[subspace] + obs[subspace]*np.random.normal(np.zeros(obs[subspace].shape),self.obs_noise_std)
            else:
                obs = obs + np.random.normal(np.zeros(obs.shape),self.obs_noise_std)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            if isinstance(obs, dict) and "forces" in obs.keys():
                if len(self.obs_buffer) < self.obs_buffer.maxlen:
                    for _ in range(self.obs_buffer.maxlen-len(self.obs_buffer)):
                        self.obs_buffer.append(obs["forces"])
                else:
                    self.obs_buffer.append(obs["forces"])
            if self.encoder is not None:
                if "im" in self.encoder.keys():
                    obs['imside'] = self.encoder['im'][0].predict( obs['imside'].reshape(1,*obs['imside'].shape))[0]
                    obs['imtop'] = self.encoder['im'][1].predict( obs['imtop'].reshape(1,*obs['imtop'].shape))[0]
                if "forces" in self.encoder.keys():
                    try:
                        obs['forces'] = self.encoder['forces'](np.vstack(self.obs_buffer))
                    except:
                        pass
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            if isinstance(obs,dict) and  len(self.obs_buffer) < self.obs_buffer.maxlen and "forces" in obs.keys():
                for _ in range(self.obs_buffer.maxlen-len(self.obs_buffer)):
                    self.obs_buffer.append(obs["forces"])
            if self.encoder is not None:
                if "im" in self.encoder.keys():
                    obs['imside'] = self.encoder['im'][0].predict( obs['imside'].reshape(1,*obs['imside'].shape))[0]
                    obs['imtop'] = self.encoder['im'][1].predict( obs['imtop'].reshape(1,*obs['imtop'].shape))[0]
                if "forces" in self.encoder.keys():
                    try:
                        obs['forces'] = self.encoder['forces'](np.vstack(self.obs_buffer))
                    except:
                        pass
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

def unwrap_env(unwrapped_env, timeout=10):
    for i in range(timeout):
        return
