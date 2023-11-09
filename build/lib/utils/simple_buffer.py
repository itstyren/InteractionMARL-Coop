
import numpy as np
from .util import get_shape_from_obs_space, get_shape_from_act_space


class Buffer(object):
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads

        self.obs_shape = get_shape_from_obs_space(obs_space)

        if type(self.obs_shape[-1]) == list:
            self.obs_shape = self.obs_shape[:1]
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *self.obs_shape), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.step = 0

    def insert(self, obs,reward):
        self.obs[self.step + 1] = obs.copy()
        self.rewards[self.step] = reward
        self.step = (self.step + 1) % self.episode_length

    def after_update(self,obs):
        self.obs[0] = obs.copy()