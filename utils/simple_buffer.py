
import numpy as np
from .util import get_shape_from_obs_space, get_shape_from_act_space
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

class Buffer(object):
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads

        self.obs_shape = get_obs_shape(obs_space)

        self.obs = np.array(
            [
                [
                        {key: np.zeros(shape) for key, shape in self.obs_shape.items()}

                    for _ in range(self.n_rollout_threads)
                ]
                for _ in range(self.episode_length + 1)
            ]
        )
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.step = 0

    def insert(self, obs,reward):
        self.obs[self.step + 1] = obs.copy()
        self.rewards[self.step] = reward
        # print(reward)
        self.step = (self.step + 1) % self.episode_length

    def after_update(self,obs):
        self.obs[0] = obs.copy()