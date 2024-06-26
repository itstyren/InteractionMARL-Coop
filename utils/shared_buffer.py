import numpy as np
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
import torch
from typing import NamedTuple
from typing import Dict,Union


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class SharedReplayBuffer(object):
    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        self.num_agents = args.env_dim**2
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.algo = args.algorithm_name
        self.device = device
        self.obs_shape = get_obs_shape(obs_space)
        self.act_shape = get_action_dim(action_space)
        self.action_space = action_space
        self.obs = np.array(
            [
                [
                    [
                        {key: np.zeros(shape) for key, shape in self.obs_shape.items()}
                        for _ in range(self.num_agents)
                    ]
                    for _ in range(self.n_rollout_threads)
                ]
                for _ in range(self.episode_length + 1)
            ]
        )
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents),
            dtype=np.int16,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32
        )
        self.termination = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents), dtype=bool
        )
        self.step = 0
        self.full = False

    def insert(self, obs, rewards, termination, actions):
        """
        Insert data into the buffer
        """
        self.obs[self.step + 1] = obs.copy()
        self.rewards[self.step] = rewards.copy()
        self.actions[self.step] = actions.copy()
        self.termination[self.step] = [
            [d] * len(actions[0]) for d in termination.copy()
        ]
        self.step += 1
        if self.step == self.episode_length:
            self.full = True
            self.step = 0

    def after_update(self):
        """
        Copy last timestep data to first index. Called after update to model.
        """
        self.obs[0] = self.obs[-1].copy() 


    def sample(self, batch_size: int):
        if self.full:
            batch_inds = (
                np.random.randint(1, self.episode_length, size=batch_size) + self.step
            ) % self.episode_length
        else:
            batch_inds = np.random.randint(0, self.step, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray):
        # Sample randomly the env idx
        env_indices = np.random.randint(
            0, high=self.n_rollout_threads, size=(len(batch_inds),)
        )

        data = (
            np.concatenate(self.obs[batch_inds, env_indices, :]),
            np.concatenate(self.actions[batch_inds, env_indices, :]),
            np.concatenate(self.obs[batch_inds + 1, env_indices, :]),
            np.concatenate(self.termination[batch_inds, env_indices, :]).astype(int),
            np.concatenate(self.rewards[batch_inds, env_indices, :]),     
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(
        self, array: Union[np.ndarray, Dict[str, np.ndarray]], copy: bool = True
    ) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        # deal with obvservation
        if isinstance(array[0], dict):
            for _, obs in enumerate(array):
                array[_] = {
                    key: torch.as_tensor(_obs, device=self.device)
                    for (key, _obs) in obs.items()
                }
            return array
        else:
            return torch.tensor(array, device=self.device)
