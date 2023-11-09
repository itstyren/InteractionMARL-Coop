import numpy as np
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
import torch
from typing import NamedTuple
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from utils.segment_tree import SumSegmentTree, MinSegmentTree
import random
from abc import ABC, abstractmethod
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param obs_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    obs_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        episode_length: int = 1,
        normalize_pattern: str = "none",
    ):
        super().__init__()
        self.obs_shape = get_obs_shape(obs_space)
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.episode_length = episode_length
        self.normalize_pattern = normalize_pattern

        self.device = device

        self.step = 0
        self.full = False

        self.obs = np.array(
            [
                [
                    {key: np.zeros(shape) for key, shape in self.obs_shape.items()}
                    for _ in range(self.n_envs)
                ]
                for _ in range(self.buffer_size)
            ]
        )
        self.next_obs = np.array(
            [
                [
                    {key: np.zeros(shape) for key, shape in self.obs_shape.items()}
                    for _ in range(self.n_envs)
                ]
                for _ in range(self.buffer_size)
            ]
        )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, 1),
            dtype=np.int32,
        )
        self.interaction = np.zeros(
            (self.buffer_size, self.n_envs, 1),
            dtype=np.int32,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.termination = np.zeros((self.buffer_size, self.n_envs, 1), dtype=bool)
        self.truncation = np.zeros((self.buffer_size, self.n_envs, 1), dtype=bool)
        self.norm_rewards = np.zeros((self.buffer_size, self.n_envs, 1), dtype=bool)

    def current_buffer_size(self):
        """
        return how many buffer have been add to buffer
        the maximum is exactly the batch_size
        """
        if self.full:
            _indx = self.buffer_size
        else:
            _indx = self.step
        return _indx

    def normalized_rewards(self):
        """
        Normalize reward by mean and std
        """
        _indx = self.current_buffer_size()
        # print(_indx)

        rewards_copy = np.concatenate(self.rewards[:_indx]).copy()
        mean_rewards = np.nanmean(rewards_copy)
        std_rewards = np.nanstd(rewards_copy)
        rewards = (np.concatenate(self.rewards[:_indx]) - mean_rewards) / (
            std_rewards + 1e-5
        )
        # self.rewards[:_indx] = rewards.reshape(self.rewards[:_indx].shape).copy()
        self.norm_rewards = rewards.reshape(self.rewards[:_indx].shape).copy()
        # print(len(self.norm_rewards))
        # print('===')
        # print(mean_rewards)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.step = 0
        self.full = False

    def insert(
        self, obs, next_obs, rewards, termination, truncation, actions, interactions
    ):
        """
        Insert data into the buffer
        """
        # print('===========================',self.step)
        # print(self.obs[self.step + 1])
        # print(self.rewards[self.step])
        self.obs[self.step] = obs.copy()
        self.next_obs[self.step] = next_obs.copy()
        self.rewards[self.step] = rewards.copy().reshape(-1, 1)
        self.actions[self.step] = actions.copy().reshape(-1, 1)
        self.termination[self.step] = termination.copy().reshape(-1, 1)
        self.truncation[self.step] = truncation.copy().reshape(-1, 1)
        if len(interactions) > 0:
            self.interaction[self.step] = interactions.copy().reshape(-1, 1)
        self.step += 1

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences uniformly
        :return: sampled Replay Buffer
        """
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.step
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.step, size=batch_size)

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        return self._get_samples(batch_inds, env_indices)

    def _get_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, action_flag
    ):
        # print("env_indices", env_indices)
        # print("batch_inds", batch_inds)

        # print('current obs',self.obs[batch_inds, env_indices, :])
        # print('next obs',self.obs[batch_inds+1, env_indices, :])
        # print('next',self.rewards[batch_inds, env_indices, :])
        # print('action:',self.actions[batch_inds, env_indices, :])
        if self.normalize_pattern == "all":
            self.normalized_rewards()
            sample_rewards = np.concatenate(self.norm_rewards[batch_inds, env_indices])
        else:
            sample_rewards = np.concatenate(self.rewards[batch_inds, env_indices])
            if self.normalize_pattern == "sample":
                rewards_copy = sample_rewards.copy()
                mean_rewards = np.nanmean(rewards_copy)
                std_rewards = np.nanstd(rewards_copy)
                sample_rewards = (sample_rewards - mean_rewards) / (std_rewards + 1e-5)

        data = (
            self.obs[batch_inds, env_indices],
            np.concatenate(self.actions[batch_inds, env_indices])
            if action_flag == 0
            else np.concatenate(self.interaction[batch_inds, env_indices]),
            self.next_obs[batch_inds, env_indices],  # next obs
            np.concatenate(self.termination[batch_inds, env_indices]).astype(int),
            sample_rewards,
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


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class SeparatedReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        # Adjust buffer size
        self.buffer_size = max(args.buffer_size // args.n_rollout_threads, 1)

        super().__init__(
            self.buffer_size,
            obs_space,
            action_space,
            device,
            args.n_rollout_threads,
            args.episode_length,
            args.normalize_pattern,
        )

    def insert(
        self, obs, next_obs, rewards, termination, truncation, actions, interaction
    ):
        """
        Insert data into the buffer
        """
        super().insert(
            obs, next_obs, rewards, termination, truncation, actions, interaction
        )
        if self.step == self.buffer_size:
            self.full = True
            self.step = 0


class SeparatedRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        super().__init__(
            args.episode_length,
            obs_space,
            action_space,
            device,
            args.n_rollout_threads,
            args.normalize_pattern,
        )

    def insert(
        self, obs, next_obs, rewards, termination, truncation, actions, interaction
    ):
        """
        Insert data into the buffer
        """
        super().insert(
            obs, next_obs, rewards, termination, truncation, actions, interaction
        )

        if self.step == self.buffer_size:
            self.full = True

    def after_update(self):
        """
        Copy last timestep data to first index. Called after update to model.
        """
        self.obs[0] = self.obs[-1].copy()
        self.reset()


class PrioritizedReplayBuffer(SeparatedReplayBuffer):
    """
    Create Prioritized Replay buffer

    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, action_space, device)

        self.buffer_long = self.buffer_size * self.n_envs

        # how much prioritization is used  (0 - no prioritization, 1 - full prioritization)
        assert args.prioritized_replay_alpha >= 0
        self._alpha = args.prioritized_replay_alpha

        # Total size of the items array
        it_capacity = 1
        while it_capacity < self.buffer_long:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def insert(
        self, obs, next_obs, rewards, termination, truncation, actions, interaction
    ):
        """
        set priority of transition at current step
        """
        for _ in range(self.n_envs):
            self._it_sum[self.step + _ * self.buffer_size] = (
                self._max_priority**self._alpha
            )
            self._it_min[self.step + _ * self.buffer_size] = (
                self._max_priority**self._alpha
            )

        return super().insert(
            obs, next_obs, rewards, termination, truncation, actions, interaction
        )

    def _sample_proportional(self, batch_size):
        """
        Get the index for k (minibatch_size) ranges within the buffer
        :param batch_size: Number of element to sample (minibatch size)
        :return: the replay exprience index list
        """
        res = []
        # the sum over all priorities
        p_total = self._it_sum.sum(0, self.buffer_long - 1)

        # divide the  priority range [0,p_totoal] equally into k(minibatch size) ranges. A value is uniformly sampled from each range.
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            # a value is uniformly sampled from each range
            mass = random.random() * every_range_len + i * every_range_len
            # Find the trainsitions that correspond to each of these sampled value (retrieved from the tree)
            idx = self._it_sum.find_prefixsum_idx(mass)
            # append retrived
            res.append(idx)
        return np.array(res)

    def sample(self, batch_size: int, beta, action_flag):
        """
        Sample a batch of experiences with use PER

        :param batch_size: Number of element to sample (minibatch size)
        :param  beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return: sampled Replay Buffer, importacnce weights
        """

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        # use mininal priority to find the max weight
        max_weight = (p_min * self.buffer_long) ** (-beta)
        for idx in idxes:
            # get the probability of sampling transition of idx
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            # correct bias by importance-sampling weights
            weight = (p_sample * self.buffer_long) ** (-beta)
            # normalize weight
            weights.append(weight / max_weight)

        weights = np.array(weights)
        env_indices = idxes // self.buffer_size
        batch_inds = idxes % self.buffer_size
        return self._get_samples(batch_inds, env_indices, action_flag), idxes

    def update_priorities(self, idxes: np.ndarray, priorities: float):
        """
        Update priorities of sampled transitions

        :param priorities:List of updated priorities corresponding to transitions at the sampled idxes
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.buffer_long
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha
            self._max_priority = max(self._max_priority, priority)
