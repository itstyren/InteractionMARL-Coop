from abc import ABC
from stable_baselines3.common.utils import (
    get_schedule_fn,
    update_learning_rate,
)
import numpy as np

from typing import (
    List,
    Type,
    Union,
)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
import torch
from algorithms.basePolicy import BasePolicy


class BaseAlgorithm(ABC):
    """
    Base class for Reinforcement Learning algorithms.

    :param learning_rate: The learning rate for the optimizer. 
                          This can be a fixed float value or a function that dynamically adjusts based on the current progress (decreasing from 1 to 0).
    :param action_flag: Specifies the mode of action. It can be one of the following:
                        0 - Strategy only; 1 - Interaction only; 2 - Strategy and Interaction
    """

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        policy_class: Union[str, Type[BasePolicy]],
        learning_rate: Union[float, Schedule],
        prioritized_replay_beta: Union[float, Schedule],
        prioritized_replay_eps:float,
        gamma: float = 0.99,            
        gradient_steps: int = 1,
        device=torch.device("cpu"),
        action_flag: int=0
    ) -> None:
        self.all_args = all_args
        self.logger=logger
        self.policy_class = policy_class
        self.env = env

        self.gradient_steps = gradient_steps
        self.action_flag=action_flag

        # Used for updating schedules
        self._total_timesteps = 0
        self.action_size=self.env.action_spaces["agent_0"][self.action_flag].n

        self.prioritized_replay_beta=prioritized_replay_beta
        self.prioritized_replay_eps=prioritized_replay_eps

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = all_args.tau
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        self.device = device
        
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int

    def _setup_model(self) -> None:
        self._setup_schedule()
        optimizer_kwargs = {
            "weight_decay": self.all_args.weight_decay,
            "eps": self.all_args.opti_eps,
        }

        if self.action_flag==1: # 1 means obs space for interaction model
            obs_space=self.env.interact_observation_spaces["agent_0"]
        else:
            obs_space=self.env.observation_spaces["agent_0"]
            
        # obs_space=self.env.observation_spaces["agent_0"]

        self.policy = self.policy_class(
            self.all_args,
            obs_space,
            self.env.action_spaces["agent_0"][self.action_flag],
            self.lr_schedule,
            self.all_args.net_arch,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _setup_schedule(self) -> None:
        """Transform to callable if needed."""
        # return constant value of callable function based on input type of learning_rate
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.beta_schedual=get_schedule_fn(self.prioritized_replay_beta)
        
    def _update_schedule(
        self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
    ) -> float:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        lr =self.lr_schedule(self._current_progress_remaining)
        prioritized_replay_beta=self.beta_schedual(self._current_progress_remaining)
        
        # update optimizer
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, lr)
            
        return lr,prioritized_replay_beta

    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )
