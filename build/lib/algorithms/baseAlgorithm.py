from abc import ABC, abstractmethod
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TensorDict,
)
import torch
from algorithms.basePolicy import BasePolicy
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)


class BaseAlgorithm(ABC):
    """
    The base of RL algorithms
    """

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        policy_class: Union[str, Type[BasePolicy]],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        gamma: float = 0.99,
        gradient_steps: int = 1,
        device=torch.device("cpu"),
    ) -> None:
        self.all_args = all_args
        self.logger = logger
        self.policy_class = policy_class
        self.env = env

        self.gradient_steps = gradient_steps

        # Used for updating schedules
        self._total_timesteps = 0

        self.action_size = self.env.action_spaces["agent_0"].n

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
        self.policy = self.policy_class(
            self.all_args,
            self.env.observation_spaces["agent_0"],
            self.env.action_spaces["agent_0"],
            self.lr_schedule,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _setup_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_schedule(
        self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # print("lr_schedule:", self.lr_schedule(self._current_progress_remaining))
        # print("current_progress_remaining", self._current_progress_remaining)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

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
