import numpy as np
import torch
import math
from gymnasium import spaces
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    polyak_update,
)
from stable_baselines3.dqn.policies import QNetwork
from algorithms.dqn.policy import DQN_Policy
from algorithms.baseAlgorithm import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from torch.nn import functional as F
from utils.util import convert_array_to_two_arrays




class Strategy_DQN(BaseAlgorithm):
    """
    Policy class with Q-Value Net and target net for DQN

    :param all_args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQN_Policy

    def __init__(
        self,
        all_args,
        logger,
        policy_class: Union[str, Type[DQN_Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        prioritized_replay_beta: Union[float, Schedule] = 0.4,
        prioritized_replay_eps:float=1e-6,
        batch_size: int = 32,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        device=torch.device("cpu"),
        _init_setup_model: bool = True,
        action_flag:int=0
    ) -> None:
        super().__init__(
            all_args,
            logger,
            env,
            policy_class,
            learning_rate,
            prioritized_replay_beta,
            prioritized_replay_eps,
            batch_size,
            gamma,
            gradient_steps,
            action_flag=action_flag
        )
        self.device = device
        # self.policy = policy
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = all_args.target_update_interval

        # For updating the target network with multiple envs:
        self._n_calls = 0
        self._target_update=0
        self.max_grad_norm = max_grad_norm

        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if all_args.tune_entropy:
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=all_args.lr, eps=1e-4
            )
        else:
            self.alpha = all_args.entropy_weight

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.q_net_target, ["running_"]
        )
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def prep_training(self):
        self.policy.q_net.train()

    def prep_rollout(self):
        self.policy.q_net.eval()

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        if (
            self._n_calls
            % max(self.target_update_interval // self.all_args.n_rollout_threads, 1)
            == 0
        ):
            polyak_update(
                self.q_net.parameters(), self.q_net_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._target_update+=1

        return self.exploration_rate,self._target_update

    def predict(
        self,
        obs_buffer: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :return: the model's action
        """
        # print(obs_buffer)
        # print(self.policy.action_space)
        # print(obs_buffer)
        action = []
        # print("exploration_rate:", self.exploration_rate)
        # actions = []
        # # print(self.policy.action_space)
        # # print(self.action_flag)
        # # print('====')
        # for _ in range(len(obs_buffer)):
        #     action = torch.tensor([self.policy.action_space.sample()])
        #     actions.append(action)
        # return torch.cat(actions, dim=0)        
        
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = []
            # print(self.policy.action_space)
            # print(self.action_flag)
            # print('====')
            for _ in range(len(obs_buffer)):
                action = torch.tensor([self.policy.action_space.sample()])
                actions.append(action)
            return torch.cat(actions, dim=0)
        else:
            # return self.policy.get_actions(obs_buffer)
            # print(self.policy.action_space)
            # print(self.policy.predict(obs_buffer))
            # print('====')
            return self.policy.predict(obs_buffer)
        # return self.policy.predict(obs_buffer)

    def train(self, batch_size, replay_buffer,action_flag) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        lr, beta = self._update_schedule(self.policy.optimizer)
        train_info = {}

        train_rewards=[[],[]]

        losses = []
        for _ in range(self.gradient_steps):
            if self.all_args.replay_scheme == "uniform":
                replay_data = replay_buffer.sample(batch_size)
            else:
                # prioritized sampling
                replay_data,priority_idxes = replay_buffer.sample(batch_size, beta,action_flag)

            # train every agent
            next_q_values = []
            current_q_values = []
            for i in range(len(replay_data.next_observations)):
                with torch.no_grad():
                    # Compute the next Q-values using the target network
                    next_q = self.q_net_target(replay_data.next_observations[i])
                    next_q_values.append(next_q)
                # Get current Q-values estimates
                current_q = self.q_net(replay_data.observations[i])
                current_q_values.append(current_q)
            with torch.no_grad():
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = torch.stack(next_q_values).max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                if action_flag==0 or action_flag==2:
                    if action_flag==0:
                        acts=replay_data.actions
                    else:
                        acts,intacts=convert_array_to_two_arrays(replay_data.actions)
                    for a,r in zip(acts,replay_data.rewards):
                        train_rewards[a.item()].append(r.item())
                    for i,rewards in enumerate(train_rewards):
                        if len(rewards)>0:
                            train_rewards[i]=np.mean(rewards)
                        else:
                            train_rewards[i]=None
                            
            # print(replay_data.actions,self.action_flag)
            # print(self.gamma * next_q_values)
            # print(replay_data.dones)
            # print('next_q:',next_q_values)
            # print('rewards:',replay_data.rewards)
            current_q_values = torch.stack(current_q_values)
            # print('action code:',replay_data.actions.unsqueeze(1).long())
            # print('reward:',replay_data.rewards)
            # print('action:',replay_data.actions)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(
                current_q_values, dim=1, index=replay_data.actions.unsqueeze(1).long()
            ).squeeze()
            # print('current_q_values',current_q_values)
            # print('target_q_values:',target_q_values)

            # Compute Huber loss (aka the loss, less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            # print('loss:',loss)
            losses.append(loss.item())
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.all_args.replay_scheme == "prioritized":
                td_errors=(current_q_values-target_q_values).numpy(force=True)
                # if np.any(np.isnan(td_errors)):
                #     print(current_q_values)
                #     print(target_q_values)
                    # print(replay_data)
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                # print(new_priorities,self.prioritized_replay_eps)
                replay_buffer.update_priorities(priority_idxes, new_priorities)

        # Increase update counter
        self._n_updates += self.gradient_steps
        train_info["train/loss"] = np.mean(losses)
        train_info["train/n_updates"] = self._n_updates
        train_info["train/lr"] = lr
        train_info["train/prioritized_replay_beta"] = beta
        train_info["train/cooperation_reward"] = train_rewards[0]
        train_info["train/defection_reward"] = train_rewards[1]
        return train_info
