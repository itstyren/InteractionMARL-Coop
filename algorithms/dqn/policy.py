import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import QNetwork
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Type
from algorithms.basePolicy import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor, 
        features_dim: int,               
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = False,
    ) -> None:
            
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64,64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim 
        # print(get_action_dim(self.action_space))
        action_dim = int(self.action_space.n)  # number of actions
        # print(action_dim)
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        '''
        Predict action
        '''
 
        q_values = self(observation)
        action = q_values.argmax().reshape(-1)

        return action

class DQN_Policy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    
    :param args: cli args input
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        args,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        device=torch.device("cpu")
        
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )        

        self.args = args
        self.lr = args.lr
        self.device=device


        self.observation_space = observation_space
        self.action_space = action_space

        if net_arch is None:
            net_arch = [32, 32]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
        }

        self._build(lr_schedule)

    def _build(self,lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
        lr_schedule(1) is the initial learning rate        
        """
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        self.optimizer = self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def get_actions(self,obs_buffer:torch.Tensor)-> torch.Tensor:
        actions=[]
        for i in range(len(obs_buffer)):
            with torch.no_grad():
                action=self(obs_buffer[i])
            actions.append(action)
        return torch.cat(actions, dim=0)