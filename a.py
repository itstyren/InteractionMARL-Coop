# from gymnasium import spaces
# import numpy as np
# import torch
# from torch.nn import functional as F
# from typing import Dict, Tuple, Union
# import warnings
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     MlpExtractor,
#     NatureCNN,
#     create_mlp,
# )


# def preprocess_obs(
#     obs: torch.Tensor,
#     observation_space: spaces.Space,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     if isinstance(observation_space, spaces.Box):
#         return obs.float()
#     if isinstance(observation_space, spaces.MultiDiscrete):
#         # Tensor concatenation of one hot encodings of each Categorical sub-space
#         print(obs)
#         return torch.cat([
#             F.one_hot(obs[_].long(), num_classes=dim).float()
#             for _,dim in enumerate(observation_space.nvec)
#         ],dim=-1).view(obs.shape[0], observation_space.nvec[0])
#     elif isinstance(observation_space, spaces.Dict):
#         # Do not modify by reference the original observation
#         assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
#         preprocessed_obs = {}
#         for key, _obs in obs.items():
#             preprocessed_obs[key] = preprocess_obs(
#                 _obs, observation_space[key]
#             )
#         return preprocessed_obs


# # agents=[1,2,3]

# # obs_space=spaces.Dict({
# #                 "n_i": spaces.MultiDiscrete([len(agents)]*4),
# #                 "n_s": spaces.MultiDiscrete([1]*4),
# #                 "n_r": spaces.Box(low=-10, high=10, shape=(4,))
# #             }

# #             )

# # obs=obs_space.sample()
# # preprocess_obs(obs,obs_space)
# # print(obs)

# agents = ["a" for agent in range(10)]
# obs_space = spaces.Dict(
#     {
#         "n_i": spaces.MultiDiscrete([len(agents)] * 4),
#         "n_s": spaces.MultiDiscrete([2]*4),
#         "n_r": spaces.Box(low=-10, high=10, shape=(4,)),
#     }
# )
# obs = obs_space.sample()
# print(obs)
# print(spaces.utils.flatdim(obs_space))


# for key, value in obs.items():
#     if isinstance(value, np.ndarray):
#         # Convert NumPy array to a PyTorch tensor
#         tensor_value = torch.tensor(value)
#         obs[key] = tensor_value

# # obs=torch.tensor(obs)
# print(obs)

# po=preprocess_obs(obs, obs_space)
# print(po)

# # for idx, obs_ in enumerate(obs):
# #     print(obs_)
# #     print(obs_space[idx].nvec)
# #     print('obs_=',obs_)
# #     f=F.one_hot(obs_.long()).float()
# #     print('f=',f)
# # print(F.one_hot(obs_.long(), num_classes=1).float())
# # print(obs)

# # print(torch.arange(0, 5) % 3)
# # print(spaces.utils.flatdim(obs_space))
# # dim=obs_space.nvec[0]
# # f=F.one_hot(obs.long(),num_classes=dim).float()
# # print(f)


# import numpy as np

# a=np.array(
#     [
#         [
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 0.0],
#         ],
#         [
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [1.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 0.0],
#         ],
#     ]
# )

# print(a)



# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
# import argparse
# import os
# import random
# import time
# from distutils.util import strtobool

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer


# def parse_args():
#     # fmt: off
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
#         help="the name of this experiment")
#     parser.add_argument("--seed", type=int, default=1,
#         help="seed of the experiment")
#     parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, `torch.backends.cudnn.deterministic=False`")
#     parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, cuda will be enabled by default")
#     parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="if toggled, this experiment will be tracked with Weights and Biases")
#     parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
#         help="the wandb's project name")
#     parser.add_argument("--wandb-entity", type=str, default=None,
#         help="the entity (team) of wandb's project")
#     parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to capture videos of the agent performances (check out `videos` folder)")
#     parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to save model into the `runs/{run_name}` folder")
#     parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to upload the saved model to huggingface")
#     parser.add_argument("--hf-entity", type=str, default="",
#         help="the user or org name of the model repository from the Hugging Face Hub")

#     # Algorithm specific arguments
#     parser.add_argument("--env-id", type=str, default="CartPole-v1",
#         help="the id of the environment")
#     parser.add_argument("--total-timesteps", type=int, default=500000,
#         help="total timesteps of the experiments")
#     parser.add_argument("--learning-rate", type=float, default=2.5e-4,
#         help="the learning rate of the optimizer")
#     parser.add_argument("--num-envs", type=int, default=1,
#         help="the number of parallel game environments")
#     parser.add_argument("--buffer-size", type=int, default=10000,
#         help="the replay memory buffer size")
#     parser.add_argument("--gamma", type=float, default=0.99,
#         help="the discount factor gamma")
#     parser.add_argument("--tau", type=float, default=1.,
#         help="the target network update rate")
#     parser.add_argument("--target-network-frequency", type=int, default=500,
#         help="the timesteps it takes to update the target network")
#     parser.add_argument("--batch-size", type=int, default=128,
#         help="the batch size of sample from the reply memory")
#     parser.add_argument("--start-e", type=float, default=1,
#         help="the starting epsilon for exploration")
#     parser.add_argument("--end-e", type=float, default=0.05,
#         help="the ending epsilon for exploration")
#     parser.add_argument("--exploration-fraction", type=float, default=0.5,
#         help="the fraction of `total-timesteps` it takes from start-e to go end-e")
#     parser.add_argument("--learning-starts", type=int, default=10000,
#         help="timestep to start learning")
#     parser.add_argument("--train-frequency", type=int, default=10,
#         help="the frequency of training")
#     args = parser.parse_args()
#     # fmt: on
#     assert args.num_envs == 1, "vectorized envs are not supported at the moment"

#     return args


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)

#         return env

#     return thunk


# # ALGO LOGIC: initialize agent here:
# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, env.single_action_space.n),
#         )

#     def forward(self, x):
#         return self.network(x)


# def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
#     slope = (end_e - start_e) / duration
#     return max(slope * t + start_e, end_e)


# if __name__ == "__main__":
#     import stable_baselines3 as sb3

#     if sb3.__version__ < "2.0":
#         raise ValueError(
#             """Ongoing migration: run the following command to install the new dependencies:

# poetry run pip install "stable_baselines3==2.0.0a1"
# """
#         )
#     args = parse_args()
#     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

#     # TRY NOT TO MODIFY: seeding
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = args.torch_deterministic

#     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

#     # env setup
#     envs = gym.vector.SyncVectorEnv(
#         [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
#     )
#     assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

#     q_network = QNetwork(envs).to(device)
#     optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
#     target_network = QNetwork(envs).to(device)
#     target_network.load_state_dict(q_network.state_dict())

#     rb = ReplayBuffer(
#         args.buffer_size,
#         envs.single_observation_space,
#         envs.single_action_space,
#         device,
#         handle_timeout_termination=False,
#     )
#     start_time = time.time()

#     # TRY NOT TO MODIFY: start the game
#     obs, _ = envs.reset(seed=args.seed)
#     for global_step in range(args.total_timesteps):
#         # ALGO LOGIC: put action logic here
#         epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
#         if random.random() < epsilon:
#             actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
#         else:
#             q_values = q_network(torch.Tensor(obs).to(device))
#             actions = torch.argmax(q_values, dim=1).cpu().numpy()

#         # TRY NOT TO MODIFY: execute the game and log data.
#         next_obs, rewards, terminated, truncated, infos = envs.step(actions)


#         # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
#         real_next_obs = next_obs.copy()
#         for idx, d in enumerate(truncated):
#             if d:
#                 real_next_obs[idx] = infos["final_observation"][idx]
#         rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

#         # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
#         obs = next_obs

#         # ALGO LOGIC: training.
#         if global_step > args.learning_starts:
#             if global_step % args.train_frequency == 0:
#                 data = rb.sample(args.batch_size)
#                 with torch.no_grad():
#                     target_max, _ = target_network(data.next_observations).max(dim=1)
#                     td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
#                 # print('td_target',td_target)
#                 # print(q_network(data.observations))
#                 # print('actions',data.actions)
#                 print(q_network(data.observations).gather(1, data.actions))
#                 old_val = q_network(data.observations).gather(1, data.actions).squeeze()
#                 print(old_val)
#                 # print('old_val',old_val)
#                 print('======================')
#                 loss = F.mse_loss(td_target, old_val)

#                 # optimize the model
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # update target network
#             if global_step % args.target_network_frequency == 0:
#                 for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
#                     target_network_param.data.copy_(
#                         args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
#                     )



#     envs.close()
#     writer.close()


# import torch
# print(torch.cuda.is_available())

# import numpy as np

# # Define the range of numbers (0 to 15)
# num_range = np.arange(16)

# # Convert each number to a binary representation with a length of 4 using NumPy
# binary_matrix = (num_range[:, np.newaxis] & (2**np.arange(4))) > 0

# # The binary_matrix now contains the mappings for numbers 0 to 15
# print(binary_matrix.astype(int)[0])

# import torch
# import torch.nn as nn
# import numpy as np

# # NLP Example
# batch, sentence_length, embedding_dim = 20, 5, 10
# embedding = torch.randn(batch, sentence_length, embedding_dim)
# layer_norm = nn.LayerNorm(embedding_dim)
# # Activate module
# layer_norm(embedding)
# # Image Example

# array_data = np.random.rand(10)
# print(array_data)
# array_copy = array_data.copy()
# mean_rewards = np.nanmean(array_copy)
# std_rewards = np.nanstd(array_copy)
# sample_rewards = (array_data - mean_rewards) / (std_rewards + 1e-5)
# print(sample_rewards)

# tensor_data = torch.tensor(array_data)
# tensor_data = tensor_data.to(torch.bfloat16)


# layer_norm = nn.LayerNorm(10)

# output = layer_norm(tensor_data)
# # Convert the PyTorch tensor to a CPU float tensor and detach gradients
# cpu_float_tensor = output.cpu().detach().float()
# print(cpu_float_tensor.numpy())
import numpy as np

strategy_reward = [[], []]
strategy_mean_reward = [np.nanmean(s_r) if s_r else 0 for s_r in strategy_reward]
print(strategy_mean_reward)