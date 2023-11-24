# import torch
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     MlpExtractor,
#     NatureCNN,
#     create_mlp,
# )
# from gymnasium import spaces

# from utils.util import preprocess_obs
# import numpy as np
# # Example dictionary of tensors
# # my_dict = {
# #     'n_i': torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.],
# #                         [0., 0., 0., 1., 0., 0., 0., 0., 0.],
# #                         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
# #                         [0., 0., 0., 0., 0., 0., 1., 0., 0.]]),
# #     'n_s': torch.tensor([[1., 0.],
# #                         [1., 0.],
# #                         [0., 1.],
# #                         [0., 1.]]),
# #     'n_r': torch.tensor([0., 0., 0., 0.])
# # }
# agents = ["a" for agent in range(10)]
# obs_space = spaces.Dict(
#     {
#         "n_i": spaces.MultiDiscrete([len(agents)] * 4),
#         "n_s": spaces.MultiDiscrete([2]*4),
#         "n_r": spaces.Box(low=-10, high=10, shape=(4,1)),
#     }
# )
# obs = obs_space.sample()
# print(obs)
# features_extractor=CombinedExtractor(obs_space)
# print(features_extractor,features_extractor.features_dim)

# for key, value in obs.items():
#     if isinstance(value, np.ndarray):
#         # Convert NumPy array to a PyTorch tensor
#         tensor_value = torch.tensor(value)
#         obs[key] = tensor_value

# p_obs=preprocess_obs(obs,obs_space)
# print(p_obs)
# # Combine the tensors into one tensor
# extract_features=features_extractor(p_obs)

# print(extract_features)

# # combined_tensor = torch.cat([my_dict[key].flatten() for key in my_dict], dim=0)
# # combined_tensor.flatten(1,-1)


# # import torch.nn as nn
# # import torch

# # # Create a Flatten layer with start_dim=1 and end_dim=-1
# # flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)

# # # Suppose you have a tensor of shape (batch_size, channels, height, width)
# # input_tensor = torch.randn(3, 3, 64, 64)

# # # Apply the Flatten layer to the input tensor
# # output_tensor = flatten_layer(input_tensor)

# # print(input_tensor)
# # print(output_tensor)


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

# print(a.shape)

# assq=np.squeeze(a,axis=0)

# print(assq.shape)

# from configs.config import get_config, update_config
# import sys
# import torch
# import numpy as np
# from pathlib import Path
# import os
# import setproctitle
# import wandb
# import socket
# from datetime import datetime
# from envs.env_wrappers import DummyVecEnv,SubprocVecEnv
# from envs.matrix_dilemma._md_utils.utils import (
#     make_env,
#     gen_lattice_neighbours,
#     parallel_wrapper_fn,
# )

# def parse_args(parser):
#     parser.add_argument(
#         "--scenario_name",
#         type=str,
#         default="lattice_egt",
#         help="Which scenario to run on",
#     )

#     return parser


# if __name__ == "__main__":
#     parser = get_config()
#     parser = parse_args(parser)

#     # parse command-line arguments and pre-set argument from config.py
#     # The first element of the tuple is an object containing the parsed arguments
#     # The second element is a list of any remaining, unrecognized arguments.
#     parsed_args = parser.parse_known_args(sys.argv[1:])[0]
#     all_args = update_config(parsed_args)

#     # cuda
#     if all_args.cuda and torch.cuda.is_available():
#         print("choose to use gpu...")
#         device = torch.device("cuda:0")
#         torch.set_num_threads(all_args.n_training_threads)
#         if all_args.cuda_deterministic:
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#     else:
#         print("choose to use cpu...")
#         device = torch.device("cpu")
#         torch.set_num_threads(all_args.n_training_threads)


#     run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
#                    0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
#     if not run_dir.exists():
#         os.makedirs(str(run_dir))

#     if all_args.use_wandb:
#         run = wandb.init(config=all_args,
#                          project=all_args.env_name,
#                          entity=all_args.user_name,
#                          notes=socket.gethostname(),
#                          name=str(all_args.algorithm_name) + "_" +
#                          str(all_args.experiment_name) +
#                          "_seed" + str(all_args.seed),
#                          group=all_args.scenario_name,
#                          dir=str(run_dir),
#                          job_type="training",
#                          reinit=True)
#     else:
#         # Generate a run name based on the current timestamp
#         current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#         curr_run = f'run_{current_time}'


#         # Create the full path for the new run directory                
#         run_dir = run_dir / curr_run
#         if not run_dir.exists():
#             os.makedirs(str(run_dir))


#     setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
#         str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    
    
#     # seed
#     # torch.manual_seed(all_args.seed)
#     # torch.cuda.manual_seed_all(all_args.seed)
#     # np.random.seed(all_args.seed)

#     # env init
#     if all_args.scenario_name == "lattice_egt":
#         from envs.matrix_dilemma import lattice_egt_v0 as LatticeENV
#     elif all_args.scenario_name == "lattice_rl":
#         from envs.matrix_dilemma import lattice_rl_v0 as LatticeENV

#     env=LatticeENV.parallel_env(all_args)

#     env.reset()
#     # while env.agents:
#     #     # this is where you would insert your policy
#     actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos=env.step(actions)
    
#     # print(observations)
#     # print(rewards)
    
#     env.close()

# from pettingzoo.utils.conversions import aec_to_parallel_wrapper

# def make_train_env(all_args,raw_env):
#     def get_env_fn():
#         def init_env():
#            env= raw_env(all_args,max_cycles=all_args.num_env_steps)
#         #    env=aec_to_parallel_wrapper(env)
#            return env
#         return init_env
#     if all_args.n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn()])
#     else:
#         return SubprocVecEnv([get_env_fn() for i in range(all_args.n_rollout_threads)])

# envs=make_train_env(all_args,LatticeENV.raw_env)

# obs=envs.reset()
# print(obs)

# actions = {agent: envs.action_spaces[agent].sample() for agent in envs.agents}
# print(envs.action_spaces)
# print(envs.agents)


# import gymnasium as gym

# from stable_baselines3 import DQN

# env = gym.make("CartPole-v1")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000, log_interval=4)

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# writer = SummaryWriter()

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# writer = SummaryWriter()
# # a=np.array([
# #     [[2,4],[1,2]],[[2,4],[1,2]]
# # ])
# # print(np.mean(a))
# for n_iter in range(50):
#     writer.add_scalar("Loss/train", 0.2,n_iter)


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
# from torch.utils.tensorboard import SummaryWriter


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
#     if args.track:
#         import wandb

#         wandb.init(
#             project=args.wandb_project_name,
#             entity=args.wandb_entity,
#             sync_tensorboard=True,
#             config=vars(args),
#             name=run_name,
#             monitor_gym=True,
#             save_code=True,
#         )
#     writer = SummaryWriter(f"runs/{run_name}")
#     writer.add_text(
#         "hyperparameters",
#         "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
#     )

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

#         # TRY NOT TO MODIFY: record rewards for plotting purposes
#         if "final_info" in infos:
#             for info in infos["final_info"]:
#                 # Skip the envs that are not done
#                 if "episode" not in info:
#                     continue
#                 print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
#                 writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
#                 writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
#                 writer.add_scalar("charts/epsilon", epsilon, global_step)

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
#                 old_val = q_network(data.observations).gather(1, data.actions).squeeze()
#                 loss = F.mse_loss(td_target, old_val)
#                 print('q value',td_target,old_val)
#                 print(loss)
#                 if global_step % 100 == 0:
#                     writer.add_scalar("losses/td_loss", loss, global_step)
#                     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
#                     print("SPS:", int(global_step / (time.time() - start_time)))
#                     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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



# import numpy as np
# import matplotlib.pyplot as plt

# # ensure your arr is sorted from lowest to highest values first!
# arr = np.array([1,4,6,9,100])

# def gini(arr):
#     count = arr.size
#     coefficient = 2 / count
#     indexes = np.arange(1, count + 1)
#     weighted_sum = (indexes * arr).sum()
#     total = arr.sum()
#     constant = (count + 1) / count
#     return coefficient * weighted_sum / total - constant

# def lorenz(arr):
#     # this divides the prefix sum by the total sum
#     # this ensures all the values are between 0 and 1.0
#     scaled_prefix_sum = arr.cumsum() / arr.sum()
#     # this prepends the 0 value (because 0% of all people have 0% of all wealth)
#     return np.insert(scaled_prefix_sum, 0, 0)



# def gini2(array):
#     """Calculate the Gini coefficient of a numpy array."""
#     # based on bottom eq:
#     # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
#     # from:
#     # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
#     # All values are treated equally, arrays must be 1d:
#     array = array.flatten()
#     if np.amin(array) < 0:
#         # Values cannot be negative:
#         array -= np.amin(array)
#     # Values cannot be 0:
#     array = array + 0.0000001
#     # Values must be sorted:
#     array = np.sort(array)
#     # Index per array element:
#     index = np.arange(1,array.shape[0]+1)
#     # Number of array elements:
#     n = array.shape[0]
#     # Gini coefficient:
#     return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

# # show the gini index!
# print(gini(arr))
# print(gini2(arr))

# lorenz_curve = lorenz(arr)

# # we need the X values to be between 0.0 to 1.0
# plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)
# # plot the straight line perfect equality curve
# plt.plot([0,1], [0,1])
# plt.show()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Net().to(device)

# summary(model, (1, 28, 28))

# import numpy as np
# my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# print(my_array[-3:])
# print(my_array[:3])


# import numpy as np

# # Your list of arrays
# arrays = [np.array([[1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
#                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
#                    [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
#                    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]]),
#          np.array([[1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
#                    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
#                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]])]


# def consecutive_counts(row):
#     """
#     Calculate the consecutive counts, average counts, and longest consecutive counts for a given row.

#     :param row: The row for which consecutive counts should be calculated.
#     :return: A tuple of consecutive counts, average counts, and longest consecutive counts.
#     """
#     consecutive_counts = []
#     current_count = 0
#     current_target = None
#     total_counts = {}
#     longest_consecutive = {}

#     for num in row:
#         if num == current_target:
#             current_count += 1
#         else:
#             if current_target is not None:
#                 consecutive_counts.append(current_count)
#                 if current_target not in total_counts:
#                     total_counts[current_target] = [current_count, 1]
#                 else:
#                     total_counts[current_target][0] += current_count
#                     total_counts[current_target][1] += 1

#                 if current_count > longest_consecutive.get(current_target, 0):
#                     longest_consecutive[current_target] = current_count

#             current_target = num
#             current_count = 1

#     consecutive_counts.append(current_count)
#     if current_target not in total_counts:
#         total_counts[current_target] = [current_count, 1]
#     else:
#         total_counts[current_target][0] += current_count
#         total_counts[current_target][1] += 1
        
#     if current_count > longest_consecutive.get(current_target, 0):
#         longest_consecutive[current_target] = current_count


#     return consecutive_counts, total_counts,longest_consecutive

# # Calculate consecutive counts and average counts for each row and print the summary
# for array in arrays:
#     for i, row in enumerate(array):
#         counts, total_counts,longest_consecutive = consecutive_counts(row)
#         print(f"Row {i + 1}: Consecutive Counts: {counts}")
#         print(f"Row {i + 1}: Average Counts:")
#         print(longest_consecutive)
#         for target, (total, count) in total_counts.items():
#             # print((total, count))
#             average_count = total / count
#             print(f"  Target {target} (or unknown): {average_count}")
#         for target,duration in longest_consecutive.items():
#             print(target,duration)
# import numpy as np

# # Create a sample array with rows of different sizes
# data = np.array([[1, 2, 3],
#                  [4, 5],
#                  [6, 7, 8, 9]])

# # Create a mask to represent the valid elements in each row
# mask = ~np.isnan(data)
# print(mask)
# # Calculate the mean for each row, considering only valid elements
# row_means = np.nanmean(data, axis=1)

# print(row_means)


# import numpy as np

# # Sample 2D NumPy array
# array = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [np.nan, 8, 9]])

# # Calculate the mean by column
# column_means = np.nanmean(array, axis=0)

# print("Mean by column:")
# print(column_means)



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors

# # Define color set for plt.imshow
# color_set = np.array(["#0c056d", "#000000"])

# # Divide each color into four subcolors
# num_subcolors = 4
# subcolors = []

# for color in color_set:
#     # Convert hex color to RGB
#     rgb_color = colors.hex2color(color)
    
#     # Create a gradient of colors
#     gradient = np.linspace(0, 1, num_subcolors)
    
#     # Interpolate RGB values along the gradient
#     subcolor_set = [colors.rgb2hex(np.interp(gradient, [0, 1], [rgb_color[j], 1])) for j in range(3)]
    
#     subcolors.append(subcolor_set)

# # Flatten the subcolors list
# subcolors = np.array(subcolors).flatten()

# # Define a new colormap using the subcolors
# cmap = colors.ListedColormap(subcolors)

# # Set up bounds and normalization
# bounds = np.arange(2 * num_subcolors + 1)
# ncolors = len(bounds) - 1  # Set ncolors to the number of bins
# norm = colors.BoundaryNorm(bounds, ncolors)

# # Example usage with plt.imshow
# data = np.random.random((10, 10))  # Replace this with your actual data
# plt.imshow(data, cmap=cmap, norm=norm)
# plt.colorbar()

# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib import colors
# # Make some illustrative fake data:

# x = np.arange(0, np.pi, 0.1)
# y = np.arange(0, 2 * np.pi, 0.1)
# X, Y = np.meshgrid(x, y)
# Z = np.cos(X) * np.sin(Y) * 10

# # # Define color set for plt.imshow
# # color_set = np.array(["#0c056d", "red"])

# # for color in color_set:
# #     # Convert hex color to RGB
# #     rgb_color = colors.hex2color(color)
# #     print(rgb_color)

# print(Z)

# colors = ["red","#0c056d"]  # R -> G -> B
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
# cmap_name = 'my_list'
# fig, axs = plt.subplots(2, 2, figsize=(6, 9))
# fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
# for n_bin, ax in zip(n_bins, axs.flat):
#     # Create the colormap
#     cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
#     print(cmap)
#     # Fewer bins will result in "coarser" colomap interpolation
#     im = ax.imshow(Z, origin='lower', cmap=cmap)
#     ax.set_title("N bins: %s" % n_bin)
#     fig.colorbar(im, ax=ax)
# plt.show()


# from matplotlib.colors import to_rgba

# def interpolate_color(color1, color2, factor):
#     # Convert color names to RGB
#     def color_to_rgb(color):
#         rgba = to_rgba(color)
#         return tuple(int(c * 255) for c in rgba[:3])

#     def rgb_to_hex(rgb_color):
#         return "#{:02x}{:02x}{:02x}".format(*rgb_color)

#     rgb_color1 = color_to_rgb(color1)
#     rgb_color2 = color_to_rgb(color2)

#     # Interpolate RGB values
#     interpolated_rgb = tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(rgb_color1, rgb_color2))

#     # Convert back to hex
#     interpolated_hex = rgb_to_hex(interpolated_rgb)

#     return interpolated_hex

# # Example usage
# color1 = "#0c056d"
# color2 = "red"
# factor = 1 / 8  # Interpolation factor

# interpolated_color = interpolate_color(color1, color2, factor)
# print(interpolated_color)

# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.colors import Normalize


# def normal_pdf(x, mean, var):
#     return np.exp(-(x - mean)**2 / (2*var))


# # Generate the space in which the blobs will live
# xmin, xmax, ymin, ymax = (0, 100, 0, 100)
# n_bins = 100
# xx = np.linspace(xmin, xmax, n_bins)
# yy = np.linspace(ymin, ymax, n_bins)

# # Generate the blobs. The range of the values is roughly -.0002 to .0002
# means_high = [20, 50]
# means_low = [50, 60]
# var = [150, 200]

# gauss_x_high = normal_pdf(xx, means_high[0], var[0])
# gauss_y_high = normal_pdf(yy, means_high[1], var[0])

# gauss_x_low = normal_pdf(xx, means_low[0], var[1])
# gauss_y_low = normal_pdf(yy, means_low[1], var[1])

# weights = (np.outer(gauss_y_high, gauss_x_high)
#            - np.outer(gauss_y_low, gauss_x_low))

# print(len(weights))

# # Create an alpha channel of linearly increasing values moving to the right.
# alphas = np.ones(weights.shape)
# alphas[:, 30:] = np.linspace(1, 0, 70)

# print(alphas)

# # We'll also create a grey background into which the pixels will fade
# greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)
# # print(greys)

# # First we'll plot these blobs using ``imshow`` without transparency.
# vmax = np.abs(weights).max()
# imshow_kwargs = {
#     'vmax': vmax,
#     'vmin': -vmax,
#     'cmap': 'RdYlBu',
#     'extent': (xmin, xmax, ymin, ymax),
# }

# fig, ax = plt.subplots()
# # ax.imshow(greys)
# ax.imshow(weights,alpha=alphas, **imshow_kwargs)
# # ax.imshow(weights,alpha=alphas, **imshow_kwargs)
# ax.set_axis_off()
# # plt.show()


# def get_next_elements(arr, start_index, num_elements):
#     # Calculate the effective start and end indices
#     effective_start = start_index % len(arr)
#     effective_end = (effective_start + num_elements) % len(arr)

#     # Handle the case where the range wraps around the array boundary
#     if effective_end > effective_start:
#         result_indices = list(range(effective_start, effective_end))
#     else:
#         result_indices = list(range(effective_start, len(arr))) + list(range(0, effective_end))

#     return result_indices

# # Example usage
# array_size = 100
# start_index = 99
# num_elements = 10

# indices = get_next_elements(list(range(array_size)), start_index, num_elements)
# print(indices)

# def get_indices(start_index, array_size, num_elements):
#     indices = [(start_index - i) % array_size for i in range(1, num_elements + 1)]
#     return indices

# array_size = 100
# start_index = 5
# num_elements = 10

# result = get_indices(start_index, array_size, num_elements)
# print(result)

# episode_acts_flattened=[
#     [],[]
# ]
# episode_acts_transposed = [
#             np.array(_).T for _ in episode_acts_flattened
#         ] 

# def middle_and_nearby_indices(dim, radius_length):
#     assert radius_length*2<dim,f"Expected radius_length< {int(dim/2)}, but got {radius_length}"
#     middle_idx = dim // 2 * dim + dim // 2

#     nearby_indices = []

#     for i in range(-radius_length, radius_length + 1):
#         for j in range(-radius_length, radius_length + 1):
#             # Calculate the flattened index
#             index = middle_idx + i * dim + j

#             # Check if the index is within the bounds of the flattened list
#             if 0 <= index < dim * dim:
#                 nearby_indices.append(index)

#     return middle_idx, nearby_indices

# # Example usage:
# dim = 10
# radius = 1

# middle_point, nearby_idxs = middle_and_nearby_indices(dim, radius)

# print("Middle point index:", middle_point)
# print("Nearby indices with radius {}: {}".format(radius, nearby_idxs))

import numpy as np

# Your list of arrays
array_list = [
    np.array([[[-0.3, -0.63333333], [-0.61538462, -1.3]]]),
    np.array([[[-0.53333333, 0.], [-0.53333333, 1.3]]]),
    # Add more arrays as needed
]
print(array_list)
# Combine the first values from all arrays in the list
first_values_combined = np.concatenate([arr[..., 0].ravel() for arr in array_list])

# Combine the second values from all arrays in the list
second_values_combined = np.concatenate([arr[..., 1].ravel() for arr in array_list])

# Reshape the arrays to get the desired shape
first_values_combined = first_values_combined.reshape(len(array_list), -1)
second_values_combined = second_values_combined.reshape(len(array_list), -1)

# Print the results
print("First values combined:\n", first_values_combined)
print("\nSecond values combined:\n", second_values_combined)