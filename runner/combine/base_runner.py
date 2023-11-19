import wandb
import io
import os
from tensorboardX import SummaryWriter
from utils.util import linear_schedule, round_up, FileManager, find_latest_file
import time
import sys
import torch
import numpy as np
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
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
import pathlib
from stable_baselines3.common.callbacks import ConvertCallback
from utils.callback import CheckpointCallback, BaseCallback
from torchinfo import summary
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
import zipfile
from envs.matrix_dilemma._md_utils.utils import get_central_and_nearby_indices
from gymnasium import spaces


class Runner(object):
    """
    Base Runner calss for RL training
    """

    def __init__(
        self,
        config,
    ):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.env_name = self.all_args.env_name
        self.use_wandb = self.all_args.use_wandb
        self.lr = self.all_args.lr
        self.beta = self.all_args.prioritized_replay_beta

        # interval
        self.log_interval = self.all_args.log_interval
        self.video_interval = self.all_args.video_interval
        self.save_interval = self.all_args.save_interval
        self.verbose = 2

        # dir
        self.model_dir = self.all_args.model_dir

        self._setup_learn(config)

        egt_obs_spaces = spaces.Dict(
            {
                "n_i": spaces.MultiDiscrete(
                    [self.num_agents] * 4
                ),  # Discrete agent number
                "n_s": spaces.MultiDiscrete(
                    [2] * 4
                ),  # Discrete 2 - Coop[0], Defection[1]
                "n_r": spaces.Box(low=-4, high=4, shape=(4, 1)),
            }
        )

        # Load RL policy and trainer
        from algorithms.dqn.dqn_trainer import Strategy_DQN as DQN_Trainer
        from algorithms.dqn.policy import DQN_Policy as DQN_Policy

        # Load EGT policy, trainer and replay buffer
        from algorithms.egt.egt_trainer import EGT as EGT_Trainer
        from algorithms.egt.policy import EGT_Policy as EGT_Policy
        from utils.simple_buffer import Buffer as EGT_Buffer

        self.trainer = []
        self.buffer = []
        # get the central agent idx and nearby agent idx
        center_idx, nearby_indices = get_central_and_nearby_indices(
            self.all_args.env_dim, self.all_args.eval_dim
        )
        self.rl_agent_indices=nearby_indices

        for agent_id in range(self.num_agents):
            if np.isin(agent_id, nearby_indices):
                tr = DQN_Trainer(
                    all_args=self.all_args,
                    logger=self.logger,
                    env=self.envs,
                    gamma=self.all_args.gamma,
                    policy_class=DQN_Policy,
                    learning_rate=self.lr,
                    prioritized_replay_beta=self.beta,
                    prioritized_replay_eps=self.all_args.prioritized_replay_eps,
                    device=self.device,
                    action_flag=0 if self.all_args.train_pattern == "strategy" else 2,
                )
                bu = None
            else:
                tr = EGT_Trainer(
                    self.all_args, policy=None, device=self.device, action_flag=0
                )
                bu = EGT_Buffer(
                    self.all_args,
                    egt_obs_spaces)

            self.trainer.append(tr)
            self.buffer.append(bu)

        # Restore a pre-trained model if the model directory is specified
        self.restore()


    def _setup_learn(self, config):
        # Wandb is being used for logging
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger = None
            self.gif_dir = str(self.run_dir + "/gifs")
            self.plot_dir = str(self.run_dir + "/plots")
        else:
            #  Configure directories for logging and saving models manually
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.logger = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.gif_dir = str(self.run_dir / "gifs")
            self.plot_dir = str(self.run_dir / "plots")

        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)


    def restore(self):
        """
        Restoring Model
        """
        latest_model_file = find_latest_file(self.model_dir, "zip")
        self.load_trainer(latest_model_file)


    def load_trainer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        Load train model for each agent.

        :param path: Path to the pickled.
        """
        # Open the zip file
        with zipfile.ZipFile(path, "r") as zipf:
            # Get the list of file names in the zip file
            file_list = zipf.namelist()

            idx_list = np.arange(len(file_list)) 

            np.random.shuffle(idx_list)

            # iterate all rl agent
            for _,agent_id in enumerate(self.rl_agent_indices):
                if _>=len(file_list):
                    _=_%len(file_list)
                # Extract data for each agent
                agent_data = zipf.read(f"{_}.pt")
                # Load the data using torch.load
                pt_data = io.BytesIO(agent_data)
                state_dict = torch.load(pt_data)
                # Use the state_dict to initialize the new_model_trainer for the agent
                self.trainer[agent_id].policy.q_net.load_state_dict(state_dict)



    def _dump_logs(self, episode) -> None:
        """
        Write log.
        """
        log_info = {}
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        # Number of frames per seconds (includes time taken by gradient update)
        fps = int(
            (self.num_timesteps - self._num_timesteps_at_start)
            * self.n_rollout_threads
            / time_elapsed
        )
        log_info["time/fps"] = fps
        log_info["time/episode"] = episode

        print(
            "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.all_args.scenario_name,
                self.algorithm_name,
                self.experiment_name,
                episode,
                self.episodes,
                self.num_timesteps,
                self.num_env_steps,
                fps,
            )
        )

    def write_to_video(self, all_frames, episode):
        """
        record this episode and
        save the gif to local or wandb
        """
        import imageio

        images = []
        for png in all_frames:
            img = imageio.imread(png)
            images.append(img)
        # print(len(images))
        if self.all_args.use_wandb:
            import wandb

            imageio.mimsave(
                str(self.gif_dir) + "/episode.gif",
                images,
                duration=self.all_args.ifi,
            )
            wandb.log(
                {
                    "video": wandb.Video(
                        str(self.gif_dir) + "/episode.gif", fps=4, format="gif"
                    )
                },
                step=self.num_timesteps,
            )

        elif self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/episode_{}.gif".format(episode),
                images,
                duration=self.all_args.ifi,
            )
