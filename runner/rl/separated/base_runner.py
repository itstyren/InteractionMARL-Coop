import wandb
import io
import os
from tensorboardX import SummaryWriter
from utils.util import linear_schedule_to_0,linear_schedule_to_1, round_up, FileManager, find_latest_file
import time, sys
import torch
import numpy as np
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    TrainFreq,
    TrainFrequencyUnit,
)
from typing import (
    Union,
)
import pathlib
from stable_baselines3.common.callbacks import ConvertCallback
from utils.callback import CheckpointCallback, BaseCallback
from torchinfo import summary
from stable_baselines3.common.save_util import load_from_pkl
import zipfile


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
        self.eval_envs = config['eval_envs']
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads=self.all_args.n_eval_rollout_threads
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.env_name = self.all_args.env_name
        self.use_wandb = self.all_args.use_wandb

        self.learning_starts = self.all_args.learning_starts

        # Save train freq parameter, will be converted later to TrainFreq object
        train_freq = round_up(self.all_args.train_freq / self.n_rollout_threads, 0)
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        # set learning rate (schedule or float)
        if self.all_args.use_linear_lr_decay:
            self.lr = linear_schedule_to_0(self.all_args.lr)
        else:
            self.lr = self.all_args.lr
        if self.all_args.use_linear_beta_growth:
            self.beta = linear_schedule_to_1(self.all_args.prioritized_replay_beta)
        else:
            self.beta = self.all_args.prioritized_replay_beta

        # interval
        self.log_interval = self.all_args.log_interval
        self.video_interval = self.all_args.video_interval
        self.save_interval = self.all_args.save_interval
        self.verbose = 2
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval


        # dir
        self.model_dir = self.all_args.model_dir

        print("===================")
        if self.all_args.train_pattern =='seperate':
            print("Strategy observation_space: ", self.envs.observation_spaces["agent_0"])
            print("Interaction observation_space: ", self.envs.interact_observation_spaces["agent_0"])
            print("Strategy action_space: ", self.envs.action_spaces["agent_0"][0])
            print("Interaction action_space: ", self.envs.action_spaces["agent_0"][1])
        else:
            print("observation_space(together): ", self.envs.observation_spaces["agent_0"])
            if self.all_args.train_pattern =='together':
                print("action_space(together): ", self.envs.action_spaces["agent_0"][2])
            else:  # train_pattern =='strategy'
                print("action_space(strategy): ", self.envs.action_spaces["agent_0"][0])


        # Select the training algorithm and action policy based on configuration
        if self.all_args.algorithm_name == "DQN":
            from algorithms.dqn.dqn_trainer import Strategy_DQN as TrainAlgo
            from algorithms.dqn.policy import DQN_Policy as Policy

            if self.all_args.replay_scheme == "uniform":
                from utils.separated_buffer import SeparatedReplayBuffer as ReplayBuffer
            else:
                from utils.separated_buffer import (
                    PrioritizedReplayBuffer as ReplayBuffer,
                )
        else:
            from algorithms.dqn.dqn_trainer import Strategy_DQN as TrainAlgo
            from algorithms.dqn.policy import DQN_Policy as Policy
            from utils.separated_buffer import SeparatedRolloutBuffer as ReplayBuffer

        self._setup_learn(config)

        self.trainer = []
        self.iteract_trainer=[]
        self.buffer = []
        self.interact_buffer = []

        # even policy load from file, still need initial trainer first 
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(
                all_args=self.all_args,
                logger=self.logger,
                env=self.envs,
                gamma=self.all_args.gamma,
                policy_class=Policy,
                learning_rate=self.lr,
                prioritized_replay_beta=self.beta,
                prioritized_replay_eps=self.all_args.prioritized_replay_eps,
                exploration_fraction=self.all_args.exploration_fraction,
                exploration_final_eps=self.all_args.strategy_final_exploration,
                device=self.device,
                action_flag=0 if self.all_args.train_pattern=='strategy' or self.all_args.train_pattern=='seperate' else 2
            )
            self.trainer.append(tr)

            if self.all_args.train_pattern == "seperate":
                iteract_tr=TrainAlgo(
                    all_args=self.all_args,
                    logger=self.logger,
                    env=self.envs,
                    gamma=self.all_args.gamma,
                    policy_class=Policy,
                    learning_rate=self.lr,
                    prioritized_replay_beta=self.beta,
                    prioritized_replay_eps=self.all_args.prioritized_replay_eps,
                    exploration_fraction=self.all_args.exploration_fraction,
                    exploration_final_eps=self.all_args.insteraction_final_exploration,
                    device=self.device,
                    action_flag=1
                )
                self.iteract_trainer.append(iteract_tr)

            
            

        # Restore a pre-trained model if the model directory is specified
        have_load_buffer=False
        if self.model_dir is not None:
            have_load_buffer=self.restore()
        
        # The eval process doesnt need replay buffer
        if not have_load_buffer:
            for agent_id in range(self.num_agents):
                bu = ReplayBuffer(
                    self.all_args,
                    self.envs.observation_spaces["agent_{}".format(agent_id)],
                    device=self.device,
                )
                self.buffer.append(bu)

                if self.all_args.train_pattern == "seperate":
                    interact_bu=ReplayBuffer(
                    self.all_args,
                    self.envs.interact_observation_spaces["agent_{}".format(agent_id)],
                    device=self.device,
                )
                    self.interact_buffer.append(interact_bu)
                
        
        print("\nReport Model Structure...")
        tensor_date = {}
        for key, value in self.envs.observation_spaces["agent_0"].items():
            tensor_date[key] = torch.tensor(value.sample(), device=self.device)
        summary(self.trainer[0].policy.q_net, input_data=[tensor_date])
        if self.all_args.train_pattern == "seperate":
            tensor_date = {}
            for key, value in self.envs.interact_observation_spaces["agent_0"].items():
                tensor_date[key] = torch.tensor(value.sample(), device=self.device)
            # print(tensor_date)
            summary(self.iteract_trainer[0].policy.q_net, input_data=[tensor_date])
        print("\nStrat Training...\n")

        # setup callback function
        callback = None
        # print(self.save_dir)
        if self.save_interval > 0:
            callback = CheckpointCallback(
                save_freq=self.save_interval,
                save_path=self.save_dir,
                name_prefix=self.experiment_name,
                save_replay_buffer=self.all_args.save_replay_buffer,
                max_files=self.all_args.max_files,
                verbose=2,
            )
            self.buffer_manager = FileManager(
                self.save_dir, max_files=self.all_args.max_files, suffix="pkl"
            )

        # Create eval callback if needed
        self.callback = self._init_callback(callback)

    def _setup_learn(self, config):
        # Wandb is being used for logging
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger = None
            self.gif_dir = str(self.run_dir + "/gifs")
            self.plot_dir=str(self.run_dir + "/plots")
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
            self.plot_dir=str(self.run_dir / "plots")

        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def restore(self):
        """
        Restoring Model
        """
        have_load_buffer=False
        latest_model_file = find_latest_file(self.model_dir, "zip")
        self.load_trainer(latest_model_file)

        if any(file.endswith(".pkl") for file in os.listdir(self.model_dir)):
            latest_buffer_file = find_latest_file(self.model_dir, "pkl")
            self.load_replay_buffer(latest_buffer_file)
            have_load_buffer=True
        elif self.all_args.eval_mode:
            have_load_buffer=True
        return have_load_buffer


    def collect_rollouts(self):
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        """
        # Sample actions
        actions,interactions = self.collect()
        # print(actions,interactions)
        
        # one step to environment
        if self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate":
            combine_action=np.dstack((actions, interactions))
            next_obs,i_next_obs, rews, terminations, truncations, infos = self.envs.step(combine_action)
        else:
            next_obs, i_next_obs,rews, terminations, truncations, infos = self.envs.step(actions)

        
        # handle `final_observation` for trunction
        # where the next_obs for calculate TD Q-value is differnt then predict action
        real_next_obs = next_obs.copy()
        
        real_next_i_obs = i_next_obs.copy()

        for idx, trunc in enumerate(truncations):
            if trunc:
                # print(next_obs[idx])
                # print(infos[idx]["final_observation"])
                real_next_obs[idx] = infos[idx]["final_observation"]
                if self.all_args.train_pattern=='seperate':
                    real_next_i_obs[idx] = infos[idx]["final_i_observation"]
                # print(real_next_obs[idx])
                # print('===============\n')

        data = real_next_obs,real_next_i_obs, rews, terminations, truncations, actions,interactions

        # insert data into buffer
        self.insert(data, self.obs,self.interact_obs)
        self.obs = next_obs.copy()
        self.interact_obs = i_next_obs.copy()

        return infos

    def train(self):
        """
        Train policies with data in buffer.
        """
        # print(self.num_timesteps)
        train_infos = []
        for agent_id in torch.randperm(self.num_agents):
            if self.all_args.train_pattern == 'together':
                ti = self.trainer[agent_id].train(
                batch_size=self.all_args.mini_batch, replay_buffer=self.buffer[agent_id],action_flag=2
            )
            else: # only train strategy
                ti = self.trainer[agent_id].train(
                    batch_size=self.all_args.mini_batch, replay_buffer=self.buffer[agent_id],action_flag=0
                )
            if self.all_args.train_pattern == 'seperate': # train anohter interaction model
                self.iteract_trainer[agent_id].train(
                    batch_size=self.all_args.mini_batch, replay_buffer=self.interact_buffer[agent_id],action_flag=1
                )
            if self.all_args.algorithm_name != "DQN":
                # if True:
                self.buffer[agent_id].after_update()
            train_infos.append(ti)

        loss_values = np.array(
            [entry["train/loss"] for entry in train_infos if "train/loss" in entry]
        )
        coop_reward = np.array(
            [entry["train/cooperation_reward"] for entry in train_infos], dtype=float
        )
        defect_reward = np.array(
            [entry["train/defection_reward"] for entry in train_infos], dtype=float
        )
        ti = {
            "train/loss": np.mean(loss_values),
            "train/n_updates": ti["train/n_updates"],
            "train/lr": ti["train/lr"],
            "train/prioritized_replay_beta": ti["train/prioritized_replay_beta"],
            "train/cooperation_reward": np.nanmean(coop_reward),
            "train/defection_reward": np.nanmean(defect_reward),
        }
        return ti

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

        for k, v in log_info.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(
                    k, {k: v}, self.num_timesteps * self.n_rollout_threads
                )

    def log_train(self, train_infos):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        # print(train_infos)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(
                    k, {k: v}, self.num_timesteps * self.n_rollout_threads
                )

    def log_eval(self, train_infos):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        # print(train_infos)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(
                    k, {k: v}, self.num_timesteps * self.n_eval_rollout_threads
                )


    def log_rollout(self, rollout_info):
        """
        Log rollout info.
        """
        for k, v in rollout_info.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(k, {k: v}, self.num_timesteps)

    def print_train(self, train_infos, extra_info):
        """
        print train info
        """
        (
            episode_loss,
            cooperation_reward_during_training,
            defection_reward_during_training,
            episode_exploration_rate,
        ) = extra_info
        print("-" * 44)
        print("| Payoff/ {:>33}|".format(" " * 10))
        print(
            "|    Cooperation Episode Payoff  {:>9.4f} |".format(
                train_infos["payoff/cooperation_episode_payoff"]
            )
        )
        print(
            "|    Defection Episode Payoff  {:>11.4f} |".format(
                train_infos["payoff/defection_episode_payoff"]
            )
        )
        print(
            "|    Average Episode Payoff  {:>13.4f} |".format(
                train_infos["payoff/episode_payoff"]
            )
        )
        print("| Reward/ {:>32} |".format(" " * 10))
        print(
            "|    Cooperation Episode Rewards  {:>8.4f} |".format(
                train_infos["results/coopereation_episode_rewards"]
            )
        )
        print(
            "|    Defection Episode Rewards  {:>10.4f} |".format(
                train_infos["results/defection_episode_rewards"]
            )
        )
        print("| Train/ {:>34}|".format(" " * 10))
        print(
            "|    Average Coop Level  {:>17.2f} |".format(
                train_infos["results/episode_cooperation_level"]
            )
        )
        print(
            "|    Final Coop Level  {:>19.2f} |".format(
                train_infos["results/episode_final_cooperation_performance"]
            )
        )
        print(
            "|    Termination Proportion  {:>13.2f} |".format(
                train_infos["results/termination_proportion"]
            )
        )
        print(
            "|    Average Coop Reward  {:>16.4f} |".format(
                cooperation_reward_during_training
            )
        )
        print(
            "|    Average Defect Reward  {:>14.4f} |".format(
                defection_reward_during_training
            )
        )
        print("|    Average Train Loss  {:>17.2f} |".format(episode_loss))
        print(
            "|    Average Exploration Rate  {:>11.2f} |".format(
                episode_exploration_rate
            )
        )
        print("|    n_updates  {:>26.0f} |".format(train_infos["train/n_updates"]))
        print("|    Learning Rate  {:>22.2f} |".format(train_infos["train/lr"]))
        print(
            "|    Prioritized Replay Beta  {:>12.2f} |".format(
                train_infos["train/prioritized_replay_beta"]
            )
        )
        print("| Robutness/ {:>30}|".format(" " * 10))
        print(
            "|    Average Coop Robutness  {:>13.2f} |".format(
                train_infos["robutness/average_cooperation_length"]
            )
        )
        print(
            "|    Average Defection Robutness  {:>8.2f} |".format(
                train_infos["robutness/average_defection_length"]
            )
        )
        print(
            "|    Best Cooperation Robutness  {:>9.2f} |".format(
                train_infos["robutness/best_cooperation_length"],
            )
        )
        print(
            "|    Best Defection Robutness  {:>11.2f} |".format(
                train_infos["robutness/best_defection_length"]
            )
        )        
        print("-" * 44, "\n")

    def write_to_video(self, all_frames, episode,video_type='train'):
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
                    video_type: wandb.Video(
                        str(self.gif_dir) + "/episode.gif", fps=4, format="gif"
                    )
                },
                step=self.num_timesteps,
            )

        elif self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/{}_episode_{}.gif".format(video_type,episode),
                images,
                duration=self.all_args.ifi,
            )

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(
                    f"The frequency of `train_freq` must be an integer and not {train_freq[0]}"
                )

            self.train_freq = TrainFreq(*train_freq)

    def _init_callback(
        self,
        callback: MaybeCallback,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback

    def save_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.buffer is not None, "The replay buffer is not defined"
        self.buffer_manager.create_file(self.buffer, path)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """
        self.buffer = load_from_pkl(path, self.verbose)
        for b in self.buffer:
            b.device = self.device

    def load_trainer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        Load train model for each agent.

        :param path: Path to the pickled.
        """
        # np.random.seed(self.all_args.seed)
        # Open the zip file
        with zipfile.ZipFile(path, "r") as zipf:
            idx_list = np.arange(self.num_agents) 
            # Using shuffle() method 
            if self.all_args.eval_mode:
                np.random.shuffle(idx_list)
            for agent_id in idx_list:
                # Extract data for each agent
                agent_data = zipf.read(f"{agent_id}.pt")
                # Load the data using torch.load
                pt_data = io.BytesIO(agent_data)
                state_dict = torch.load(pt_data)
                # Use the state_dict to initialize the new_model_trainer for the agent
                self.trainer[agent_id].policy.q_net.load_state_dict(state_dict)
