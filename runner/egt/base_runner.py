from utils.simple_buffer import Buffer
import wandb
import os
from tensorboardX import SummaryWriter


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
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

        # interval
        self.log_interval = self.all_args.log_interval
        self.video_interval = self.all_args.video_interval

        print("===================")
        print("observation_space(single): ", self.envs.observation_spaces["agent_0"])
        print("action_space(single): ", self.envs.action_spaces["agent_0"])

        if self.all_args.algorithm_name == "EGT":
            from algorithms.egt.egt_trainer import EGT as TrainAlgo
            from algorithms.egt.policy import EGT_Policy as Policy
        # improt RL training
        if self.all_args.train_interaction:
            from algorithms.dqn.dqn_trainer import DQN_Policy as InteractAlgo
            from algorithms.dqn.policy import DQN_Policy as InteractPolicy
            if self.all_args.replay_scheme == "uniform":
                from utils.separated_buffer import SeparatedReplayBuffer as ReplayBuffer
            else:
                from utils.separated_buffer import (
                    PrioritizedReplayBuffer as ReplayBuffer,
                )

        self.policy = []
        for agent_id in range(self.num_agents):
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_spaces[f"agent_{agent_id}"],
                self.envs.action_spaces[f"agent_{agent_id}"],
                device=self.device,
            )
            self.policy.append(po)

        self._setup_learn(config)

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device,action_flag=0)
            bu = Buffer(
                self.all_args,
                self.envs.observation_spaces[f"agent_{agent_id}"],
                self.envs.action_spaces[f"agent_{agent_id}"],
            )
            self.trainer.append(tr)
            self.buffer.append(bu)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def log_train(self, train_infos, total_num_steps):
        print(
            "Coop Level {}, Average Group Reward {}".format(
                train_infos[-1].get("coop_level"),
                train_infos[-1].get("average_episode_rewards"),
            )
        )

    def _setup_learn(self, config):
        # Wandb is being used for logging
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger = None
            self.gif_dir = str(self.run_dir + "/gifs")
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

        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)

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
