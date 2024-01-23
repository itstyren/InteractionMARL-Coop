import wandb
import os
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
from utils.util import linear_schedule
import time,sys

class Runner(object):
    """
    Base Runner calss for RL training
    """

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
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        if self.use_linear_lr_decay:
            self.lr = linear_schedule(self.all_args.lr)
        else:
            self.lr = self.all_args.lr

        # interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        print("===================")
        print("observation_space(single): ", self.envs.observation_spaces["agent_0"])
        print("action_space(single): ", self.envs.action_spaces["agent_0"])

        # Wandb is being used for logging
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger=None
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

        # Select the training algorithm and action policy based on configuration
        if self.all_args.algorithm_name == "DQN":
            from algorithms.dqn.dqn_trainer import Strategy_DQN as TrainAlgo
            from algorithms.dqn.policy import DQN_Policy as Policy


        # Restore a pre-trained model if the model directory is specified
        if self.model_dir is not None:
            self.restore()

        # share trainer or all agents
        self.trainer = TrainAlgo(
            all_args=self.all_args,
            logger=self.logger,
            env=self.envs,
            gamma=self.all_args.gamma,
            policy_class=Policy,
            learning_rate=self.lr,
            device=self.device,
        )

        # share buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.envs.observation_spaces["agent_0"],
            self.envs.action_spaces["agent_0"],
            device=self.device,
        )

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
        pass

    def train(self):
        """
        Train policies with data in buffer.
        """
        train_infos=self.trainer.train(
            batch_size=self.all_args.mini_batch, replay_buffer=self.buffer
        )
        self.buffer.after_update()
        return train_infos
    
    def _dump_logs(self,episode) -> None:
        """
        Write log.
        """
        log_info={}
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start)*self.episode_length / time_elapsed)
        log_info['time/fps']=fps


        print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}."
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode+1,
                                self.episodes,
                                self.num_timesteps,
                                self.num_env_steps,
                                fps))

        for k, v in log_info.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(k, {k: v}, self.num_timesteps)


    def log_train(self,train_infos):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self.num_timesteps)
            else:
                self.logger.add_scalars(k, {k: v}, self.num_timesteps)
    
    def log_rollout(self,rollout_info,t):
        '''
        Log rollout info.
        '''
        for k, v in rollout_info.items():
            if self.use_wandb:
                wandb.log({k: v}, step=t)
            else:
                self.logger.add_scalars(k, {k: v}, t)