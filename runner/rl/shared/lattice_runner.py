from .base_runner import Runner
import numpy as np
import torch
import time


def _t2n(x):
    """
    Convert a tensor into a NumPy array.
    """
    return x.detach().cpu().numpy()


class LatticeRunner(Runner):
    """
    Runner class to perform training, evaluation. and data collection for the RL Lattice.
    See parent class for details.
    """

    def __init__(self, config):
        super(LatticeRunner, self).__init__(config)

    def run(self):
        self.warmup()
        self.start_time = time.time_ns()
        self.episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        self._total_timesteps = int(self.num_env_steps) // self.n_rollout_threads
        current_timesteps = 0
        self._num_timesteps_at_start = 0
        self.num_timesteps = 0
        for episode in range(self.episodes):
            for step in range(self.episode_length):
                # Sample actions
                actions = self.collect(step, current_timesteps)
                # Obser reward
                obs, rews,termination = self.envs.step(actions)
                data = obs, rews,termination,actions

                # insert data into buffer
                self.insert(data)
                current_timesteps += 1
                rollout_info={
                    "rollout/exploration_rate":self.exploration_rate
                }
                self.log_rollout(rollout_info,current_timesteps*self.n_rollout_threads)

            train_infos = self.train()

            # post process
            self.num_timesteps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # log information
            if episode % self.log_interval == 0 or step == self.episode_length - 1:
                self._dump_logs(episode)
                self._num_timesteps_at_start = self.num_timesteps
            train_infos["results/average_episode_rewards"]=np.mean(self.buffer.rewards)
            train_infos["results/average_cooperation_level"]=1-np.mean(self.buffer.actions)
            train_infos["results/termination_proportion"]=np.count_nonzero(self.buffer.termination)/self.buffer.termination.size
            print('='*40)
            print("|| Average Episode Rewards is {:>10.2f} ||".format(train_infos["results/average_episode_rewards"]))
            print("|| Average Coop Level is {:>15.2f} ||".format(train_infos["results/average_cooperation_level"]))
            print("|| Termination Proportion is {:>11.2f} ||".format(train_infos["results/termination_proportion"]))
            print("|| Average Train Loss is {:>15.2f} ||".format(train_infos["train/loss"]))
            print(' ','='*40,'\n')
            self.log_train(train_infos)
            

    def warmup(self):
        """
        Initial runner
        """
        # reset env
        obs, coop_level = self.envs.reset()
        print(
            "====== Initial Cooperative Level {:.2f} ======".format(np.mean(coop_level))
        )
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step, current_timesteps):
        action = self.trainer.predict(np.concatenate(self.buffer.obs[step]))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        self.trainer._update_current_progress_remaining(
            current_timesteps, self._total_timesteps
        )
        self.exploration_rate=self.trainer._on_step()
        return actions

    def insert(self, data):
        obs, rewards,termination, actions = data
        self.buffer.insert(obs, rewards,termination ,actions)
