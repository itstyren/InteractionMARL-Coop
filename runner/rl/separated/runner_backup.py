from .base_runner import Runner
import numpy as np
import torch
import time
from stable_baselines3.common.utils import should_collect_more_steps


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
        # initializing 
        self.warmup()
           
        self.have_train = False
        step = 0
        episode = 0
        episode_info = []
        all_frames = []
        while self.num_timesteps < self._total_timesteps:
            num_collected_steps, num_collected_episodes = 0, 0
            while should_collect_more_steps(
                self.train_freq, num_collected_steps, num_collected_episodes
            ):
                # record every step for current episode
                if (
                    self.num_timesteps > self.learning_starts
                    and self.all_args.use_render
                    and (
                        episode % self.video_interval == 0
                        or episode == self.episodes - 1
                    )
                ):
                    image = self.render(self.num_timesteps)
                    all_frames.append(image[0])
                
                
                # Sample actions
                infos = self.collect_rollouts()
                episode_info.append(infos)

                rollout_info = {"rollout/exploration_rate": self.exploration_rate}
                self.log_rollout(
                    rollout_info, self.num_timesteps * self.n_rollout_threads
                )

                # # record every step for current episode
                # if (
                #     self.num_timesteps > self.learning_starts
                #     and self.all_args.use_render
                #     and (
                #         episode % self.video_interval == 0
                #         or episode == self.episodes - 1
                #     )
                # ):
                #     image = self.render(self.num_timesteps)
                #     all_frames.append(image[0])

                self.num_timesteps += 1
                step += 1
                num_collected_steps += 1

                # if one episode end
                if step % self.episode_length == 0:
                    # train evey end of episode
                    if self.all_args.algorithm_name != "DQN":
                        self.train_infos = self.train()

                    # self._num_timesteps_at_start = self.num_timesteps

                    # log information
                    if self.have_train and (
                        episode % self.log_interval == 0 or episode == self.episodes - 1
                    ):  
                        self.log_episode(episode, self.train_infos, episode_info)

                    if (
                        self.have_train
                        and self.all_args.use_render
                        and (
                            episode % self.video_interval == 0
                            or episode == self.episodes - 1
                        )
                    ):
                        self.write_to_video(all_frames, episode)

                    num_collected_episodes += 1
                    episode += 1
                    episode_info = []
                    all_frames = []
                    step = 0

            if (
                self.all_args.algorithm_name == "DQN"
                and self.num_timesteps > 0
                and self.num_timesteps > self.learning_starts
            ):
                self.have_train = True
                self.train_infos = self.train()
                self.log_train(self.train_infos)

        # while episode < self.episodes:
        #     # self.warmup()
        #     all_frames = []
        #     step = 0
        #     episode_info = []

        #     # for step in self.episode_length:
        #     #     num_collected_steps, num_collected_episodes = 0, 0
        #     #     if should_collect_more_steps(
        #     #         self.train_freq, num_collected_steps, num_collected_episodes
        #     #     ):
        #     #         self.num_timesteps += 1

        #     #         num_collected_steps += 1

        #     #         if num_collected_steps % self.episode_length == 0:
        #     #             num_collected_episodes += 1

        #     while step < self.episode_length:
        #         while should_collect_more_steps(
        #             self.train_freq, num_collected_steps, num_collected_episodes
        #         ):
        #             self.num_timesteps += 1

        #             # Sample actions
        #             infos = self.collect_rollouts()
        #             episode_info.append(infos)
        #             # print(infos)

        #             rollout_info = {"rollout/exploration_rate": self.exploration_rate}
        #             self.log_rollout(
        #                 rollout_info, self.num_timesteps * self.n_rollout_threads
        #             )

        #             step += 1
        #             if step % self.episode_length == 0:
        #                 episode += 1

        #             # record every step for current episode
        #             if self.all_args.use_render and (
        #                 episode % self.video_interval == 0
        #                 or episode == self.episodes - 1
        #             ):
        #                 # print('step',step,'episode',episode)

        #                 image = self.render(self.num_timesteps)
        #                 all_frames.append(image[0])

        #             num_collected_steps += 1

        #             if num_collected_steps % self.episode_length == 0:
        #                 num_collected_episodes += 1

        #             # print(num_collected_steps,num_collected_episodes)

        #         num_collected_steps, num_collected_episodes = 0, 0
        #         if (
        #             self.all_args.algorithm_name == "DQN"
        #             and self.num_timesteps > 0
        #             and self.num_timesteps > self.learning_starts
        #         ):
        #             # print(self.num_timesteps)
        #             train_infos = self.train()
        #             self.log_train(train_infos)

        #     print(episode, len(all_frames))

        #     # train evey end of episode
        #     if self.all_args.algorithm_name != "DQN":
        #         # if True:
        #         train_infos = self.train()

        #     rwds, acts, terms = self.extract_buffer()

        #     # log information
        #     if self.num_timesteps > self.learning_starts:
        #         if episode % self.log_interval == 0 or episode == self.episodes - 1:
        #             self._dump_logs(episode)
        #             self._num_timesteps_at_start = self.num_timesteps

        #             # payoff for cooperator and defector
        #             c_p = []
        #             d_p = []
        #             for infos in episode_info:
        #                 for info in infos:
        #                     for _, a in enumerate(info["individual_action"]):
        #                         if a == 0:
        #                             c_p.append(info["instant_payoff"][_])
        #                         else:
        #                             d_p.append(info["instant_payoff"][_])
        #             # reward
        #             c_r = []
        #             d_r = []
        #             for r, a in zip(
        #                 np.array(rwds).flatten().round(2), np.array(acts).flatten()
        #             ):
        #                 if a == 0:
        #                     c_r.append(r)
        #                 else:
        #                     d_r.append(r)

        #             # print(np.array(rwds).flatten().round(2))
        #             # print(np.array(acts).flatten())
        #             # print(np.array(acts).shape,np.array(rwds).shape)

        #             train_infos["payoff/cooperation_episode_payoff"] = np.mean(c_p)
        #             train_infos["payoff/defection_episode_payoff"] = np.mean(d_p)
        #             train_infos["payoff/episode_payoff"] = np.mean(
        #                 [np.mean(c_p), np.mean(d_p)]
        #             )

        #             train_infos["results/coopereation_episode_rewards"] = np.mean(c_r)
        #             train_infos["results/average_episode_rewards"] = np.mean(d_r)
        #             train_infos["results/average_episode_rewards"] = np.mean(rwds)
        #             train_infos["results/average_cooperation_level"] = 1 - np.mean(acts)
        #             train_infos["results/termination_proportion"] = np.mean(terms)
        #             self.print_train(train_infos)
        #             self.log_train(train_infos)

        #         if self.all_args.use_render and (
        #             episode % self.video_interval == 0 or episode == self.episodes - 1
        #         ):
        #             # print('render episode',episode)
        #             self.write_to_video(all_frames, episode)

    def warmup(self):
        """
        Initial runner and  environment
        """
        # reset env
        self.obs, coop_level = self.envs.reset()
        print(
            "====== Initial Cooperative Level {:.2f} ======".format(np.mean(coop_level))
        )
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].obs[0] = np.array(list(self.obs[:, agent_id])).copy()


        # total episode num
        self.episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        # total timestep for each thread
        self._total_timesteps = int(self.num_env_steps) // self.n_rollout_threads

        self.start_time = time.time_ns()

        # current timesteps for single thread
        self.num_timesteps = 0
        self._num_timesteps_at_start = self.all_args.learning_starts

    @torch.no_grad()
    def collect(self):
        """
        Collect action according to current step obs

        """
        actions = []
        exploration_rates = []
        # print('step:',self.buffer[0].step)

        # print('step:',step)
        for agent_id in range(self.num_agents):
            step = self.buffer[agent_id].step
            # Select action randomly or according to policy
            if self.num_timesteps < self.learning_starts:
                # Warmup phase
                # action = np.array([self.trainer[agent_id].policy.action_space.sample() for _ in range(self.n_rollout_threads)])
                action = self.trainer[agent_id].predict(self.buffer[agent_id].obs[step])
                action = _t2n(action)
            else:
                action = self.trainer[agent_id].predict(self.buffer[agent_id].obs[step])
                action = _t2n(action)

            # convert to Numpy array with shape (n_rollout_threads,num_agents)
            actions.append(action)
            self.trainer[agent_id]._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )
            exploration_rates.append(self.trainer[agent_id]._on_step())

        self.exploration_rate = np.mean(np.array(exploration_rates))
        # self.trainer._on_step()
        actions = np.column_stack(actions)
        return actions

    def insert(self, data,next_obs):
        '''
        Inster experience data to replay buffer

        :param data: replay data
        :next_obs: The next observation for predicting action
                   different from real_next_obs when trunction is true
        '''
        real_next_obs, rewards, termination,truncation, actions = data
        # print('obs:',obs)
        # print('rewards:',rewards)
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                np.array(list(self.obs[:, agent_id])),
                np.array(list(real_next_obs[:, agent_id])),
                rewards[:, agent_id],
                termination,
                truncation,
                actions[:, agent_id],
            )
        self.obs=next_obs

    @torch.no_grad()
    def render(self, num_timesteps):
        """
        Visualize the env at current state
        """
        envs = self.envs
        image = envs.render("rgb_array", num_timesteps)
        return image

    def extract_buffer(self):
        """
        log episode info
        """
        rwds = []
        acts = []
        terms = []

        if self.algorithm_name == "DQN":
            for br in self.buffer:
                _idx=br.current_buffer_size()

                rwds.append(br.norm_rewards)
                acts.append(br.actions[: _idx])
                terms.append(
                    np.count_nonzero(br.termination[: _idx])
                    / br.termination[: _idx].size
                )
        else:
            for br in self.buffer:
                rwds.append(br.norm_rewards)
                acts.append(br.actions)
                terms.append(np.count_nonzero(br.termination) / br.termination.size)

        return rwds, acts, terms

    def log_episode(self, episode, train_infos, episode_info):
        """
        log episode info
        """
        rwds = []
        acts = []
        terms = []
        if self.algorithm_name == "DQN":
            for br in self.buffer:
                _idx=br.current_buffer_size()

                rwds.append(br.norm_rewards[: _idx])
                acts.append(br.actions[: _idx])
                terms.append(
                    np.count_nonzero(br.termination[: _idx])
                    / br.termination[: _idx].size
                )
        else:
            for br in self.buffer:
                rwds.append(br.norm_rewards)
                acts.append(br.actions)
                terms.append(np.count_nonzero(br.termination) / br.termination.size)

        self._dump_logs(episode)

        # payoff for cooperator and defector
        c_p = []
        d_p = []
        for infos in episode_info:
            for info in infos:
                for _, a in enumerate(info["individual_action"]):
                    if a == 0:
                        c_p.append(info["instant_payoff"][_])
                    else:
                        d_p.append(info["instant_payoff"][_])
        # reward
        c_r = []
        d_r = []
        for r, a in zip(np.array(rwds).flatten().round(2), np.array(acts).flatten()):
            if a == 0:
                c_r.append(r)
            else:
                d_r.append(r)

        # print(np.array(rwds).flatten().round(2))
        # print(np.array(acts).flatten())
        # print(np.array(acts).shape,np.array(rwds).shape)

        train_infos["payoff/cooperation_episode_payoff"] = np.mean(c_p)
        train_infos["payoff/defection_episode_payoff"] = np.mean(d_p)
        train_infos["payoff/episode_payoff"] = np.mean([np.mean(c_p), np.mean(d_p)])

        train_infos["results/coopereation_episode_rewards"] = np.mean(c_r)
        train_infos["results/average_episode_rewards"] = np.mean(d_r)
        train_infos["results/average_episode_rewards"] = np.mean(rwds)
        train_infos["results/average_cooperation_level"] = 1 - np.mean(acts)
        train_infos["results/termination_proportion"] = np.mean(terms)
        self.print_train(train_infos)
        self.log_train(train_infos)
