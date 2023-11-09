from .base_runner import Runner
import numpy as np
import torch
import time
from stable_baselines3.common.utils import should_collect_more_steps
from utils.util import gini, consecutive_counts
import copy


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
        episode = 1
        episode_info = []
        all_frames = []
        self.br_start_idx = 0

        episode_loss = []
        episode_c_reward_during_training = []
        episode_d_reward_during_training = []
        episode_exploration_rate = []

        while self.num_timesteps < self.num_env_steps:
            num_collected_steps, num_collected_episodes = 0, 0
            while should_collect_more_steps(
                self.train_freq, num_collected_steps, num_collected_episodes
            ):
                # if one episode end
                if step >= self.episode_length:
                    # # train evey end of episode
                    # if self.all_args.algorithm_name != "DQN":

                    # alway train at end of episode
                    self.train_infos = self.train()
                    self.log_train(self.train_infos)
                    episode_loss.append(self.train_infos["train/loss"])
                    episode_c_reward_during_training.append(
                        self.train_infos["train/cooperation_reward"]
                    )
                    episode_d_reward_during_training.append(
                        self.train_infos["train/defection_reward"]
                    )

                    if self.have_train:
                        # log information
                        if episode % self.log_interval == 0 or episode == self.episodes:
                            extra_info = (
                                np.mean(episode_loss),
                                np.mean(episode_c_reward_during_training),
                                np.mean(episode_d_reward_during_training),
                                np.mean(episode_exploration_rate),
                            )
                            self.log_episode(
                                episode, self.train_infos, episode_info, extra_info
                            )

                        if self.all_args.use_render and (
                            episode % self.video_interval == 0
                            or episode == self.episodes - 1
                        ):
                            self.write_to_video(all_frames, episode)

                        self.callback.on_step()

                    num_collected_episodes += 1
                    episode += 1
                    episode_info = []
                    all_frames = []
                    episode_loss = []
                    episode_c_reward_during_training = []
                    episode_d_reward_during_training = []
                    episode_exploration_rate = []
                    step = 0
                    self.br_start_idx = self.buffer[0].step

                # record every step for current episode
                if self.all_args.use_render and (
                    episode % self.video_interval == 0
                    or episode == self.episodes - 1
                    or step + 1 == self.episode_length
                ):
                    # print('step:',step,'episode:',episode)
                    image = self.render(self.num_timesteps)
                    all_frames.append(image[0])

                # Sample actions
                infos = self.collect_rollouts()
                episode_info.append(infos)
                rollout_info = {
                    "rollout/exploration_rate": self.exploration_rate,
                    "rollout/avg_coop_for_cooperation": self.avg_strategy_coop_based[0],
                    "rollout/avg_coop_for_defection": self.avg_strategy_coop_based[1],
                    "rollout/target_update": self.target_update,
                    "rollout/step_cooperation_level": 1
                    - np.mean([info["current_cooperation"] for info in infos]),
                }
                self.log_rollout(rollout_info)
                episode_exploration_rate.append(self.exploration_rate)

                self.num_timesteps += self.n_rollout_threads
                step += 1
                num_collected_steps += 1

            if (
                self.all_args.algorithm_name == "DQN"
                and self.num_timesteps > self.learning_starts
            ):
                self.have_train = True
                self.train_infos = self.train()
                self.log_train(self.train_infos)
                episode_loss.append(self.train_infos["train/loss"])

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
        # self._total_timesteps = int(self.num_env_steps) // self.n_rollout_threads

        self.start_time = time.time_ns()

        # current timesteps for single thread
        self.num_timesteps = 0
        self._num_timesteps_at_start = self.all_args.learning_starts

    @torch.no_grad()
    def collect(self):
        """
        Collect(predict) actions according to the current step observations.
        """
        actions = []
        interactions = []
        exploration_rates = []
        strategy_coop_based = (
            []
        )  # how much previous coop make each agent have coop or defection decision

        # print('step:',step)
        for agent_id in range(self.num_agents):
            step = self.buffer[agent_id].step - 1

            # Select action randomly or according to policy
            if self.num_timesteps < self.learning_starts:
                # Warmup phase
                agent_action = np.array(
                    [
                        self.trainer[agent_id].policy.action_space.sample()
                        for _ in range(self.n_rollout_threads)
                    ]
                )
                if self.all_args.train_interaction:
                    agent_interaction = np.array(
                        [
                            self.iteract_trainer[agent_id].policy.action_space.sample()
                            for _ in range(self.n_rollout_threads)
                        ]
                    )
                # print(agent_interaction)
            else:
                agent_action = self.trainer[agent_id].predict(
                    self.buffer[agent_id].obs[step]
                )
                agent_action = _t2n(agent_action)
                # wwether
                if self.all_args.train_interaction:
                    agent_interaction = self.iteract_trainer[agent_id].predict(
                        self.buffer[agent_id].obs[step]
                    )
                    agent_interaction = _t2n(agent_interaction)

            # Calculate previous base for the current agent
            previouse_base = self._calculate_previous_base(agent_id, agent_action, step)

            strategy_coop_based.append(previouse_base)

            # convert to Numpy array with shape (n_rollout_threads,num_agents)
            actions.append(agent_action)
            self.trainer[agent_id]._update_current_progress_remaining(
                self.num_timesteps, self.num_env_steps
            )
            exploration_rate, target_update = self.trainer[agent_id]._on_step()
            exploration_rates.append(exploration_rate)

            if self.all_args.train_interaction:
                interactions.append(agent_interaction)
                self.iteract_trainer[agent_id]._update_current_progress_remaining(
                    self.num_timesteps, self.num_env_steps
                )
                _, _ = self.iteract_trainer[agent_id]._on_step()

        # Calculate the average strategy based on cooperation for all agents
        self.avg_strategy_coop_based = np.nanmean(strategy_coop_based, axis=0)

        self.exploration_rate = np.mean(np.array(exploration_rates))
        self.target_update = target_update

        return (
            np.column_stack(actions),
            np.column_stack(interactions) if self.all_args.train_interaction else None,
        )

    def _calculate_previous_base(self, agent_id, agent_action, step):
        """
        Calculate the previous base for cooperation and defection actions for a specific agent.
        """
        previouse_base = [[], []]
        # Iterate through environment indices
        for env_idx, a in enumerate(self.buffer[agent_id].obs[step]):
            act = np.array(a["n_s"])
            # Calculate cooperation proportion
            coop_proportion = (len(act) - np.count_nonzero(act)) / len(act)
            # Append cooperation proportion to the corresponding strategy
            previouse_base[agent_action[env_idx]].append(coop_proportion)

        # Calculate the average cooperation proportion for each strategy
        return [
            np.nanmean(pre_base) if len(pre_base) > 0 else np.nan
            for pre_base in previouse_base
        ]

    def insert(self, data, obs):
        """
        Inster experience data to replay buffer

        :param data: Replay data
        :next_obs: The current observation for predicting action
                   different from real_next_obs when trunction is true
        """

        real_next_obs, rewards, termination, truncation, actions, interactions = data

        # Unpack data for interaction training
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                np.array(list(obs[:, agent_id])),
                np.array(list(real_next_obs[:, agent_id])),
                rewards[:, agent_id],
                termination,
                truncation,
                actions[:, agent_id],
                interactions[:, agent_id] if self.all_args.train_interaction else [],
            )

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
        episode_rwds = []
        episode_acts = []
        episode_final_acts = []  # last 1/10 actions in this episode
        terms = []
        start_index = self.br_start_idx
        end_index = self.buffer[0].step
        # Calculate the range of indices with wrap-around
        if start_index > end_index:
            indices = list(range(start_index, self.buffer[0].buffer_size)) + list(
                range(0, end_index)
            )
        else:
            indices = list(range(start_index, end_index))
        # print(end_index)
        # print(indices)
        # print(len(indices))
        if self.algorithm_name == "DQN":
            # iterate all agents
            for br in self.buffer:
                if self.all_args.normalize_pattern == "all":
                    episode_rwds.append([br.norm_rewards[i] for i in indices])
                else:
                    episode_rwds.append([br.rewards[i] for i in indices])
                _acts = [br.actions[i] for i in indices]
                episode_acts.append(_acts)
                episode_final_acts.append(_acts[-(len(_acts) // 20) :])
                terms.append(
                    np.count_nonzero([br.termination[i] for i in indices])
                    / self.episode_length
                )
        else:
            for br in self.buffer:
                episode_rwds.append(br.norm_rewards)
                episode_acts.append(br.actions)
                terms.append(np.count_nonzero(br.termination) / br.termination.size)
        return episode_rwds, episode_acts, episode_final_acts, terms

    def log_episode(self, episode, train_infos, episode_info, extra_info):
        """
        log episode info
        """
        episode_rwds, episode_acts, episode_final_acts, terms = self.extract_buffer()

        episode_acts_copy = np.array(episode_acts).copy()
        episode_acts_flattened = [
            [np.concatenate(episode) for episode in agent]
            for agent in episode_acts_copy
        ]
        episode_acts_transposed = [
            np.array(_).T for _ in episode_acts_flattened
        ]  # Agent-Env-Epsisode
        # Calculate and print the summary result

        # print(episode_acts_flattened[-1])
        # print(episode_acts_copy[-1])
        # print(episode_acts_transposed)
        # print(len(episode_acts_transposed[0]))

        # Calculate and print the summary result
        summary_consecutive_ = [
            [consecutive_counts(env) for i, env in enumerate(agent)]
            for agent in episode_acts_transposed
        ]
        # print(summary_consecutive_)
        strategy_average_robutness = [[], []]
        strategy_best_robutness = [[], []]

        for agent_idx, agent_consecutive in enumerate(summary_consecutive_):
            for counts, total_counts, longest_consecutive in agent_consecutive:
                for target, (total, count) in total_counts.items():
                    # print(total, count)
                    average_count = total / count
                    strategy_average_robutness[target].append(average_count)
                for target, duration in longest_consecutive.items():
                    strategy_best_robutness[target].append(duration)

        self.average_robutness = [
            np.mean(strategy) for strategy in strategy_average_robutness
        ]
        self.best_robutness = [np.max(strategy) for strategy in strategy_best_robutness]

        self._dump_logs(episode)

        # payoff for cooperator and defector
        c_p = []
        d_p = []
        # print(episode_info)
        # gini_value = 0
        for infos in episode_info:
            for info in infos:
                if "cumulative_payoffs" in info:
                    gini_value = gini(info["cumulative_payoffs"])

                for _, a in enumerate(info["individual_action"]):
                    if a == 0:
                        c_p.append(info["instant_payoff"][_])
                    else:
                        d_p.append(info["instant_payoff"][_])
        # reward
        episode_coop_rewards = []
        episode_defect_rewards = []
        # print(episode_rwds)
        for r, a in zip(
            np.array(episode_rwds).flatten().round(2), np.array(episode_acts).flatten()
        ):
            if a == 0:
                episode_coop_rewards.append(r)
            else:
                episode_defect_rewards.append(r)

        train_infos["payoff/cooperation_episode_payoff"] = np.mean(c_p)
        train_infos["payoff/defection_episode_payoff"] = np.mean(d_p)
        train_infos["payoff/episode_payoff"] = np.mean(
            np.concatenate((c_p, d_p), axis=0)
        )
        train_infos["payoff/gini_coefficient "] = gini_value

        train_infos["results/coopereation_episode_rewards"] = np.mean(
            episode_coop_rewards
        )
        train_infos["results/defection_episode_rewards"] = np.mean(
            episode_defect_rewards
        )
        train_infos["results/average_episode_rewards"] = np.mean(episode_rwds)
        # print(episode_acts)
        # print(len(episode_acts))
        train_infos["results/episode_cooperation_level"] = 1 - np.mean(episode_acts)
        train_infos["results/episode_final_cooperation_performance"] = 1 - np.mean(
            episode_final_acts
        )
        train_infos["results/termination_proportion"] = np.mean(terms)

        train_infos["robutness/average_cooperation_length"] = self.average_robutness[0]
        train_infos["robutness/average_defection_length"] = self.average_robutness[1]
        train_infos["robutness/best_cooperation_length"] = int(self.best_robutness[0])
        train_infos["robutness/best_defection_length"] = int(self.best_robutness[1])

        self.print_train(train_infos, extra_info)
        self.log_train(train_infos)

    def get_env(self):
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return None
