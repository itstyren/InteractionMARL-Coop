from .base_runner import Runner
import numpy as np
import torch
import time
from stable_baselines3.common.utils import should_collect_more_steps
from utils.util import gini, consecutive_counts, convert_array_to_two_arrays, save_array
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
        interaction_log = []

        while self.num_timesteps < self.num_env_steps:
            num_collected_steps, num_collected_episodes = 0, 0
            while should_collect_more_steps(
                self.train_freq, num_collected_steps, num_collected_episodes
            ):
                # print(step)
                # if one episode end
                if step >= self.episode_length:
                    # print('episode',episode)
                    if self.all_args.normalize_pattern == "episode":
                        for _, br in enumerate(self.buffer):
                            br.normalized_episode_rewards(self.episode_length)
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

                        # save model callback
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
                    episode % self.video_interval == 0 or episode == self.episodes - 1
                ):
                    # print('step:',step,'episode:',episode)
                    image, interaction_n = self.render(self.num_timesteps)
                    # print(interaction_n)
                    all_frames.append(image[0])

                    if (
                        self.have_train
                        and self.all_args.save_result
                        and step + 1 == self.episode_length
                    ):
                        interaction_log.append(interaction_n)
                        save_array(
                            interaction_log, self.plot_dir, "agent_intraction.npz"
                        )

                # Sample actions
                infos = self.collect_rollouts()
                # print(infos)

                episode_info.append(infos)
                rollout_info = {
                    "rollout/exploration_rate": self.exploration_rate,
                    "rollout/avg_coop_for_cooperation": self.avg_strategy_coop_based[0],
                    "rollout/avg_coop_for_defection": self.avg_strategy_coop_based[1],
                    "rollout/target_update": self.target_update,
                    "rollout/step_cooperation_level": np.mean(
                        [info["current_cooperation"] for info in infos]
                    ),
                    # 'rollout/ave_intreaction_for_cooperation':np.mean(
                    #     [info["current_cooperation"][0] for info in infos]
                    # ),
                    # 'rollout/ave_intreaction_for_defection':np.mean(
                    #     [info["current_cooperation"][1] for info in infos]
                    # ),
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
                # print(self.num_timesteps,episode)
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
            "====== Initial Cooperative Level {:.2f} ======".format(
                np.mean(coop_level))
        )
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].obs[0] = np.array(
                list(self.obs[:, agent_id])).copy()

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
                # print(agent_action)
                if self.all_args.train_pattern == "together":
                    agent_action, agent_interaction = convert_array_to_two_arrays(
                        agent_action
                    )
                # print(agent_action)
                # print(agent_interaction)
                # print('=====')
                if self.all_args.train_pattern == "seperate":
                    agent_interaction = np.array(
                        [
                            self.iteract_trainer[agent_id].policy.action_space.sample(
                            )
                            for _ in range(self.n_rollout_threads)
                        ]
                    )
                # print(agent_interaction)
            else:
                agent_action = self.trainer[agent_id].predict(
                    self.buffer[agent_id].obs[step]
                )
                agent_action = _t2n(agent_action)
                if self.all_args.train_pattern == "together":
                    agent_action, agent_interaction = convert_array_to_two_arrays(
                        agent_action
                    )
                # wwether
                if self.all_args.train_pattern == "seperate":
                    # print(self.buffer[agent_id].obs[step])
                    agent_interaction = self.iteract_trainer[agent_id].predict(
                        self.buffer[agent_id].obs[step]
                    )
                    agent_interaction = _t2n(agent_interaction)
                    # print(agent_interaction)

            # Calculate previous base for the current agent
            previouse_base = self._calculate_previous_base(
                agent_id, agent_action, step)

            strategy_coop_based.append(previouse_base)

            # convert to Numpy array with shape (n_rollout_threads,num_agents)
            actions.append(agent_action)
            self.trainer[agent_id]._update_current_progress_remaining(
                self.num_timesteps, self.num_env_steps
            )
            exploration_rate, target_update = self.trainer[agent_id]._on_step()
            exploration_rates.append(exploration_rate)

            if self.all_args.train_pattern == "seperate":
                self.iteract_trainer[agent_id]._update_current_progress_remaining(
                    self.num_timesteps, self.num_env_steps
                )
                _, _ = self.iteract_trainer[agent_id]._on_step()

            if self.all_args.train_pattern != "strategy":
                interactions.append(agent_interaction)
            # print(agent_interaction)    

        # Calculate the average strategy based on cooperation for all agents
        self.avg_strategy_coop_based = np.nanmean(strategy_coop_based, axis=0)

        self.exploration_rate = np.mean(np.array(exploration_rates))
        self.target_update = target_update

        return (
            np.column_stack(actions),
            np.column_stack(interactions)
            if self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate"
            else None,
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
            # previouse_base(env_idx)
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

        if self.all_args.train_seperate:
            # Combine the first values from all arrays in the list
            strategy_reward=np.concatenate([arr[..., 0].ravel() for arr in rewards])
            # Combine the second values from all arrays in the list
            interaction_reward = np.concatenate([arr[..., 1].ravel() for arr in rewards])
            # Reshape the arrays to get the desired shape
            strategy_reward = strategy_reward.reshape(len(rewards), -1)
            interaction_reward = interaction_reward.reshape(len(rewards), -1)
        else:
            strategy_reward=rewards

        # print(strategy_reward)

        # Unpack data for interaction training
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                np.array(list(obs[:, agent_id])),
                np.array(list(real_next_obs[:, agent_id])),
                strategy_reward[:, agent_id],
                interaction_reward[:, agent_id] if self.all_args.train_seperate else None,
                termination,
                truncation,
                actions[:, agent_id],
                interactions[:, agent_id]
                if self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate"
                else [],
            )

    @torch.no_grad()
    def render(self, num_timesteps, render_env=0):
        """
        Visualize the env at current state
        :param render_env: 0 render training env, 1 render eval env:
        """
        if render_env == 0:
            envs = self.envs
        else:
            envs = self.eval_envs

        image, intraction_array = envs.render("rgb_array", num_timesteps)
        # print
        return image, intraction_array

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
        # print(end_index)

        # Calculate the range of indices with wrap-around
        if start_index > end_index:
            indices = list(range(start_index, self.buffer[0].buffer_size)) + list(
                range(0, end_index)
            )
        else:
            indices = list(range(start_index, end_index))

        # print(end_index)
        # print(indices)
        # input()
        # print(len(indices))
        if self.algorithm_name == "DQN":
            # iterate all agents
            for br in self.buffer:
                if self.all_args.normalize_pattern == "all":
                    episode_rwds.append([br.norm_rewards[i] for i in indices])
                elif self.all_args.normalize_pattern == "episode":
                    # print(indices)
                    # print(br.episode_norm_rewards)
                    episode_rwds.append(
                        [br.episode_norm_rewards[i] for i in indices])
                    # print(episode_rwds)
                    # input()
                else:
                    episode_rwds.append([br.rewards[i] for i in indices])
                _acts = [br.actions[i] for i in indices]
                episode_acts.append(_acts)
                episode_final_acts.append(_acts[-(len(_acts) // 20):])
                terms.append(
                    np.count_nonzero([br.termination[i] for i in indices])
                    / self.episode_length
                )
        else:
            for br in self.buffer:
                episode_rwds.append(br.norm_rewards)
                episode_acts.append(br.actions)
                terms.append(np.count_nonzero(
                    br.termination) / br.termination.size)
        return episode_rwds, episode_acts, episode_final_acts, terms

    def log_episode(self, episode, train_infos, episode_info, extra_info):
        """
        log episode info
        """
        episode_rwds, episode_acts, episode_final_acts, terms = self.extract_buffer()
        # print(episode_acts)

        self.calculate_strategy_roubutness(np.array(episode_acts).copy())

        self._dump_logs(episode)

        # payoff for cooperator and defector
        c_p = []
        d_p = []
        # interaction ratio for cooperation and defection
        c_interaction = []
        d_interaction = []

        # print(episode_info)
        # gini_value = 0
        for infos in episode_info:
            for info in infos:
                if "cumulative_payoffs" in info:
                    gini_value = gini(info["cumulative_payoffs"])

                c_p.append(info["instant_payoff"][0])
                d_p.append(info["instant_payoff"][1])

                c_interaction.append(info["strategy_based_interaction"][0])
                d_interaction.append(info["strategy_based_interaction"][1])

        # reward
        episode_coop_rewards = []
        episode_defect_rewards = []
        # print(episode_rwds)
        for r, a in zip(
            np.array(episode_rwds).flatten().round(
                2), np.array(episode_acts).flatten()
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
        train_infos["interaction/cooperation_interaction_ratio"] = np.mean(
            c_interaction
        )
        train_infos["interaction/defection_interaction_ratio"] = np.mean(
            d_interaction)
        train_infos["interaction/average_interaction"] = np.mean(
            np.concatenate((c_interaction, d_interaction), axis=0)
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
        train_infos["results/episode_cooperation_level"] = 1 - \
            np.mean(episode_acts)
        train_infos["results/episode_final_cooperation_performance"] = 1 - np.mean(
            episode_final_acts
        )
        train_infos["results/termination_proportion"] = np.mean(terms)

        train_infos["robutness/average_cooperation_length"] = self.average_robutness[0]
        train_infos["robutness/average_defection_length"] = self.average_robutness[1]
        train_infos["robutness/best_cooperation_length"] = self.best_robutness[0]
        train_infos["robutness/best_defection_length"] = self.best_robutness[1]

        self.print_train(train_infos, extra_info)
        self.log_train(train_infos)

    def get_env(self):
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return None

    def calculate_strategy_roubutness(self, episode_acts):
        """
        :param episode_acts:
        :return:
        """
        episode_acts_flattened = [
            [np.concatenate(episode) for episode in agent] for agent in episode_acts
        ]
        episode_acts_transposed = [
            np.array(_).T for _ in episode_acts_flattened
        ]  # Agent-Env-Epsisode
        # Calculate and print the summary result
        summary_consecutive_ = [
            [consecutive_counts(env) for i, env in enumerate(agent)]
            for agent in episode_acts_transposed
        ]
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
            np.mean(strategy) / self.episode_length if strategy else 0.0
            for strategy in strategy_average_robutness
        ]
        self.best_robutness = [
            np.max(strategy) / self.episode_length if strategy else 0.0
            for strategy in strategy_best_robutness
        ]

    def eval_run(self, eval_time=6):
        """
        eval the trained model
        :param eval_time: The totoal eval trials
        """

        eval_envs = self.eval_envs
        trials = int(eval_time / self.n_rollout_threads)
        eval_scores = []
        for trial in range(trials):
            print("trail is {}".format(trial))
            self.num_timesteps = 0
            self.episodes = (
                int(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
            self.start_time = time.time_ns()
            self._num_timesteps_at_start = 0
            eval_obs, coop_level = eval_envs.reset()
            print(
                "====== Initial Cooperative Level {:.2f} ======".format(
                    np.mean(coop_level)
                )
            )
            step = 0
            episode = 1
            all_frames = []

            while self.num_timesteps < self.num_env_steps:
                actions = []
                interactions = []
                for agent_id in range(self.num_agents):
                    agent_action = self.trainer[agent_id].predict(
                        np.array(list(eval_obs[:, agent_id])))
                    agent_action = _t2n(agent_action)
                    if self.all_args.train_pattern == "together":
                        agent_action, agent_interaction = convert_array_to_two_arrays(
                            agent_action
                        )

                    actions.append(agent_action)
                    if self.all_args.train_pattern == "together":
                        interactions.append(agent_interaction)

                # print(actions,interactions)

                actions = np.column_stack(actions)
                interactions = (
                    np.column_stack(interactions)
                    if self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate"
                    else None
                )

                # print(actions)

                if (
                    self.all_args.train_pattern == "together" or self.all_args.train_pattern == "seperate"
                ):
                    combine_action = np.dstack((actions, interactions))
                    # print(combine_action)
                    (
                        eval_obs,
                        eval_rewards,
                        terminations,
                        truncations,
                        eval_infos,
                    ) = eval_envs.step(combine_action)
                else:
                    (
                        eval_obs,
                        eval_rewards,
                        terminations,
                        truncations,
                        eval_infos,
                    ) = eval_envs.step(actions)
                self.num_timesteps += self.n_rollout_threads
                step += 1

                if step >= self.episode_length:
                    self._dump_logs(episode)

                    # self._dump_logs(episode)

                    if self.all_args.use_render and (
                        episode % self.video_interval == 0
                        or episode == self.episodes - 1
                    ):
                        self.write_to_video(all_frames, episode)

                    step = 0
                    episode += 1
                    all_frames = []

                # record every step for current episode
                if self.all_args.use_render and (
                    episode % self.video_interval == 0 or episode == self.episodes - 1
                ):
                    # print('step:',step,'episode:',episode)
                    image, interaction_n = self.render(
                        self.num_timesteps, render_env=1)
                    # print(interaction_n)
                    all_frames.append(image[0])
