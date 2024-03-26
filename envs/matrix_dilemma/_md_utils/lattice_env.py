from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.utils import seeding
import numpy as np
from gymnasium import spaces
import gymnasium
from matplotlib import colors
import matplotlib.pyplot as plt
import math


class LatticeEnv(AECEnv):
    """
    A Matirx game environment has gym API for soical dilemma

    :param max_cycles: (_max_episode_steps)Maximum number of timesteps is exceeded one episode
    """

    # The metadata holds environment constants
    metadata = {
        "name": "lattice_v0",
        "is_parallelizable": True,
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        continuous_actions=False,
        render_mode=None,
        args=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed()
        self.args = args
        # (_max_episode_steps)game terminates after the number of cycles
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        # get agent index by name
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self._agent_selector = agent_selector(self.agents)

        # Convert each number to a binary representation
        num_range = np.arange(2**4)
        self.binary_interaction_matrix = (
            (num_range[:, np.newaxis] & (2 ** np.arange(4))) > 0
        ).astype(int)

        interat_dim = 1 if self.args.interact_pattern == "seperate" else 4

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self.interact_observation_spaces = dict()
        for agent in self.world.agents:
            # set action_spaces
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, dtype=np.int32
                )
            else:
                # strategy type 0 or 1
                self.action_spaces[agent.name] = [
                    spaces.Discrete(2),
                    # spaces.Discrete(2),
                    spaces.Discrete(2**interat_dim),
                    spaces.Discrete(2 * (2**4)),
                ]

            # set observation_spaces
            if self.args.algorithm_name == "EGT":
                self.observation_spaces[agent.name] = spaces.Dict(
                    {
                        "n_i": spaces.MultiDiscrete(
                            [len(self.agents)] * 4
                        ),  # Discrete agent number
                        "n_s": spaces.MultiDiscrete(
                            [2] * 4
                        ),  # Discrete 2 - Coop[0], Defection[1]
                        "n_r": spaces.Box(low=-4, high=4, shape=(4, 1)),
                    }
                )
            else:
                if self.args.train_pattern == "seperate":
                    self.observation_spaces[agent.name] = spaces.Dict(
                        {
                            "n_s": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),  # Discrete 2 - Coop[0], Defection[1]
                            "p_a": spaces.MultiDiscrete([2] * self.args.memory_length),
                        }
                    )
                    # interacion decision only need one neighbour info
                    self.interact_observation_spaces[agent.name] = spaces.Dict(
                        {
                            "n_s": spaces.MultiDiscrete(
                                np.full((interat_dim * self.args.memory_length), 2)
                            ),  # Discrete 2 - Coop[0], Defection[1]
                            "p_interact": spaces.MultiDiscrete(
                                np.full((interat_dim * self.args.memory_length), 2)
                            ),
                        }
                    )
                elif self.args.train_pattern == "together":
                    self.observation_spaces[agent.name] = spaces.Dict(
                        {
                            "n_s": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),  # Discrete 2 - Coop[0], Defection[1]
                            "p_a": spaces.MultiDiscrete([2] * self.args.memory_length),
                            "p_interact": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),
                        }
                    )

                else:
                    self.observation_spaces[agent.name] = spaces.Dict(
                        {
                            "n_s": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),  # Discrete 2 - Coop[0], Defection[1]
                            "p_a": spaces.MultiDiscrete([2] * self.args.memory_length),
                        }
                    )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        """
        Returns:
            random number generator and seed
        """
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        """
        observation current scenario info
        """
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def interact_observe(self, agent):
        """
        observation current scenario info
        """
        return self.scenario.interact_observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def reset(self, seed=None, options="truncation"):
        """
        Resets the environment to an initial internal state

        Returns:
            an initial observation
            some auxiliary information.
        """
        self._seed(seed=self.args.seed)

        # reset scenario (strategy and memory)
        self.scenario.reset_world(self.world)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self.current_actions = [agent.action.s for agent in self.world.agents]
        # 15 mean interact with all neighbour
        self.current_interaction = [agent.action.ia for agent in self.world.agents]
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.termination_time = 0
        self.agent_selection = self._agent_selector.reset()
        if options == "truncation":
            self.steps = 0

        obs_n = []
        interact_obs_n = []
        # Get current obs info for each agent
        for agent in self.world.agents:
            obs_n.append(self.observe(agent.name))
            if self.args.train_pattern == "seperate":
                interact_obs_n.append(self.interact_observe(agent.name))

        # get initial cooperative level
        cl = self.state()
        # print(cl)
        return obs_n, interact_obs_n, cl

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def _execute_world_step(self):
        """
        Set agent action to env and get rewards.

        :return observation: The [index,strategy,payoff] for self agent and its neighbour
        """
        # record observations for each agent
        obs_n = []
        i_obs_n = []
        instant_payoff_n = []
        # payoff_n = []  # use for logging actual reward
        action_n = []
        final_reward_n = []
        compare_reward_n = []
        interact_reward_n = []

        # set action for all agents
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            # action_n.append(action)
            self._set_action(action, agent)

            if (
                self.args.train_pattern == "together"
                or self.args.train_pattern == "seperate"
            ):
                interaction = self.current_interaction[i]
                self._set_interaction(interaction, agent)

            action_n.append(action)

        # no action for this env
        self.world.step()

        # Access reward
        for agent in self.world.agents:
            # Calculate instant payoff
            agent_reward = float(self.scenario.reward(agent, self.world))
            instant_payoff_n.append(agent_reward)

            # Calculate and update final reward if rewards_pattern is set to "final"
            if self.args.rewards_pattern == "final":
                # Append the current agent_reward to the past_reward list
                agent.past_reward.append(agent_reward)

                # Copy the past_reward list for further calculations
                reward_memory = agent.past_reward.copy()
                reversed_reward_memory = np.array(reward_memory)[::-1]

                # Initialize variables for decayed reward calculation
                decay_r = 0.0
                weight_r = 0
                # Iterate over reversed_reward_memory to calculate decayed reward and weight
                for _, r in enumerate(reversed_reward_memory):
                    _ += 1
                    if r != -99:
                        decay_r += agent.memory_alpha**_ * r
                        weight_r += agent.memory_alpha**_

                final_reward = (agent_reward + decay_r) / (1 + weight_r)

                # Append the final_reward to the past_reward list
                agent.past_reward.append(final_reward)

                # Update agent's reward and the rewards dictionary
                agent.reward = final_reward
                self.rewards[agent.name] = final_reward
                final_reward_n.append(final_reward)
            else:
                # If rewards_pattern is not "final", update agent's reward and rewards dictionary normally
                agent.reward = agent_reward
                self.rewards[agent.name] = agent_reward

        if self.args.comparison_benchmarks=='selfishness':
            for agent in self.world.agents:
                other_rewards = (sum(self.rewards.values())-agent.reward)
                agent.reward= agent_reward+self.world.selfishness*other_rewards
                self.rewards[agent.name] = agent.reward
        elif self.args.comparison_benchmarks=='svo':
            for agent in self.world.agents:
                # arithmetic mean
                other_rewards = (sum(self.rewards.values())-agent.reward)/(len(self.world.agents)-1)
                reward_angle=math.atan2(other_rewards,agent.reward)

                # Angle in degrees
                angle_deg = 45
                # Convert angle to radians
                angle_rad = math.radians(angle_deg)
                # Compute tangent
                tan_value = math.tan(angle_rad)
                agent.reward= agent.reward-0.1*abs(tan_value-reward_angle)
                self.rewards[agent.name] = agent.reward

        
        
        
        # Check if comparison of rewards is enabled
        if self.args.compare_reward_pattern == "all":
            # Initialize a list to store rewards for each strategy
            strategy_reward = [[], []]

            # Iterate through agents to categorize rewards based on their actions
            for idx, agent in enumerate(self.world.agents):
                strategy_reward[agent.action.s].append(agent.reward)
            # Calculate the mean reward for each strategy
            strategy_mean_reward = [
                np.nanmean(s_r) if s_r else 0 for s_r in strategy_reward
            ]

        # Get current obs info for each agent
        for agent in self.world.agents:
            if self.args.seperate_interaction_reward:
                interact_reward_n.append(agent.reward)

            if self.args.compare_reward_pattern != "none":
                if self.args.compare_reward_pattern == "neighbour":
                    strategy_reward = [[], []]
                    for idx, neighbour_idx in enumerate(agent.neighbours):
                        strategy_reward[
                            self.world.agents[neighbour_idx].action.s
                        ].append(self.world.agents[neighbour_idx].reward)
                    strategy_mean_reward = [
                        np.nanmean(s_r) if s_r else 0 for s_r in strategy_reward
                    ]

                # count neighbour strategy
                dim_lengths = [0, 0]
                for _, neighbour_idx in enumerate(agent.neighbours):
                    dim_lengths[self.world.agents[neighbour_idx].action.s] += 1

                # add self strategy into count
                dim_lengths[agent.action.s] += 1

                ratios = [element / np.sum(dim_lengths) for element in dim_lengths]

                if agent.action.s == 0:
                    agent_strategy_reward = (
                        agent.reward * ratios[0] - strategy_mean_reward[1] * ratios[1]
                    )
                else:
                    agent_strategy_reward = (
                        agent.reward * ratios[1] - strategy_mean_reward[0] * ratios[0]
                    )

                compare_reward_n.append(agent_strategy_reward)

            obs_n.append(self.observe(agent.name))
            if self.args.train_pattern == "seperate":
                i_obs_n.append(self.interact_observe(agent.name))

        # Check if the state is below a certain threshold for termination
        termination = False
        coop_level = self.state()
        if coop_level < 0.05 and self.render_mode == "train":
            # print('cooperation level',coop_level)
            termination = True

        interaction_n = self.count_interacted_time()
        effecitve_interaction_n,average_intensity,average_link=self.count_effective_interaction()
        ave_effective_ic=np.mean(effecitve_interaction_n[np.array(action_n) == 0])
        ave_effective_id=np.mean(effecitve_interaction_n[np.array(action_n) == 1])

        # Calculate the average conntected time by neighbour for each strategy
        ave_interact_c = np.mean(interaction_n[np.array(action_n) == 0])
        ave_interact_d = np.mean(interaction_n[np.array(action_n) == 1])

        ave_payoff_c = np.mean(np.array(instant_payoff_n)[np.array(action_n) == 0])
        ave_payoff_d = np.mean(np.array(instant_payoff_n)[np.array(action_n) == 1])

        # Prepare info dictionary
        infos = {
            "instant_payoff": [ave_payoff_c, ave_payoff_d],
            "current_cooperation": coop_level,
            "strategy_based_interaction": [ave_interact_c, ave_interact_d],
            'average_intensity':average_intensity,
            'average_link':average_link,
            'effective_interaction':[ave_effective_ic,ave_effective_id]
        }

        if self.args.compare_reward_pattern != "none":
            if self.args.seperate_interaction_reward:
                combined_reward_n = np.column_stack(
                    (compare_reward_n, interact_reward_n)
                )
                return obs_n, i_obs_n, combined_reward_n, termination, infos
            else:
                return obs_n, i_obs_n, compare_reward_n, termination, infos
        else:
            if self.args.rewards_pattern == "normal":
                if self.args.seperate_interaction_reward:
                    combined_reward_n = np.column_stack(
                        (instant_payoff_n, interact_reward_n)
                    )
                    return obs_n, i_obs_n, combined_reward_n, termination, infos
                else:
                    return obs_n, i_obs_n, instant_payoff_n, termination, infos
            else:
                if self.args.seperate_interaction_reward:
                    combined_reward_n = np.column_stack(
                        (final_reward_n, interact_reward_n)
                    )
                    return obs_n, i_obs_n, combined_reward_n, termination, infos
                else:
                    return obs_n, i_obs_n, final_reward_n, termination, infos

    def _set_action(self, action, agent):
        """Set environment action for a particular agent."""
        agent.action.s = action

    def _set_interaction(self, interaction, agent):
        """Set environment interaction action for a particular agent."""
        agent.action.ia = self.binary_interaction_matrix[interaction]

    def step(self, action):
        """
        Contains most of the logic of your environment
        Automatically switches control to the next agent.

        Returns:
            the 5-tuple (observation, reward, terminated, truncated, info)
        """
        # get current agent index
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        # set agent_selection to next agent
        self.agent_selection = self._agent_selector.next()
        # set action for current agent
        if isinstance(action, np.ndarray):
            self.current_actions[current_idx] = action[0]
            self.current_interaction[current_idx] = action[1]
        else:
            self.current_actions[current_idx] = action

        # do _execute_world_step only when all agent have go through step once
        # clear reward for all agent first
        if next_idx == 0:
            # get obs after take action
            obs_n, i_obs_n, reward_n, termination, infos = self._execute_world_step()
            self.steps += 1

            truncation = False
            self._accumulate_rewards()

            # if enter to final episode step
            # print(self.steps,self.max_cycles)
            if self.steps >= self.max_cycles:
                truncation = True
                for a in self.agents:
                    self.truncations[a] = True
                infos["final_observation"] = obs_n
                infos["final_i_observation"] = i_obs_n
                infos["cumulative_payoffs"] = list(self._cumulative_rewards.values())

            return obs_n, i_obs_n, reward_n, termination, truncation, infos

        else:
            self._clear_rewards()

    def render(self, mode, step):
        """
        render current env
        :param mode:
        :param step: current global step for all envs
        """
        self.render_mode = mode
        i_n = []
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Define color map for rendering
        color_set = np.array(["#0c056d", "#eaeaea", "#ff1414"])

        cmap = colors.ListedColormap(np.array(["#0c056d", "red"]))
        # Create the colormap
        if (
            self.args.train_pattern == "together"
            or self.args.train_pattern == "seperate"
        ):
            cmap_interact = colors.LinearSegmentedColormap.from_list(
                "my_list", color_set, N=9
            )

        else:
            cmap = colors.ListedColormap(np.array(["#0c056d", "red"]))
            bounds = [0, 1, 2]
            norm = colors.BoundaryNorm(bounds, cmap.N)

        action_n = []
        interaction_n = []

        # Convert agent actions to a 2D NumPy array
        action_n = np.array(self.current_actions).reshape(
            self.scenario.env_dim, self.scenario.env_dim
        )
        if (
            self.args.train_pattern == "together"
            or self.args.train_pattern == "seperate"
        ):
            interaction_n = self.count_interacted_time()

            # Modify interaction_n based on action_n
            i_n = np.where(
                action_n.flatten() == 0,
                -np.maximum(interaction_n, 0.05),
                np.maximum(interaction_n, 0.05),
            )
            i_n = i_n.reshape(self.scenario.env_dim, self.scenario.env_dim)

        # Create a subplot for rendering
        if (
            self.args.train_pattern == "together"
            or self.args.train_pattern == "seperate"
        ):
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            for idx, ax in enumerate(axs.flat):
                # im = ax.imshow(action_n, cmap=cmap, alpha=scaled_interaction_n,norm=norm)
                if idx == 0:
                    im = ax.imshow(action_n, cmap=cmap)
                else:
                    im = ax.imshow(i_n, cmap=cmap_interact, vmin=-1, vmax=1)
                    fig.colorbar(im, ax=ax)
                ax.axis("off")
            fig.suptitle(
                "Mode {}, Step {}, Dilemma {}".format(
                    mode, step, self.world.payoff_matrix[1][0]
                )
            )
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            im = ax.imshow(action_n, cmap=cmap, norm=norm)
            ax.axis("off")
            ax.set_title(
                "Mode {}, Step {}, Dilemma {}".format(
                    mode, step, self.world.payoff_matrix[1][0]
                )
            )
            # Configure plot aesthetics

        fig.tight_layout()

        # Save the plot as an image in memory
        import io

        with io.BytesIO() as buffer:  # use buffer memory
            plt.savefig(buffer, format="png", dpi=plt.gcf().dpi)
            buffer.seek(0)
            image = buffer.getvalue()
            buffer.flush()

        # plt.show()
        # Close the figure to avoid warning
        plt.close()
        return image, i_n

    def state(self) -> np.ndarray:
        """
        return the cooperative state for whole system
        """
        coop_level = np.mean(self.current_actions)
        return 1 - coop_level

    def count_interacted_time(self):
        """
        count how many time agent connected by neighbour
        :return interaction_n: The effecitve ratio for every agent
        """
        interaction_n = []
        for agent in self.world.agents:
            # be connected time by neighbour
            interaction_time = 0
            for _, n_idx in enumerate(agent.neighbours):
                if (
                    self.world.agents[n_idx].action.ia[
                        self.world.agents[n_idx].neighbours.index(agent.index)
                    ]
                    == 1
                ):
                    interaction_time += 1

            interaction_n.append(interaction_time)
        return np.array(interaction_n) / 4

    def count_effective_interaction(self):
        """
        count the effective interaction
        connected to neighbour, and neighbour connected to self
        """
        # link-strategy configuration list
        # store CC CD(or DC) DD
        l_s_list = np.zeros(3)
        # store actual interaction
        actual_l_s_list = np.zeros(3)


        interaction_n = []
        for agent in self.world.agents:
            # be connected time by neighbour
            interaction_time = 0
            for _, n_idx in enumerate(agent.neighbours):
                # strategy set can be 0 1 2
                # denote as DD DC/CD CC
                l_s_list[agent.action.s+self.world.agents[n_idx].action.s]+=1

                if (
                    agent.action.ia[_] == 1
                    and self.world.agents[n_idx].action.ia[
                        self.world.agents[n_idx].neighbours.index(agent.index)
                    ]
                    == 1
                ):
                    interaction_time += 1
                    # actual interaction
                    actual_l_s_list[agent.action.s+self.world.agents[n_idx].action.s]+=1
            average_intensity = np.divide(
                actual_l_s_list,
                l_s_list,
                out=np.zeros_like(actual_l_s_list),
                where=l_s_list != 0,
            )
            average_link=np.divide(
                l_s_list,
                np.sum(l_s_list),
            )
            interaction_n.append(interaction_time)
        return np.array(interaction_n) / 4,average_intensity,average_link