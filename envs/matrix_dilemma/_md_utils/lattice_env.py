from pettingzoo import AECEnv
import functools
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.utils import seeding
import numpy as np
from gymnasium import spaces
import gymnasium
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import pdb
import wandb


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

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
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
                    spaces.Discrete(2**4),
                    spaces.Discrete(2 * (2**4)),
                ]
                # self.action_spaces[agent.name] = spaces.MultiDiscrete(
                #     np.array([2,2])
                # )

            # obs_dim_x = len(self.scenario.observation(agent, self.world))
            # obs_dim_y = len(self.scenario.observation(agent, self.world)[0])
            # set observation_spaces
            if self.args.algorithm_name == "EGT":
                self.observation_spaces[agent.name] = spaces.Dict(
                    {
                        "n_i": spaces.MultiDiscrete(
                            [len(self.agents)] * 4
                        ),  #  Discrete agent number
                        "n_s": spaces.MultiDiscrete(
                            [2] * 4
                        ),  #  Discrete 2 - Coop[0], Defection[1]
                        "n_r": spaces.Box(low=-4, high=4, shape=(4, 1)),
                    }
                )
            else:
                if self.args.train_interaction or self.args.train_pattern == "both":
                    self.observation_spaces[agent.name] = spaces.Dict(
                        {
                            "n_s": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),  #  Discrete 2 - Coop[0], Defection[1]
                            "p_a": spaces.MultiDiscrete([2] * self.args.memory_length),
                            "p_r": spaces.Box(
                                low=-5, high=5, shape=(self.args.memory_length, 1)
                            ),
                            "n_interact": spaces.MultiDiscrete(
                                np.full((4 * self.args.memory_length), 2)
                            ),  # Discrete 2 - interact 1 no_interact 0
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
                            ),  #  Discrete 2 - Coop[0], Defection[1]
                            "p_a": spaces.MultiDiscrete([2] * self.args.memory_length),
                            "p_r": spaces.Box(
                                low=-5, high=5, shape=(self.args.memory_length, 1)
                            ),
                        }
                    )
                # print(self.observation_spaces[agent.name].sample()  )

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

    def reset(self, seed=None, options="truncation"):
        """
        Resets the environment to an initial internal state

        Returns:
            an initial observation
            some auxiliary information.
        """
        if seed is not None:
            self._seed(seed=seed)

        # reset scenario (strategy and memory)
        self.scenario.reset_world(self.world)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self.current_actions = [agent.action.s for agent in self.world.agents]
        self.current_interaction = [15 for agent in self.world.agents]
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.termination_time = 0
        self.agent_selection = self._agent_selector.reset()
        if options == "truncation":
            self.steps = 0

        # set reward for initial strategy
        # obs_n, reward_n, termination, infors = self._execute_world_step()

        obs_n = []
        # Get current obs info for each agent
        for agent in self.world.agents:
            obs_n.append(self.observe(agent.name))

        # get initial cooperative level
        cl = self.state()
        # print(cl)
        return obs_n, cl

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
        instant_payoff_n = []
        # payoff_n = []  # use for logging actual reward
        action_n = []
        final_reward_n = []
        compare_reward_n = []

        # set action for all agents
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            # action_n.append(action)
            self._set_action(action, agent)

            if self.args.train_interaction or self.args.train_pattern == "both":
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
                # If rewards_pattern is not "final", update agent's reward and rewards dictionary
                agent.reward = agent_reward
                self.rewards[agent.name] = agent_reward

        # Check if comparison of rewards is enabled
        if self.args.compare_reward:
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
            if self.args.compare_reward:
                # n_r = []
                # for _ in agent.neighbours:
                #     if agent.action.s != self.world.agents[_].action.s:
                #         n_r.append(self.world.agents[_].reward)
                #     else:
                #         n_r.append(agent.reward)

                # reward = np.mean(agent.reward - np.array(n_r))
                # print(agent.reward)
                # print(self.steps,'|',agent.index,'|',agent.action.s,'|',reward,'|',agent.action.ia)
                # n_i=np.random.choice(agent.neighbours)
                # counter_reward=float(self.scenario.counter_reward(agent, self.world))
                # reward=agent.reward-counter_reward
                # print(reward)
                # reward=(agent.reward-1)/4
                # reward=agent.reward-1

                # Compare agent's reward with the average reward of the opposite strategy
                if agent.action.s == 0:
                    compare_reward_n.append(agent.reward - strategy_mean_reward[1])
                else:
                    compare_reward_n.append(agent.reward - strategy_mean_reward[0])

                # if agent.action.s != self.world.agents[n_i].action.s:
                # if agent.action.s==0:
                # print(agent.reward,self.world.agents[n_i].reward)
                # reward=(agent.reward-self.world.agents[n_i].reward)/4
                # reward=agent.reward-self.world.agents[n_i].reward
                # # if agent.action.s==0 and reward>0:
                # #     print(reward)
                # compare_reward_n.append(reward)
                # reward=agent.reward-payoff_n[n_i]
                # instant_payoff_n.append(reward)

            obs_n.append(self.observe(agent.name))

        # Check if the state is below a certain threshold for termination
        termination = False
        coop_level = self.state()
        if coop_level < 0.1:
            # print('cooperation level',coop_level)
            termination = True

        interaction_n=self.count_effective_interaction()
        # Calculate the element-wise product
        # elementwise_product = interaction_n * action_n
        
        # Calculate the average for each value (0 and 1) in the second array
        ave_interact_c = np.mean(interaction_n[np.array(action_n) == 0])
        ave_interact_d = np.mean(interaction_n[np.array(action_n) == 1])

        ave_payoff_c=np.mean(np.array(instant_payoff_n)[np.array(action_n) == 0])
        ave_payoff_d=np.mean(np.array(instant_payoff_n)[np.array(action_n) == 1])

        # Prepare info dictionary
        infos = {
            "instant_payoff": [ave_payoff_c,ave_payoff_d],
            # "individual_action": action_n,
            "current_cooperation": coop_level,
            'strategy_based_interaction':[ave_interact_c,ave_interact_d]
        }
        # if  np.any(np.isnan(compare_reward_n)):
        #     print(compare_reward_n)

        if self.args.compare_reward:
            return obs_n, compare_reward_n, termination, infos
        else:
            if self.args.rewards_pattern == "normal":
                return obs_n, instant_payoff_n, termination, infos
            else:
                return obs_n, final_reward_n, termination, infos

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
        cur_agent = self.agent_selection
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
            obs_n, reward_n, termination, infos = self._execute_world_step()
            self.steps += 1

            truncation = False
            self._accumulate_rewards()

            # if enter to final episode step
            if self.steps >= self.max_cycles:
                truncation = True
                for a in self.agents:
                    self.truncations[a] = True
                infos["final_observation"] = obs_n
                infos["cumulative_payoffs"] = list(self._cumulative_rewards.values())

            return obs_n, reward_n, termination, truncation, infos

        else:
            self._clear_rewards()

        # self._cumulative_rewards[cur_agent] = 0

    def render(self, mode, step):
        """
        render current env
        :param mode:
        :param step: current global step for all envs
        """
        self.render_mode = mode
        i_n=[]
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Define color map for rendering
        # color_set = np.array(["#0c056d",'#3c368a','#6d69a7','#9d9bc4','#cecde1','#ffe7e7','#ffd0d0','#ff7272','#ff4242', "#ff1414"])
        color_set = np.array(["#0c056d", "#ff1414"])

        # cmap = colors.ListedColormap([color_set[0], color_set[1],color_set[2],color_set[3],color_set[4],color_set[5],color_set[6],color_set[7]])
        cmap = colors.ListedColormap(np.array(["#0c056d", "red"]))
        # Create the colormap
        if self.args.train_interaction or self.args.train_pattern == "both":
            cmap_interact = colors.LinearSegmentedColormap.from_list(
                "my_list", color_set, N=10
            )
            # cmap = colors.ListedColormap([color_set[0],color_set[0], color_set[1],color_set[2],color_set[3],color_set[4],color_set[5],color_set[6],color_set[7],color_set[8]])
            # cmap_interact = colors.ListedColormap(color_set)
            # bounds = [0,1,2,3,4]
            # norm = colors.BoundaryNorm(bounds, cmap.N)

        else:
            cmap = colors.ListedColormap([color_set[0], color_set[1]])
            bounds = [0, 1, 2]
            norm = colors.BoundaryNorm(bounds, cmap.N)

        action_n = []
        interaction_n = []

        # Convert agent actions to a 2D NumPy array
        action_n = np.array(self.current_actions).reshape(
            self.scenario.env_dim, self.scenario.env_dim
        )
        if self.args.train_interaction or self.args.train_pattern == "both":
            interaction_n = self.count_effective_interaction()

            # Modify interaction_n based on action_n
            i_n = np.where(
                action_n.flatten() == 0,
                -np.maximum(interaction_n, 0.05),
                np.maximum(interaction_n, 0.05),
            )
            i_n = i_n.reshape(self.scenario.env_dim, self.scenario.env_dim)

        # Create a subplot for rendering
        if self.args.train_interaction or self.args.train_pattern == "both":
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
                "Step {}, Dilemma {}".format(step, self.world.payoff_matrix[1][0])
            )
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            im = ax.imshow(action_n, cmap=cmap, norm=norm)
            ax.axis("off")
            ax.set_title(
                "Step {}, Dilemma {}".format(step, self.world.payoff_matrix[1][0])
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

    def count_effective_interaction(self):
        """
        count effective interaction agent with their neighbour
        :return interaction_n: The effecitve ratio for every agent
        """
        interaction_n = []
        for agent in self.world.agents:
            interaction_time = 0
            for _, n_idx in enumerate(agent.neighbours):
                if (
                    agent.action.ia[_] == 1
                    and self.world.agents[n_idx].action.ia[
                        self.world.agents[n_idx].neighbours.index(agent.index)
                    ]
                    == 1
                ):
                    interaction_time += 1

            interaction_n.append(interaction_time)
        return np.array(interaction_n) / 4
