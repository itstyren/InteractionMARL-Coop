from envs.matrix_dilemma._md_utils.utils import (
    make_env,
    gen_lattice_neighbours,
    parallel_wrapper_fn,
    get_central_and_nearby_indices
)
from envs.matrix_dilemma._md_utils.lattice_env import LatticeEnv
from envs.matrix_dilemma._md_utils.scenario import BaseScenario
from envs.matrix_dilemma._md_utils.core import World, Agent
import numpy as np
import torch

class raw_env(LatticeEnv):
    """
    A Matirx game environment has gym API for soical dilemma
    """

    def __init__(self, args, max_cycles=1, continuous_actions=False, render_mode=None):
        scenario = Scenario()
        world = scenario.make_world(args)
        LatticeEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            continuous_actions=False,
            render_mode=None,
            args=args
        )
        self.metadata["name"] = "lattice_rl_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
                                          
    def make_world(self, args):
        self.env_dim=args.env_dim
        self.train_pattern=args.train_pattern
        self.init_distribution=args.init_distribution
        
        agent_num = args.env_dim**2
        world = World(np.array(args.initial_ratio), args.dilemma_strength)
        # add agent
        world.agents = [Agent(args) for i in range(agent_num)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.index = i
            # random initial strategy
            # agent.action.s = np.random.choice([0, 1], p=world.initial_ratio.ravel())

        # set neighbour index
        world.agents = gen_lattice_neighbours(world.agents)
        for agent in world.agents:
            agent.init_memory(np.array(args.initial_ratio))
        # print('Agent Reward Memory Length {}'.format(len(agent.past_reward)))
        return world

    def reset_world(self, world):
        """
        random initial strategy and memory
        """
        if self.init_distribution=='circle':
            center_idx, nearby_indices=get_central_and_nearby_indices(self.env_dim,10)

        for i, agent in enumerate(world.agents):
            if self.init_distribution=='random':
                # random initial strategy
                agent.action.s = int(
                    np.random.choice([0, 1], p=world.initial_ratio.ravel())
                )
            else:
                # Assign strategy based on distance from the center
                if np.isin(i, nearby_indices):
                    agent.action.s = 0
                else:
                    agent.action.s = 1
            # set interaction action                        
            agent.action.ia=np.random.randint(2, size=4)
            agent.init_memory(world.initial_ratio)

    def reward(self, agent, world):
        """
        calculate current reward by matrix, play with all neighbours
        :param agent (Agent obj): current selected agent in World
        :param world (World obj): World environment

        :return reward (float): accumulate reward
        """
        reward = 0.0
        for neighbout_idx, j in enumerate(agent.neighbours):
            # if world.agents[j].action.s==1:
            #     agent.action.ia[neighbout_idx]=0
            # else:
            #     agent.action.ia[neighbout_idx]=1
            if self.train_pattern == "together" or self.train_pattern == "seperate":
                if agent.action.ia[neighbout_idx]==1 and world.agents[j].action.ia[world.agents[j].neighbours.index(agent.index)]==1:
                    reward += world.payoff_matrix[agent.action.s, world.agents[j].action.s]
            else:
                reward += world.payoff_matrix[agent.action.s, world.agents[j].action.s]
        return reward
    
    def counter_reward(self, agent, world):
        """
        calculate current reward by matrix, play with all neighbours
        :param agent (Agent obj): current selected agent in World
        :param world (World obj): World environment

        :return reward (float): accumulate reward
        """
        reward = 0.0
        for j in agent.neighbours:
            # print(reward)
            reward += world.payoff_matrix[int(1-agent.action.s), world.agents[j].action.s]
        return reward



    def observation(self, agent, world):
        """
        get obs info for current agent
        :param agent (Agent obj): current selected agent in World
        :param world (World obj): World environment

        :return obs (list): current neighbour strategy list, neighbour reward list
        """
        for _,n_i in enumerate(agent.neighbours):
            # agent_idx_in_neighbour=world.agents[n_i].neighbours.index(agent.index)
            agent.neighbours_act_m[_].append(world.agents[n_i].action.s)
            # agent.neighbours_intaction_m[_].append(world.agents[n_i].action.ia[agent_idx_in_neighbour])
            # agent.intaction_m[_].append(agent.action.ia[_])


        flat_neighbours_act_m = np.concatenate([list(d) for d in agent.neighbours_act_m])
        # flat_neighbours_intaction_m=np.concatenate([list(d) for d in agent.neighbours_intaction_m])
        # flat_intaction_m=np.concatenate([list(d) for d in agent.intaction_m])

        agent.self_act_m.append(agent.action.s)
        if self.train_pattern == "together" or self.train_pattern == "seperate":
            obs={
                'n_s':flat_neighbours_act_m,
                'p_a':agent.self_act_m,
                'p_r':agent.past_reward,
                # 'n_interact':flat_neighbours_intaction_m,
                # 'p_interact':flat_intaction_m

            }
        else:
            obs={
                'n_s':flat_neighbours_act_m,
                'p_a':agent.self_act_m,
                'p_r':agent.past_reward,

            }            
        # print(obs)
        return obs
