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
        self.metadata["name"] = "lattice_egt_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, args):
        self.env_dim=args.env_dim
        self.init_distribution=args.init_distribution

        agent_num = args.env_dim**2
        
        world = World(np.array(args.initial_ratio), args.dilemma_strength)
        # add agent
        world.agents = [Agent(args) for i in range(agent_num)]
        

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.index = i

            # agent.action.s = np.random.choice([0, 1], p=world.initial_ratio.ravel())

        # set neighbour index
        world.agents = gen_lattice_neighbours(world.agents)
        for agent in world.agents:
            agent.init_memory(np.array(args.initial_ratio))
        return world

    def reset_world(self, world):

        if self.init_distribution=='circle':
            center_idx, nearby_indices=get_central_and_nearby_indices(self.env_dim,10)
        # print(nearby_indices)

        # random initial strategy
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


    def reward(self, agent, world):
        """
        calculate current reward by matrix
        PD game with all neighbours
        """
        reward = 0.0
        for j in agent.neighbours:
            # print(reward)
            reward += world.payoff_matrix[agent.action.s, world.agents[j].action.s]
            # print(agent.action.s, world.agents[j].action.s)
            # print('========',reward)
        # reward += world.payoff_matrix[agent.action.s, agent.action.s]
        return reward

    def observation(self, agent, world):
        """
        get obs info for current agent
        self and neighbours infos: index,strategy,payoff
        """
        # strategy_set = np.zeros(len(agent.neighbours))
        # payoff_set=np.zeros(len(agent.neighbours))
        sp_set = []
        # index_set[0]=agent.index
        # strategy_set[0]=int(agent.action.s)
        # payoff_set[0]=agent.reward

        # for _, j in enumerate(agent.neighbours):
        #     index_set[_+1] = j
        #     strategy_set[_+1] = int(world.agents[j].action.s)
        #     payoff_set[_+1] = world.agents[j].reward

        neighbour_strategy = []
        neighbour_reward = []
        neighbour_index=[]

        for n_i in agent.neighbours:
            neighbour_index.append(world.agents[n_i].index)
            neighbour_strategy.append(world.agents[n_i].action.s)
            neighbour_reward.append([world.agents[n_i].reward])

        arr = np.column_stack((neighbour_strategy)).astype(object)

        arr[:, 0] = arr[:, 0].astype(int)
        obs={
            'n_i':neighbour_index,
            'n_s':neighbour_strategy,
            'n_r':neighbour_reward
        }
        return obs

        # return np.column_stack((index_set,strategy_set,payoff_set))
