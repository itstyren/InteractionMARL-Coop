from typing import Callable
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

def make_env(raw_env):
    def env(args,**kwargs):
        env = raw_env(args,**kwargs)
        return env
    
    return env


def parallel_wrapper_fn(env_fn: Callable) -> Callable:
    def par_fn(args,**kwargs):
        env = env_fn(args,**kwargs)
        env = aec_to_parallel_wrapper(env)
        return env

    return par_fn

def gen_lattice_neighbours(agents_list):
    '''
    Set the neighbor indices for lattice agents
    order is right,bottom,left,top
    '''
    DIM_SIZE = int(len(agents_list) ** 0.5)

    # Define relative positions of neighbors
    neighbor_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    for i in range(DIM_SIZE):
        for j in range(DIM_SIZE):
            agent_index = DIM_SIZE * j + i

            for offset_x, offset_y in neighbor_offsets:
                neighbor_i = (i + offset_x) % DIM_SIZE
                neighbor_j = (j + offset_y) % DIM_SIZE
                neighbor_index = DIM_SIZE * neighbor_j + neighbor_i

                agents_list[agent_index].neighbours.append(neighbor_index)

    return agents_list