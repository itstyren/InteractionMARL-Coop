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


def get_central_and_nearby_indices(dim, radius_length):
    assert radius_length*2<dim,f"Expected radius_length< {int(dim/2)}, but got {radius_length}"

    center_idx = dim // 2 * dim + dim // 2

    # Convert the 1D index to 2D coordinates (assuming a square lattice)
    center_x = center_idx % dim
    center_y = center_idx // dim

    # Initialize a list to store nearby indices within the circular region
    nearby_indices = []

    for x in range(center_x - radius_length, center_x + radius_length + 1):
        for y in range(center_y - radius_length, center_y + radius_length + 1):
            # Check if the coordinates are within the circular region
            if (x - center_x)**2 + (y - center_y)**2 <= radius_length**2:
                # Convert 2D coordinates back to 1D index
                nearby_indices.append(y * dim + x)

    return center_idx, nearby_indices

    # nearby_indices = []

    # for i in range(-radius_length, radius_length + 1):
    #     for j in range(-radius_length, radius_length + 1):
    #         # Calculate the flattened index
    #         index = middle_idx + i * dim + j

    #         # Check if the index is within the bounds of the flattened list
    #         if 0 <= index < dim * dim:
    #             nearby_indices.append(index)

    # print('\n')
    # print(f'nearby_indices is {round(len(nearby_indices) / (dim**2) * 100, 2)}% large \n')
    # print(nearby_indices)
    # return middle_idx, nearby_indices
