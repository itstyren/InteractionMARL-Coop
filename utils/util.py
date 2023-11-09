from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Tuple, Union
from torch.nn import functional as F
from typing import Callable
import math
import pickle
import glob
import os


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(observation_space, spaces.Box):
        return obs.float()
    if isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs[_].long(), num_classes=dim).float()
                for _, dim in enumerate(observation_space.nvec)
            ],
            dim=-1,
        ).view(obs.shape[0], observation_space.nvec[0])
    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key])
        return preprocessed_obs


def round_up(number: float, decimals: int = 2) -> float:
    """
    Round a number up to a specified number of decimal places.

    :param number: The number to be rounded.
    :param decimals: The number of decimal places to round to (default is 2).
    :return: The rounded number.
    """
    if not isinstance(decimals, int):
        raise TypeError("Decimal places must be an integer")
    if decimals < 0:
        raise ValueError("Decimal places must be 0 or more")

    if decimals == 0:
        return math.ceil(number)
    else:
        factor = 10**decimals
        return math.ceil(number * factor) / factor


def ensure_numpy_array(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = ensure_numpy_array(array).flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


class FileManager:
    def __init__(self, base_dir, max_files=5, suffix="pkl"):
        self.base_dir = base_dir
        self.max_files = max_files
        self.suffix = suffix

    def create_file(self, obj, path):
        existing_files = self.get_existing_files()

        # If the maximum number of files is reached, delete the oldest one
        if len(existing_files) >= self.max_files:
            oldest_file = min(existing_files, key=os.path.getctime)
            os.remove(oldest_file)

        # Create a new file
        new_file = path
        with open(new_file, "wb") as file_handler:
            # Use protocol>=4 to support saving large objects
            pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    def get_existing_files(self):
        return glob.glob(os.path.join(self.base_dir, f"*.{self.suffix}"))
    

def find_latest_file(directory, extension):
    file_list = glob.glob(os.path.join(directory, f"*.{extension}"))
    if not file_list:
        return None
    return max(file_list, key=os.path.getmtime)


def consecutive_counts(row):
    """
    Calculate the consecutive counts, average counts, and longest consecutive counts for a given row.

    :param row: The row for which consecutive counts should be calculated.
    :return: A tuple of consecutive counts, average counts, and longest consecutive counts.
    """
    consecutive_counts = []
    current_count = 0
    current_target = None
    total_counts = {}
    longest_consecutive = {}

    for num in row:
        if num == current_target:
            current_count += 1
        else:
            if current_target is not None:
                consecutive_counts.append(current_count)
                if current_target not in total_counts:
                    total_counts[current_target] = [current_count, 1]
                else:
                    total_counts[current_target][0] += current_count
                    total_counts[current_target][1] += 1

                if current_count > longest_consecutive.get(current_target, 0):
                    longest_consecutive[current_target] = current_count

            current_target = num
            current_count = 1

    consecutive_counts.append(current_count)
    if current_target not in total_counts:
        total_counts[current_target] = [current_count, 1]
    else:
        total_counts[current_target][0] += current_count
        total_counts[current_target][1] += 1
        
    if current_count > longest_consecutive.get(current_target, 0):
        longest_consecutive[current_target] = current_count


    return consecutive_counts, total_counts,longest_consecutive