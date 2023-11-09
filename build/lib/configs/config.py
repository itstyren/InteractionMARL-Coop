import argparse, yaml
from typing import Dict
import sys
from distutils.util import strtobool


def get_config():
    """
    setup some common variables
    """
    parser = argparse.ArgumentParser(description="My Argument Parser")
    # base config
    parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="torch",
        help="The DL framework specifier (tf2 eager is not supported).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--cuda_deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=8,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="[for wandb usage], by default False. will log date to wandb server. or else will use tensorboard to log data.",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # algorithm config
    parser.add_argument(
        "--algorithm_name",
        choices=["EGT", "DQN"],
        default="EGT",
        help="Algorithm to train agents.",
    )

    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        choices=["Lattice"],
        default="Lattice",
        help="Name of the substrate to run",
    )
    parser.add_argument(
        "--env_dim",
        type=int,
        default=10,
        help="the dim size (dim*dim) of the agent network",
    )
    parser.add_argument(
        "--dilemma_strength",
        type=float,
        default=1,
        help="the dilemma strength",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Max length for any episode"
    )

    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # algorithm parameters
    parser.add_argument(
        "--mini_batch",
        type=int,
        default=1,
        help=" Minibatch size for each gradient update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=100,
        help="time step for update the target network every",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1,
        help='the soft update coefficient ("Polyak update", between 0 and 1) default 1',
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="gradient steps to do after each rollout",
    )
    parser.add_argument(
        "--tune_entropy",
        action="store_true",
        default=False,
        help="by default False. will adjust emtropu weight via fixed term. or else will change automaticly.",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.3,
        help='the relateive importance of maximum entropy',
    )    

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice log printing.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser


ALL_CONFIGS = {
    "lattice_egt": "../../configs/lattice_egt.yaml",
    "lattice_rl": "../../configs/lattice_rl.yaml",
}


def update_config(parsed_args) -> Dict:
    """
    update config paraser from yaml file
    """
    import os

    current_directory = os.getcwd()
    print("Current Directory:", current_directory)

    with open(ALL_CONFIGS[parsed_args.scenario_name]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for class_name, class_values in config.items():
        for k, v in class_values.items():
            # if not defined in command-line
            if f"--{k}" not in sys.argv[1:]:
                setattr(parsed_args, k, v)
            else:
                print(
                    f"CLI argument {k} conflicts with yaml config. "
                    f"The latter will be overwritten "
                    f"by CLI arguments {k}={getattr(parsed_args, k)}."
                )
    print(
        "========  all config   ======== \n {} \n ==============================".format(
            parsed_args
        )
    )
    return parsed_args
