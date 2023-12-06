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
        action="store_true",
        default=False,
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
        choices=["EGT", "DQN", "Combine"],
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
    parser.add_argument(
        "--initial_ratio",
        nargs="+",  # "+" allows one or more values to be passed as a list
        type=float,
        default=[0.5, 0.5],
        help="set initial_ratio",
    )
    parser.add_argument(
        "--rewards_pattern",
        type=str,
        choices=["normal", "final"],
        default="Lattice",
        help="Name of the substrate to run",
    )
    parser.add_argument(
        "--K",
        type=float,
        default=0.1,
        help="imitation of strength",
    )
    parser.add_argument(
        "--compare_reward_pattern",
        type=str,
        choices=["all", "neighbour",'none'],
        default="all",
        help="compare reward with average level",
    )
    parser.add_argument(
        "--seperate_interaction_reward",
        action="store_true",
        default=False,
        help="if train interaction action with different reward",
    )
    parser.add_argument(
        "--init_distribution",
        type=str,
        choices=["random", "circle"],
        default="random",
        help="Initial strategy distribution should be random or a circle",
    )

    # parser.add_argument(
    #     "--memory_length",
    #     type=int,
    #     default=5,
    #     help="The maximum action memeory one individual could hold",
    # )
    parser.add_argument(
        "--memory_alpha",
        type=float,
        default=0.5,
        help="free parameter determines how fast the weight factor decays for increaing memeory lenght",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Max length for any episode"
    )
    parser.add_argument(
        "--replay_scheme",
        type=str,
        choices=["prioritized", "uniform"],
        default="prioritized",
        help="The sampling scheme of the replay memory",
    )
    parser.add_argument(
        "--prioritized_replay_alpha",
        type=float,
        default=0.6,
        help="alpha parameter for how much prioritization is used (0 - no prioritization, 1 - full prioritization)",
    )
    parser.add_argument(
        "--prioritized_replay_beta",
        type=float,
        default=0.4,
        help="initial value of beta for prioritized replay buffer",
    )
    parser.add_argument(
        "--use_linear_beta_growth",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--prioritized_replay_eps",
        type=float,
        default=1e-6,
        help="epsilon to add to the TD errors when updating priorities.",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=2000, help="the replay memory buffer size"
    )
    parser.add_argument(
        "--normalize_pattern",
        choices=["none", "all", "sample", "episode"],
        default="none",
        help="how to normalize the training reward",
    )
    parser.add_argument(
        "--strategy_final_exploration",
        type=float,
        default=0.05,
        help="The exploration rate of dilemma action at final",
    )  
    parser.add_argument(
        "--insteraction_final_exploration",
        type=float,
        default=0.1,
        help="The exploration rate of selection action at final",
    )  
    parser.add_argument(
        "--exploration_fraction",
        type=float,
        default=0.1,
        help="The exploration rate of selection action at final",
    )  



    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="initial learning rate (default: 5e-4)"
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
        help="the relateive importance of maximum entropy",
    )

    # neural network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--net_arch",
        nargs="+",  # "+" allows one or more values to be passed as a list
        type=int,
        default=[16, 16],
        help="The number and size of each layer",
    )
    parser.add_argument(
        "--train_pattern",
        choices=["strategy", "together", "seperate"],
        default="strategy",
        help="pass a unit name of frequency type, together make tuple like (5, step) or (2, episode)",
    )
    parser.add_argument(
        "--interact_pattern",
        choices=["together", "seperate"],
        default="together",
        help="get the interaction decision regarging one neighbour and all neighbour once",
    )

    # eval parameters (after training)
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        default=False,
        help="when activated, the model will not pass to traing process",
    )
    parser.add_argument(
        "--eval_dim",
        type=int,
        default=5,
        help="the radius length for instered trained agent",
    )

    # eval parameter (during training)
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    
    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=300,
        help="how many steps of the model to collect transitions for before learning starts",
    )
    parser.add_argument(
        "--train_freq",
        type=int,
        default=20,
        help="Update the model every ``train_freq`` steps.",
    )
    parser.add_argument(
        "--freq_type",
        type=str,
        default="step",
        help="pass a unit name of frequency type, together make tuple like (5, step) or (2, episode)",
    )

    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="episode duration between models saving.",
    )
    parser.add_argument(
        "--save_replay_buffer",
        action="store_true",
        default=False,
        help="by default, do not save replay buffer. If set, save replay buffer.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=2,
        help="maximum number of files, when reached delete the oldest one.",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=False,
        help="by default, do not save replay buffer. If set, save replay buffer.",
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

    # render parameters
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=20,
        help="time duration between contiunous twice video recode.",
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

    # with open(ALL_CONFIGS[parsed_args.scenario_name]) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # for class_name, class_values in config.items():
    #     for k, v in class_values.items():
    #         # if not defined in command-line
    #         if f"--{k}" not in sys.argv[1:]:
    #             setattr(parsed_args, k, v)
    #         else:
    #             print(
    #                 f"CLI argument {k} conflicts with yaml config. "
    #                 f"The latter will be overwritten "
    #                 f"by CLI arguments {k}={getattr(parsed_args, k)}."
    #             )
    print(
        "========  all config   ======== \n {} \n ==============================".format(
            parsed_args
        )
    )
    return parsed_args
