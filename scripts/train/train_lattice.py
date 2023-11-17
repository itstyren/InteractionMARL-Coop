from configs.config import get_config, update_config
import sys
import torch
import numpy as np
from pathlib import Path
import os
import setproctitle
import wandb
import socket
from datetime import datetime
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import copy


def make_run_env(all_args, raw_env, env_type=0):
    """
    make train or eval env
    :param evn_type: wether env for training (0) or eval (1). Setting differnt seed
    """

    def get_env_fn(rank):
        def init_env():
            env = raw_env(all_args, max_cycles=all_args.episode_length)
            env.seed(all_args.seed * (1 + 4999 * env_type) + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="lattice_egt",
        help="Which scenario to run on",
    )

    return parser


if __name__ == "__main__":
    parser = get_config()
    parser = parse_args(parser)

    # parse command-line arguments and pre-set argument from config.py
    # The first element of the tuple is an object containing the parsed arguments
    # The second element is a list of any remaining, unrecognized arguments.
    parsed_args = parser.parse_known_args(sys.argv[1:])[0]
    all_args = update_config(parsed_args)

    # cuda
    # print(all_args.cuda,torch.cuda.is_available())
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

        torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path(
            os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[
                0
            ]
            + "/results"
        )
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    _ = copy.copy(all_args.memory_alpha)

    memory_length = 0
    while _ >= 0.01:
        _ = _ * _
        memory_length += 1
    if memory_length == 0:
        memory_length = 1
    all_args.memory_length = memory_length
    print("Agent Memory Length {}".format(all_args.memory_length))

    if all_args.use_wandb:
        run_name = f"{all_args.algorithm_name}_{all_args.experiment_name}_{all_args.env_dim}_{all_args.dilemma_strength}(M{all_args.memory_length})[{all_args.seed}]"
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=run_name,
            group=all_args.scenario_name + "_" + all_args.rewards_pattern,
            dir=str(run_dir),
            job_type="r_"
            + str(all_args.dilemma_strength)
            + "_M_"
            + str(all_args.memory_length)
            + "_"
            + all_args.normalize_pattern,
            reinit=True,
        )
    else:
        # Generate a run name based on the current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        curr_run = f"run_{current_time}"

        # Create the full path for the new run directory
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # set the process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + str(all_args.env_dim)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    if all_args.algorithm_name == "EGT":
        from envs.matrix_dilemma import lattice_egt_v0 as LatticeENV
    else:
        from envs.matrix_dilemma import lattice_rl_v0 as LatticeENV

    envs = make_run_env(all_args, LatticeENV.raw_env, env_type=0)
    eval_envs = make_run_env(all_args, LatticeENV.raw_env, env_type=1) if all_args.model_dir is not None else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": (all_args.env_dim) ** 2,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.algorithm_name == "EGT":
        from runner.egt.lattice_runner import LatticeRunner as Runner
    else:
        # run experiments
        if all_args.share_policy:
            from runner.rl.shared.lattice_runner import LatticeRunner as Runner
        else:
            from runner.rl.separated.lattice_runner import LatticeRunner as Runner

    runner = Runner(config)

    if all_args.model_dir is None:
        runner.run()
    else:
        runner.eval_run()
    envs.close()

    # post process
    envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.logger.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.logger.close()
