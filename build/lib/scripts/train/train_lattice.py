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
from envs.env_wrappers import DummyVecEnv,SubprocVecEnv
from envs.matrix_dilemma._md_utils.utils import (
    make_env,
    gen_lattice_neighbours,
    parallel_wrapper_fn,
)


def make_train_env(all_args,raw_env):
    def get_env_fn(rank):
        def init_env():
           env= raw_env(all_args,max_cycles=all_args.num_env_steps)
           env.seed(all_args.seed + rank * 1000)
           return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn()])
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


    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name)+'_'+str(all_args.env_dim) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type='dilemma='+str(all_args.dilemma_strength),
                         reinit=True)
    else:
        # Generate a run name based on the current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        curr_run = f'run_{current_time}'


        # Create the full path for the new run directory                
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))


    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name)+str(all_args.env_dim) + "@" + str(all_args.user_name))
    
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    if all_args.scenario_name == "lattice_egt":
        from envs.matrix_dilemma import lattice_egt_v0 as LatticeENV
    elif all_args.scenario_name == "lattice_rl":
        from envs.matrix_dilemma import lattice_rl_v0 as LatticeENV

    envs = make_train_env(all_args,LatticeENV.raw_env)

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": (all_args.env_dim) ** 2,
        "device": device,
        "run_dir": run_dir
    }
    
    if all_args.algorithm_name=='EGT':
        from runner.egt.lattice_runner import LatticeRunner as Runner
    elif all_args.algorithm_name=="DQN":
        from runner.rl.shared.lattice_runner import LatticeRunner as Runner

    
    runner =  Runner(config)
    runner.run()

    envs.close()
