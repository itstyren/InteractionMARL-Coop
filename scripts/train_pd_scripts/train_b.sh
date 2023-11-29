#!/bin/sh
env='Lattice'
scenario='lattice_rl'
algo='DQN'
exp="check"
env_dim=3

echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python ../../b.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 50 --num_env_steps 4 \
    --env_name ${env}  --scenario_name ${scenario}  --user_name 'tyren' --episode_length 2 --n_rollout_threads 2 --use_linear_lr_decay
