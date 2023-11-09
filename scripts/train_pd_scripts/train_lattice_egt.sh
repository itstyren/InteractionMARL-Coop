#!/bin/sh
env='Lattice'
scenario='lattice_egt'
algo='EGT'
exp="check"
env_dim=60
dilemma_strength=1.3

echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 50 --num_env_steps 5000 \
    --env_name ${env}  --scenario_name ${scenario}  --user_name 'tyren'  --episode_length 1 --dilemma_strength ${dilemma_strength} --n_rollout_threads 1 \
    --rewards_pattern 'final' --K 0.1  --memory_alpha 0.9 --use_render --use_wandb --video_interval 20 --train_interaction 
