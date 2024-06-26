#!/bin/sh
env='Lattice'
scenario='EGT'
algo='EGT'
exp="check"
env_dim=30

echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}"

# Loop to vary dilemma_strength from 1 to 3, incrementing by 0.1 each time
for dilemma_strength in $(seq 1.2 0.1 1.2)
do
    echo "Running simulation with dilemma_strength: ${dilemma_strength}"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 50 --num_env_steps 60000 --seed 1 \
        --env_name ${env}  --scenario_name ${scenario}  --user_name 'Anonymous' --episode_length 1 --dilemma_strength ${dilemma_strength} --n_rollout_threads 1 \
        --rewards_pattern 'normal' --K 0.1  --memory_alpha 0 --use_render  --video_interval 50 --init_distribution 'random' --compare_reward_pattern 'none'
done