#!/bin/sh
env='Lattice'
scenario='T8_40_Two_f04'
algo='DQN'
exp="(e5e10)Link"
env_dim=30
dilemma_strength=1.22
seed_max=4


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=2 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 5 --num_env_steps 50000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 10 --cuda --n_rollout_threads 8 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 40 --seed ${seed} --share_policy false \
      --experiment_name ${exp} --use_render --use_wandb --lr 0.1 --video_interval 10 --use_linear_beta_growth --replay_scheme 'prioritized' --learning_starts 100 \
      --freq_type 'step' --train_freq 40 --prioritized_replay_alpha 0.6 --buffer_size 10000 --memory_alpha 0.6 --save_interval 10  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --train_pattern 'seperate' --save_result  --use_eval --eval_interval 10 \
      --interact_pattern 'together' --compare_reward_pattern 'all' --n_eval_rollout_threads 2 --seperate_interaction_reward \
      --exploration_fraction 0.04 --strategy_final_exploration 0.05 --tau 0.01
# --model_dir '../results/Lattice/Test/DQN/Test/run_2023-10-31_16-43-50/models' --seperate_interaction_reward
done