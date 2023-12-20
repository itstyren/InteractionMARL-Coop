#!/bin/sh
env='Lattice'
scenario='Test_PC_twoMode'
algo='DQN'
exp="(e5e10)"
env_dim=12
dilemma_strength=1.2
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=2 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 5 --num_env_steps 30000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 10 --cuda --n_rollout_threads 8 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 40 --seed 1 --share_policy false \
      --experiment_name ${exp} --use_render --use_wandb --lr 0.1 --video_interval 10 --use_linear_beta_growth --replay_scheme 'prioritized' --learning_starts 200 \
      --freq_type 'step' --train_freq 40 --prioritized_replay_alpha 0.6 --buffer_size 8000 --memory_alpha 0.6 --save_interval 0  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --train_pattern 'seperate'  --use_eval --eval_interval 10\
      --interact_pattern 'together'  --compare_reward_pattern 'all' --n_eval_rollout_threads 2 --seperate_interaction_reward \
      --exploration_fraction 0.05 --strategy_final_exploration 0.05 --tau 0.01 
# --model_dir '../results/Lattice/Test/DQN/Test/run_2023-10-31_16-43-50/models'
done
