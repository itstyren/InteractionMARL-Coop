#!/bin/sh
env='Lattice'
scenario='Test_PC_twoMode'
algo='DQN'
exp="(e5e10)"
env_dim=10
dilemma_strength=1.20
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=2 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 2 --num_env_steps 10000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 10 --cuda --n_rollout_threads 8 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 2500 --seed 1 --share_policy false \
      --experiment_name ${exp} --use_render --use_wandb --lr 0.1 --video_interval 5 --use_linear_beta_growth --replay_scheme 'prioritized' --learning_starts 100 \
      --freq_type 'step' --train_freq 24 --prioritized_replay_alpha 0.6 --buffer_size 8000 --memory_alpha 0 --save_interval 0  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --train_pattern 'seperate'  --use_eval --eval_interval 5\
      --interact_pattern 'together'  --compare_reward_pattern 'all' --n_eval_rollout_threads 2 \
      --exploration_fraction 0.2
# --model_dir '../results/Lattice/Test/DQN/Test/run_2023-10-31_16-43-50/models'
done
