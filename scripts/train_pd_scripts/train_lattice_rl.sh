#!/bin/sh
env='Lattice'
scenario='Test_PC'
algo='DQN'
exp="E25B32(e5)"
env_dim=8
dilemma_strength=1.2
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=2 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 1 --num_env_steps 5000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 20 --cuda --n_rollout_threads 5 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 5000 --seed 1 --share_policy false \
      --experiment_name ${exp} --use_render  --use_wandb --lr 0.1 --video_interval 10 --use_linear_beta_decay --replay_scheme 'prioritized' --learning_starts 100 \
      --freq_type 'step' --train_freq 8 --prioritized_replay_alpha 0.6 --buffer_size 10000 --memory_alpha 0 --save_interval 0  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --train_pattern 'seperate'  --compare_reward \
      --interact_pattern 'together'
# --model_dir '../results/Lattice/Test/DQN/Test/run_2023-10-31_16-43-50/models'
done