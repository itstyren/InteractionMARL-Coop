#!/bin/sh
env='Lattice'
scenario='TrainWithEval'
algo='DQN'
exp="E10B32(e1e5)_2R"
env_dim=20
dilemma_strength=1.35
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 1 --num_env_steps 50000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 10 --cuda --n_rollout_threads 4 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 5000 --seed 1 --share_policy false \
      --experiment_name ${exp} --use_render --use_wandb --lr 0.1 --video_interval 10 --use_linear_beta_decay --replay_scheme 'prioritized' --learning_starts 100 \
      --freq_type 'step' --train_freq 8 --prioritized_replay_alpha 0.6 --buffer_size 10000 --memory_alpha 0.6 --save_interval 10  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --train_pattern 'seperate' --save_result  --use_eval --eval_interval 1 \
      --interact_pattern 'together' --seperate_interaction_reward --compare_reward --n_eval_rollout_threads 2 \
# --model_dir '../results/Lattice/Test/DQN/Test/run_2023-10-31_16-43-50/models'
done