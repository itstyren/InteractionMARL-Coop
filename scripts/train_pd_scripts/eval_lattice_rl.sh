#!/bin/sh
env='Lattice'
scenario='Eval'
algo='DQN'
exp="E25B32"
env_dim=12
dilemma_strength=1.2
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algorithm name is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
  echo "seed is ${seed}:"
  CUDA_VISIBLE_DEVICES=0 python ../train/train_lattice.py --env_dim ${env_dim} --algorithm_name ${algo} --log_interval 1 --num_env_steps 5000 \
      --env_name ${env} --scenario_name ${scenario} --user_name 'tyren' --episode_length 25 --cuda --n_rollout_threads 4 --use_linear_lr_decay\
      --mini_batch 32 --gradient_steps 1 --dilemma_strength ${dilemma_strength}  --target_update_interval 10000 --seed ${seed} --share_policy false \
      --experiment_name ${exp} --use_render --save_gif --lr 0.1 --video_interval 1 --use_linear_beta_growth --replay_scheme 'prioritized' --learning_starts 50 \
      --freq_type 'step' --train_freq 8 --prioritized_replay_alpha 0.6 --buffer_size 8000 --memory_alpha 0.6 --save_interval 0  \
      --max_files 2 --rewards_pattern 'final' --normalize_pattern 'none' --compare_reward_pattern 'all' --train_pattern 'together' --seperate_interaction_reward \
      --eval_mode --model_dir '../../results/Lattice/TestPC/DQN/E25B16/run_2023-11-18_19-41-27/models' 
done
