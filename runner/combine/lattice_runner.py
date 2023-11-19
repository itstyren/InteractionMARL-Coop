from .base_runner import Runner
import numpy as np
import torch
import time
from utils.util import gini, consecutive_counts, convert_array_to_two_arrays, save_array
import pdb
import sys

def _t2n(x):
    """
    Convert a tensor into a NumPy array.
    """
    return x.detach().cpu().numpy()


class LatticeRunner(Runner):
    def __init__(self, config):
        super(LatticeRunner, self).__init__(config)

    def eval_run(self, eval_time=1):
        """
        eval the trained model
        :param eval_time: The totoal eval trials
        """
        eval_envs = self.eval_envs
        trials = int(eval_time / self.n_rollout_threads)
        eval_scores = []
        for trial in range(trials):
            print("trail is {}".format(trial))
            self.num_timesteps = 0
            self.episodes = (
                int(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
            self.start_time = time.time_ns()
            self._num_timesteps_at_start = 0
            eval_obs, coop_level = eval_envs.reset()
            eval_obs=eval_obs[0]

            for agent_id in range(self.num_agents):
                if not np.isin(agent_id, self.rl_agent_indices):
                    self.buffer[agent_id].obs[0] = np.array(eval_obs[agent_id]).copy()



            print(
                "====== Initial Cooperative Level {:.2f} ======".format(
                    np.mean(coop_level)
                )
            )
            step = 0
            episode = 1
            all_frames = []
            while self.num_timesteps < self.num_env_steps:
                actions = []
                interactions = []
                # get action for egt, rl action predfict from obs
                all_action = np.array(eval_envs.get_actions())
                for agent_id in range(self.num_agents):
                    if np.isin(agent_id, self.rl_agent_indices):
                        agent_action = self.trainer[agent_id].predict(
                            [eval_obs[agent_id]])
                        
                        agent_action = _t2n(agent_action)
                        agent_action, agent_interaction = convert_array_to_two_arrays(
                            agent_action)

                    else:
                        agent_action = all_action[agent_id]
                        agent_interaction = [
                            15 for i in range(self.n_rollout_threads)]

                    actions.append(agent_action)
                    interactions.append(agent_interaction)

                actions=np.concatenate([item if isinstance(item, list) else [item] for item in actions])
                interactions=np.concatenate(interactions)

                combine_action = np.dstack((actions, interactions))
                (
                    eval_obs,
                    eval_rewards,
                    terminations,
                    truncations,
                    eval_infos,
                ) = eval_envs.step(combine_action[0])

                self.insert(eval_obs,eval_rewards)
                self.train_egt(actions)



                self.num_timesteps += self.n_rollout_threads
                step += 1

                if step >= self.episode_length:
                    # record every step for current episode
                    if (
                        self.all_args.use_render
                        and (
                            episode % self.video_interval == 0
                            or episode == self.episodes - 1
                            or episode==1
                        )
                    ):
                        image,i_n = self.render(self.num_timesteps)
                        all_frames.append(image[0])            
                        self.write_to_video(all_frames, episode) 
                    if episode % self.log_interval == 0:
                        self._dump_logs(episode)


                    step = 0
                    episode += 1
                    all_frames = []
                    

    @torch.no_grad()
    def render(self, num_timesteps):
        """
        Visualize the env at current state
        """
        envs = self.eval_envs

        image, intraction_array = envs.render("rgb_array", num_timesteps)
        # print
        return image, intraction_array

    def train_egt(self,actions):
        '''
        train new strategy
        '''
        # training info for every agent
        train_infos = []
        arr_order = np.arange(self.num_agents)
        for agent_id in arr_order:
            if  not np.isin(agent_id, self.rl_agent_indices):
                # print('old_s',self.envs.env.world.agents[agent_id].action.s)
                train_info,actions = self.trainer[agent_id].train(agent_id,self.buffer[agent_id],actions)
                if 'new_strategy' in train_info:
                    # print('change from',self.eval_envs.env.world.agents[agent_id].action.s,'to',train_info['new_strategy'])
                    self.eval_envs.env.world.agents[agent_id].action.s=train_info['new_strategy']
                train_infos.append(train_info) 

        for agent_id in range(self.num_agents):
            obs=self.eval_envs.env.observe(self.eval_envs.env.world.agents[agent_id].name)
            if  not np.isin(agent_id, self.rl_agent_indices):
                self.buffer[agent_id].after_update(obs) 
  
    def insert(self, obs,rewards):
        '''
        insert 
        '''
        for agent_id in range(self.num_agents):
            if not np.isin(agent_id, self.rl_agent_indices):
                self.buffer[agent_id].insert(np.array(obs[agent_id]),rewards[agent_id].reshape(-1, 1))