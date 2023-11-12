from .base_runner import Runner
import numpy as np
import torch
import time


class LatticeRunner(Runner):
    def __init__(self, config):
        super(LatticeRunner, self).__init__(config)

    def run(self):
        self.warmup() 

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps=0
        for episode in range(episodes):
            all_frames = []
            for step in range(self.episode_length):
                # record every step for current episode
                if (
                    self.all_args.use_render
                    and (
                        episode % self.video_interval == 0
                        or episode == self.episodes - 1
                    )
                ):
                    image = self.render(total_num_steps)
                    all_frames.append(image[0])            
                    self.write_to_video(all_frames, episode) 

                obs, actions,rewards=self.collect(step)
                # print(obs[1])
                data=obs,actions,rewards

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            train_infos = self.train(actions)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            self.num_timesteps += self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} -- Algo {} -- Exp {} -- updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                if self.env_name == "Lattice":
                    strategy_set=[]
                    for agent_id in range(self.num_agents):
                        strategy_set.append(self.envs.env.world.agents[agent_id].action.s)
                    #     group_reward.append(train_infos[agent_id].get('reward'))
                    train_infos['results/episode_cooperation_level']=1-np.mean(strategy_set)
                    # group_info['average_episode_rewards']=np.mean(group_reward)
                    # train_infos.append(group_info)
                    
                self.print_train(train_infos)
                self.log_train(train_infos)

    def warmup(self):
        '''
        Initial runner
        '''
        # reset env
        obs, coop_level = self.envs.reset()
        print(
            "====== Initial Cooperative Level {:.2f} ======".format(np.mean(coop_level))
        )
        # all rollout threads have same initial obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

        # total episode num
        self.episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        # current timesteps for single thread
        self.num_timesteps = 0



    @torch.no_grad()
    def collect(self, step):
        '''
        collect agent action and reward
        '''
        # get strategy of all current agent
        actions = []
        for agent_id in range(self.num_agents):
            # if self.env._index_map[agent]==10:
            #     print('==============collect')
            #     print(self.buffer[self.env._index_map[agent]].obs)
            # this is where you would insert your policy
            # action=self.trainer[self.env._index_map[agent]].policy.get_actions(self.buffer[self.env._index_map[agent]].obs[step])
            # action=self.envs.world.agents[agent_id].action.s
            action=self.envs.env.world.agents[agent_id].action.s
            actions.append(action)
        obs, rews, termination, truncation,infos=self.envs.step(actions)
        # rewards = [value for value in self.env.rewards.values()]
        return obs,actions,rews

    def insert(self, data):
        obs,actions,rewards = data
        # print(obs)
        # for a,r in zip(actions,rewards):
        #     print(a,r)
        for agent_id in range(self.num_agents):
            # print(obs,rewards)
            self.buffer[agent_id].insert(np.array(obs[agent_id]),rewards[agent_id])
            # if agent_id==10:
            #     print('==============insert payoff obs')
            #     print(self.buffer[agent_id].obs) 

    def compute(self):
        pass

    def train(self,actions):
        '''
        train new strategy
        '''
        # training info for every agent
        train_infos = []

        arr_order = np.arange(self.num_agents)
        for agent_id in arr_order:
            # print('old_s',self.envs.env.world.agents[agent_id].action.s)
            train_info,actions = self.trainer[agent_id].train(agent_id,self.buffer[agent_id],actions)
            if 'new_strategy' in train_info:
                # print('agent id',agent_id)
                # print('new_strategy',train_info['new_strategy'])
                self.envs.env.world.agents[agent_id].action.s=train_info['new_strategy']
                # print('new_s',self.envs.env.world.agents[agent_id].action.s)
                # print('=====')
                # input()
            train_infos.append(train_info)    

        for agent_id in range(self.num_agents):
            obs=self.envs.env.observe(self.envs.env.world.agents[agent_id].name)
            self.buffer[agent_id].after_update(obs) 

        avarage_reward=np.array(
            [entry['reward'] for entry in train_infos],dtype=float
            )
        ti={
           'payoff/average_episode_rewards':np.mean(avarage_reward)
        }

        return ti
    
    def render(self, num_timesteps):
        """
        Visualize the env at current state
        """
        envs = self.envs
        image = envs.render("rgb_array", num_timesteps)
        return image
