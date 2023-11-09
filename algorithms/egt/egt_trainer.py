import numpy as np
import torch
import math


class EGT:
    """
    Trainer class for EGT to update policies
    """

    def __init__(self, args, policy, device=torch.device("cpu"),action_flag:int=0):
        self.device = device
        self.policy = policy
        self.K=args.K
        self.action_flag=action_flag

    def train(self, self_index,buffer,actions):
        """
        Perform a training update using minibatch GD.

        :retun train_info: new strategy for this agent
        """
        train_info = {}
        obs=buffer.obs[-1][-1]
        reward=buffer.rewards[-1][-1][0]

        # log current reward and strategy     
        train_info['reward']=float(reward)
        
        n_index= int(np.random.randint(0, len(obs['n_s'])))
        n_index_global=obs['n_i'][n_index]
        
        # print(reward,obs['n_r'][n_index][0])
        # print(actions[self_index],actions[n_index_global])
        # print('=====')

        # = int(np.random.randint(0, len(obs[1:])))+1
        if np.random.rand() < self.p_i_2_j(reward, obs['n_r'][n_index][0]):
            train_info['new_strategy']=int(obs['n_s'][n_index])
            # print(n_index,n_index_global)
            # print(obs)
            # print('reward:',reward,obs['n_r'][n_index][0])
            # print('strategy:',actions[self_index],actions[n_index_global])
            # input()
            train_info['new_strategy']=actions[n_index_global]
            # actions[self_index]=actions[n_index_global]
        return train_info,actions
    
    def p_i_2_j(self,u_i, u_j):
        '''
        Calculate the probability that i adopts strategy of j
        :param args: the payoff of individual i and j
        :retrun: the enforce probability
        '''
        # avoid too large number
        if (u_i - u_j)/ self.K>50:
            res=0
        elif (u_i - u_j)/ self.K<-50:
            res=1
        else:
            res=1 / (1 +  math.exp((u_i - u_j)/ self.K))
            
        return res
    
    def predict(self):
        pass