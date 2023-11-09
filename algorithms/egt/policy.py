import torch
import numpy as np

class EGT_Policy:
    '''
    EGT Policy class. Compute action by Fermi Function.
    '''
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")) -> None:
        self.args=args
        self.device = device
        self.K = args.K

        self.obs_space = obs_space
        self.act_space = act_space

    # def get_actions(self,obs):
    #     '''
    #     Compute actions, return current strtategy
    #     '''
    #     return obs[0][0][1].astype(np.int16)