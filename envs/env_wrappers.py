import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)




class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_spaces, action_spaces):
        self.num_envs = num_envs
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()



# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), self.env.observation_spaces, self.env.action_spaces)
        self.actions = None


    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        termination=False
        truncation=False
        for a in self.actions:
            result=self.env.step(a)
            if result is None:
                pass
            else:
                 obs_n, reward_n,termination,truncation,infos  = result
                
            # if  'bool' in truncation.__class__.__name__  or 'bool' in termination.__class__.__name__:
            #     if termination or truncation:
            #         obs_n,cl = self.env.reset()

        self.actions = None
        return obs_n, reward_n,termination,truncation,infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs,coop_l=zip(*results)
        return np.stack(obs),np.stack(coop_l)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self,  mode="rgb_array",step=0):
        if mode == "rgb_array":
            frame,intraction_array=self.env.render(mode,step)
            return np.array([frame]),np.array([intraction_array])
            # return np.array([env.render(mode,step) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def get_actions(self):
        actions=[]
        for agent in self.env.world.agents:
            actions.append(agent.action.s)
        return actions     



class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_spaces,interact_observation_spaces, action_spaces = self.remotes[0].recv()
        self.interact_observation_spaces=interact_observation_spaces
        ShareVecEnv.__init__(self, len(env_fns), observation_spaces, action_spaces)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs,rews,termination,truncation,infos = zip(*results)
        # print(rews)
        # print(np.stack(rews))
        return np.stack(obs),np.stack(rews),np.stack(termination),np.stack(truncation),np.stack(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs,coop_l=zip(*results)
        return np.stack(obs),np.stack(coop_l)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array",step=0):
        for remote in self.remotes:
            remote.send(('render', (mode,step)))
        if mode == "rgb_array":   
            results = [remote.recv() for remote in self.remotes]
            frame,intraction_array=zip(*results)
            return np.stack(frame),np.stack(intraction_array)
         
    def get_actions(self):
        for remote in self.remotes:
            remote.send(('get_actions', None))
        results = [remote.recv() for remote in self.remotes]
        return results        

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        
        if cmd == 'step':
            termination=False
            for _ in range(len(data)):
                result=env.step(data[_])
                if result is None:
                    pass
                else:
                   obs_n, reward_n,termination,truncation,infos = result 
                
            # checking whether a variable named done is of type bool or a NumPy array 
            if  'bool' in truncation.__class__.__name__  or 'bool' in termination.__class__.__name__:
                if termination:
                    obs_n,cl = env.reset(options='termination')
                elif truncation:
                    obs_n,cl = env.reset()
            remote.send((obs_n,reward_n,termination,truncation,infos))
        elif cmd == 'reset':
            ob,cl = env.reset()
            remote.send((ob,cl))
        elif cmd == 'render':
            if data[0] == "rgb_array":
                fr,interact_arr = env.render(mode=data[0],step=data[1])
                remote.send((fr,interact_arr))
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_spaces, env.interact_observation_spaces,env.action_spaces))
        elif cmd == 'get_actions':
            actions=[]
            for agent in env.world.agents:
                actions.append(agent.action.s)

            remote.send(actions)
        else:
            raise NotImplementedError
