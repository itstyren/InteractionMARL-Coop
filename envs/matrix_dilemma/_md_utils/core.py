import numpy as np
from collections import deque


class Action:  # action of the agent
    def __init__(self):
        # strategy action
        self.s = None
        # inteaction action, 1-interaction 0-no interaction
        self.ia = None


class Agent:  # properties of agent entities
    def __init__(self, args):
        # name
        self.name = ""
        self.index = 0
        # Action
        self.action = Action()
        self.reward = 0.0
        # Index of agnet's neighbour
        self.neighbours = []
        # the maximum action store in agent memory
        self.memory_alpha = args.memory_alpha
        self.memory_lenght = args.memory_length
        self.seed = args.seed
        # wether RL or EGT agent
        self.type = "RL"

    def init_memory(
        self, neighbours_act_m, neighbours_intaction_m, intaction_m, self_act_m
    ):
        """
        Initial memory list of all neighbour action with several past actions
        """
        # print(neighbours_act_m, neighbours_intaction_m, intaction_m, self_act_m)

        self.neighbours_act_m = [
            deque(
                neighbours_act_m,
                maxlen=self.memory_lenght,
            )
            for _ in range(len(self.neighbours))
        ]
        # print(self.neighbours_act_m)
        self.neighbours_intaction_m = [
            deque(
                # [
                #     1
                #     for _ in range(self.memory_lenght)
                # ],
                neighbours_intaction_m,
                maxlen=self.memory_lenght,
            )
            for _ in range(len(self.neighbours))
        ]

        self.intaction_m = [
            deque(
                # [
                #     1
                #     for _ in range(self.memory_lenght)
                # ],
                intaction_m,
                maxlen=self.memory_lenght,
            )
            for _ in range(len(self.neighbours))
        ]

        self.self_act_m = deque(self_act_m, maxlen=self.memory_lenght)

        self.past_reward = deque(
            [-99 for _ in range(self.memory_lenght)], maxlen=self.memory_lenght
        )


class World:
    """
    multi-agent world
    """

    def __init__(self, initial_ratio, dilemma_strength):
        """
        list of agents (can change at execution-time!)
        """
        self.agents = []
        self.initial_ratio = initial_ratio
        self.payoff_matrix = np.array(
            [[1, 0], [dilemma_strength, 0]]
        )  # cost-to-benefit ratio
        # self.payoff_matrix=np.array([[1,-1], [1, 0]]) # cost-to-benefit ratio

    def step(self):
        """
        Update state of the world, the interaction structure
        """
        pass
