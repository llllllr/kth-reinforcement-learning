import gym
import numpy as np
from el2805.agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, environment: gym.Env):
        super(RandomAgent, self).__init__(environment=environment)
        assert isinstance(environment.action_space, gym.spaces.Discrete)
        self._n_actions = environment.action_space.n

    def compute_action(self, *, state: np.ndarray, **kwargs) -> int:
        _ = kwargs
        return np.random.randint(0, self._n_actions)
