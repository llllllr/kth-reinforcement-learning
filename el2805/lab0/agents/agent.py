import gym
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, env: gym.Env):
        """
        :param env: gym environment
        :type env: gym.Env
        """
        self.env = env

    @abstractmethod
    def compute_action(self, state: np.ndarray):
        """Calculates the best action in a certain state according to the agent's policy.

        :param state: state for which the best action is desired
        :type state: ndarray
        """
        raise NotImplementedError
