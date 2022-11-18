import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.envs.rl_problem import RLProblem


class RLAgent(ABC):
    """Interface for RL algorithm."""

    def __init__(self, env: RLProblem, learning_rate: float):
        """
        :param env: RL environment
        :type env: gym.Env
        :param learning_rate: learning rate
        :type learning_rate: float
        """
        self.env = env
        self.learning_rate = learning_rate

        self._rng = None
        self.seed()

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Update the agent's policy according to the provided new observation.

        :param state: current state (s)
        :type state: Any
        :param action: action taken in the current state (a)
        :type action: int
        :param reward: reward received (r(s,a))
        :type reward: float
        :param next_state: next state resulting after taking action a in state s
        :type next_state: Any
        """
        raise NotImplementedError

    @abstractmethod
    def compute_action(self, state: Any, explore: bool = False) -> int:
        """Calculates the best action according to the agent's policy.

        :param state: state for which the action is desired
        :type state: any
        :param explore: whether to allow exploration (training) or not (testing)
        :type explore: bool
        :return: best action according to the agent's policy
        :rtype: int
        """
        raise NotImplementedError

    def seed(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed)
