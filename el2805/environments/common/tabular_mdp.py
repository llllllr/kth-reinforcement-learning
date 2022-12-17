import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.environments.common.tabular_rl_problem import TabularRLProblem


class TabularMDP(TabularRLProblem, ABC):
    """Interface for a homogeneous Markov Decision Process with discrete state and action spaces."""

    @abstractmethod
    def reward(self, state: Any, action: int, mean: bool = False) -> float:
        """Returns reward received by taking a certain action in a certain state.

        :param state: current state
        :type state: any
        :param action: action taken in the current state
        :type action: int
        :param mean: if True, returns the mean reward instead of sampling a reward (effect only with random rewards)
        :type mean: bool, optional
        :return: reward sample (if mean=False) or mean reward (if mean=True)
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def next_states(self, state: Any, action: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the list of next states which can be reached from a certain state and their corresponding transition
        probabilities.

        :param state: current state
        :type state: any
        :param action: action taken in the current state
        :type action: int
        :return: (next states, transition probabilities)
        :rtype: tuple[ndarray, ndarray]
        """
        raise NotImplementedError
