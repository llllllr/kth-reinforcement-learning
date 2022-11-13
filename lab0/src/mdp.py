import gym
import numpy as np
from abc import ABC, abstractmethod


class MDP(gym.Env, ABC):
    """Interface of a Markov Decision Process. Unlike reinforcement learning environments, it exposes the transition
     probabilities and reward function."""

    def __init__(self, horizon: int):
        """
        :param horizon: time horizon
        :type horizon: int
        """
        self.horizon = horizon

    @property
    @abstractmethod
    def valid_states(self) -> list[np.ndarray]:
        """Returns the list of (valid) states in the state space. The invalid states are not included.

        :return: list of valid states
        :rtype: list[ndarray]
        """
        raise NotImplementedError

    @abstractmethod
    def reward(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            mean: bool = False
    ) -> float:
        """Returns reward received by taking a certain action in a certain state.

        :param state: current state
        :type state: ndarray
        :param action: action taken in the current state
        :type action: int
        :param next_state: next state
        :type next_state: ndarray
        :param mean: if True, returns the mean reward instead of sampling a reward (effect only with random rewards)
        :type mean: bool, optional
        :return: reward sample (if mean=False) or mean reward (if mean=True)
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: np.ndarray) -> list[int]:
        """Returns the valid actions in a certain state.

        :param state: state for which to calculate the valid actions
        :type state: ndarray
        :return: valid actions in the specified state
        :rtype: list[int]
        """
        raise NotImplementedError

    @abstractmethod
    def next_states(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the list of next states which can be reached from a certain state and their corresponding transition
        probabilities.

        :param state: current state
        :type state: ndarray
        :param action: action taken in the current state
        :type action: int
        :return: (next states, transition probabilities)
        :rtype: tuple[ndarray, ndarray]
        """
        raise NotImplementedError

    @abstractmethod
    def state_to_index(self, state: np.ndarray) -> int:
        """Returns the index of a certain state in the list of valid states.

        :param state: state for which the index is desired
        :type state: ndarray
        :return: index of the specified state in the list of valid states
        :rtype: int
        """
        raise NotImplementedError
