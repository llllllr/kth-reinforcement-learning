import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class MDP(gym.Env, ABC):
    """Interface for a homogeneous Markov Decision Process."""

    def __init__(self, horizon: Optional[int] = None, discount: Optional[float] = None):
        """
        :param horizon: time horizon, if None then the MDP has infinite horizon
        :type horizon: int, optional
        :param discount: discount factor, must be provided if horizon=None (infinite-horizon MDPs)
        :type discount: float, optional
        """
        self.horizon = horizon
        self.discount = discount if discount is not None else 1
        assert self.finite_horizon() or self.infinite_horizon()

        self._rng = None
        self.seed()

    @property
    @abstractmethod
    def states(self) -> list[np.ndarray]:
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
            mean: bool = False
    ) -> float:
        """Returns reward received by taking a certain action in a certain state.

        :param state: current state
        :type state: ndarray
        :param action: action taken in the current state
        :type action: int
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

    def finite_horizon(self) -> bool:
        """Returns whether the MDP is finite horizon or not.

        :return: True if the MDP is finite horizon.
        :rtype: bool
        """
        return self.horizon is not None and self.discount == 1

    def infinite_horizon(self) -> bool:
        """Returns whether the MDP is infinite horizon (and discounted) or not.

        :return: True if the MDP is infinite horizon (discounted).
        :rtype: bool
        """
        return self.horizon is None and 0 < self.discount < 1

