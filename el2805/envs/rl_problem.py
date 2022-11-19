import gym
from abc import ABC, abstractmethod
from typing import Any
from gym.utils.seeding import np_random


class RLProblem(gym.Env, ABC):
    """Interface for a RL problem with discrete state and action spaces."""

    def __init__(self, horizon: int | None = None, discount: float | None = None):
        """
        :param horizon: time horizon, if None then the MDP has infinite horizon
        :type horizon: int, optional
        :param discount: discount factor, must be provided if horizon=None (infinite-horizon MDPs)
        :type discount: float, optional
        """
        self.horizon = horizon
        self.discount = discount if discount is not None else 1
        assert self.finite_horizon() or self.discounted()

        self._rng = None
        self.seed()

    @property
    @abstractmethod
    def states(self) -> list[Any]:
        """Returns the list of (valid) states in the state space. The invalid states are not included.

        :return: list of valid states
        :rtype: list[any]
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: Any, action: int) -> float:
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
    def valid_actions(self, state: Any) -> list[int]:
        """Returns the valid actions in a certain state.

        :param state: state for which to calculate the valid actions
        :type state: any
        :return: valid actions in the specified state
        :rtype: list[int]
        """
        raise NotImplementedError

    @abstractmethod
    def state_index(self, state: Any) -> int:
        """Returns the index of a certain state in the list of valid states.

        :param state: state for which the index is desired
        :type state: ndarray
        :return: index of the specified state in the list of valid states
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_state(self, state: Any) -> bool:
        """Returns whether the state is terminal or not.

        :param state: state to be analyzed
        :type state: Any
        :return: whether the state is terminal or not
        :rtype: bool
        """
        raise NotImplementedError

    def finite_horizon(self) -> bool:
        """Returns whether the MDP is finite horizon or not.

        :return: True if the MDP is finite horizon.
        :rtype: bool
        """
        return self.horizon is not None and self.discount == 1

    def discounted(self) -> bool:
        """Returns whether the MDP is infinite horizon (and discounted) or not.

        :return: True if the MDP is infinite horizon (discounted).
        :rtype: bool
        """
        return self.horizon is None and 0 < self.discount < 1

    def seed(self, seed: int | None = None):
        self._rng, seed = np_random(seed)
        return [seed]
