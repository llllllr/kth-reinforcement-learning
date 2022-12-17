import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.agents.common.agent import Agent
from el2805.environments import TabularMDP


class MDPAgent(Agent, ABC):
    """Interface for an algorithm solving MDPs."""

    def __init__(self, *, environment: TabularMDP, discount: float | None = None):
        """
        :param environment: MDP problem
        :type environment: TabularMDP
        :param discount: discount factor
        :type discount: float, optional
        """
        super().__init__(environment=environment, discount=discount)
        self.env = environment  # to avoid warning for type hints
        self._policy = None

    @abstractmethod
    def solve(self) -> None:
        """Calculates the optimal policy for the MDP."""
        raise NotImplementedError

    @property
    def policy(self) -> np.ndarray:
        """Getter for the agent's policy.

        :return: policy
        :rtype: ndarray
        """
        return self._policy

    @policy.setter
    def policy(self, policy: np.ndarray):
        """Setter for the agent's policy.

        :param policy: policy
        :type policy: ndarray
        """
        self._policy = policy

    def q(self, state: Any, action: int, v: np.ndarray) -> float:
        """Calculates the Q-function.

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :param v: value function or, for dynamic programming, u*_{t+1}
        :return: Q(state,action)
        :rtype: float
        """
        # note that we ask for the mean reward instead of a reward sample to support random rewards
        next_states, transition_probabilities = self.env.next_states(state, action)
        s_next = [self.env.state_index(next_state) for next_state in next_states]    # indices of next states
        v = v[s_next]   # V(s',a) for all the possible next states
        q = self.env.reward(state, action, mean=True) + self.discount * transition_probabilities.dot(v)
        return q
