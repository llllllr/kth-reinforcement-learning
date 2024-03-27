import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.agents.agent import Agent
from el2805.envs import TabularMDP


class MDPAgent(Agent, ABC):
    """Interface for an algorithm solving MDPs."""

    def __init__(self, environment: TabularMDP, discount: float | None = None):
        """
        :param environment: MDP problem
        :type environment: TabularMDP
        :param discount: discount factor
        :type discount: float, optional
        """
        super().__init__(environment=environment)
        self.environment = environment  # to avoid warning for type hints
        self.discount = discount if discount is not None else 1
        self.policy = None

    @abstractmethod
    def solve(self) -> None:
        """Calculates the optimal policy for the MDP."""
        raise NotImplementedError

    def q(self, state: Any, action: int, v: np.ndarray) -> float:
        """Calculates the Q-function.

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :param v: value function(for all states) or, for dynamic programming, u*_{t+1}
        :return: Q(state,action)
        :rtype: float
        """
        # note that we ask for the mean reward instead of a reward sample to support random rewards
        next_states, transition_probabilities = self.environment.next_states(state, action)
        s_next = [self.environment.state_index(next_state) for next_state in next_states]    # indices of next states
        v = v[s_next]   # V(s',a) for all the possible next states
        # q = reward_for_current_state + 0.9* (transition一一对应乘以v(s') )
        q = self.environment.reward(state, action, mean=True) + self.discount * transition_probabilities.dot(v)
        return q
