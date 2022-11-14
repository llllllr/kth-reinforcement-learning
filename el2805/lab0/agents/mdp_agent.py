import numpy as np
from abc import ABC, abstractmethod
from el2805.lab0.envs import MDP


class MDPAgent(ABC):
    def __init__(self, env: MDP):
        """
        :param env: MDP problem
        :type env: MDP
        """
        self.env = env

    @property
    @abstractmethod
    def policy(self) -> np.ndarray:
        """Returns the agent's policy.

        :return: policy
        :rtype: ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> None:
        """Calculates the optimal policy for the MDP."""
        raise NotImplementedError

    @abstractmethod
    def compute_action(self, state: np.ndarray, time_step: int) -> int:
        """Calculates the best action according to the agent's policy.

        :param state: state for which the action is desired
        :type state: ndarray
        :param time_step: time step at which the player is in the specified state
        :type time_step: int
        :return: best action according to the agent's policy
        :rtype: int
        """
        raise NotImplementedError

    def q(self, state: np.ndarray, action: int, v: np.ndarray) -> float:
        """Calculates the Q-function.

        :param state: state
        :type state: ndarray
        :param action: action
        :type action: int
        :param v: value function or, for dynamic programming, u*_{t+1}
        :return: Q(state,action)
        :rtype: float
        """
        # note that we ask for the mean reward instead of a reward sample, so as to support random rewards
        next_states, transition_probabilities = self.env.next_states(state, action)
        s_next = [self.env.state_to_index(next_state) for next_state in next_states]    # indices of next states
        v = v[s_next]   # v(s',a) for all the possible next states
        q = self.env.reward(state, action, mean=True) + self.env.discount * transition_probabilities.dot(v)
        return q
