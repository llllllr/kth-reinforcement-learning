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
        self.policy = None

    @abstractmethod
    def solve(self) -> None:
        """Calculates the optimal policy for the MDP."""
        raise NotImplementedError

    @abstractmethod
    def compute_action(self, state, time_step):
        raise NotImplementedError

    def q(self, state, action, v):
        # note that we ask for the mean reward instead of a reward sample, so as to support random rewards
        next_states, transition_probabilities = self.env.next_states(state, action)
        s_next = [self.env.state_to_index(next_state) for next_state in next_states]    # indices of next states
        v = v[s_next]   # v(s',a) for all the possible next states
        q = self.env.reward(state, action, mean=True) + self.env.discount * transition_probabilities.dot(v)
        return q
