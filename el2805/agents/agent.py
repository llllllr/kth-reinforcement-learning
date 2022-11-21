from abc import ABC, abstractmethod
from el2805.envs import RLProblem


class Agent(ABC):
    """Interface for an agent controlling a stochastic Markovian dynamical system."""

    def __init__(self, env: RLProblem, discount: float | None = None):
        """
        :param env: MDP problem
        :type env: MDP
        :param discount: discount factor
        :type discount: float, optional
        """
        self.env = env
        self.discount = discount if discount is not None else 1

    @abstractmethod
    def compute_action(self, **kwargs) -> int:
        """Calculates the best action according to the agent's policy. The parameters depend on the algorithm."""
        raise NotImplementedError
