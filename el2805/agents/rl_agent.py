import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.envs.rl_problem import RLProblem


class RLAgent(ABC):
    """Interface for a RL algorithm."""

    def __init__(self, env: RLProblem, learning_rate: float | str, seed: int | None = None):
        """
        :param env: RL environment
        :type env: RLProblem
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param seed: seed
        :type seed: int, optional
        """
        self.env = env
        self.learning_rate = learning_rate

        self._rng = None
        self.seed(seed)

    @abstractmethod
    def compute_action(self, **kwargs) -> int:
        """Calculates the best action according to the agent's policy. The parameters depend on the algorithm."""
        raise NotImplementedError

    def seed(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed)
