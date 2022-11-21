import numpy as np
from abc import ABC, abstractmethod
from el2805.agents.agent import Agent
from el2805.envs import RLProblem


class RLAgent(Agent, ABC):
    """Interface for a RL algorithm."""

    def __init__(
            self, env: RLProblem,
            discount: float,
            learning_rate: float | str,
            seed: int | None = None
    ):
        """
        :param env: RL environment
        :type env: RLProblem
        :param discount: discount factor of the MDP
        :type discount: float
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(env=env, discount=discount)
        self.learning_rate = learning_rate
        self._rng = None
        self.seed(seed)

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Updates the policy (or value function, or Q-function) from new observation(s).

        The parameters depend on the algorithm. For example:
        - Q-learning requires tuples (state, action, reward, next_state).
        - SARSA requires tuples (state, action, reward, next_state, next_action).
        """
        raise NotImplementedError

    def seed(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed)
