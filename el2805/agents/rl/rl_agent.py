import gym
import numpy as np
from abc import ABC, abstractmethod
from el2805.agents.agent import Agent


class RLAgent(Agent, ABC):
    """Interface for a RL algorithm."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            learning_rate: float | str,
            seed: int | None = None
    ):
        """Initializes a RLAgent.

        :param environment: RL environment
        :type environment: gym.Env
        :param discount: discount factor of the MDP
        :type discount: float
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment, discount=discount)
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
