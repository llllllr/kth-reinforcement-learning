import gym
import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class Agent(ABC):
    """Interface for an agent controlling a stochastic Markovian dynamical system."""

    def __init__(self, *, environment: gym.Env, discount: float | None = None):
        """
        :param environment: RL environment
        :type environment: gym.Env
        :param discount: discount factor
        :type discount: float, optional
        """
        self.environment = environment
        self.discount = discount if discount is not None else 1

    @abstractmethod
    def compute_action(self, **kwargs) -> int:
        """Calculates the best action according to the agent's policy. The parameters depend on the algorithm."""
        raise NotImplementedError

    def save(self, filepath: str | Path):
        """Saves the whole agent in a pickle file.

        :param filepath: path where to save the agent
        :type filepath: str or Path
        """
        with open(filepath, mode="wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str | Path):
        """Loads the whole agent from a pickle file.

        :param filepath: path where the agent is saved
        :type filepath: str or Path
        """
        with open(filepath, mode="rb") as file:
            agent = pickle.load(file)
        assert isinstance(agent, Agent)
        return agent
