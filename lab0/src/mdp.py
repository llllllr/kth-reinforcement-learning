import gym
import numpy as np
from abc import ABC, abstractmethod


class MDP(gym.Env, ABC):
    def __init__(self, horizon):
        self.horizon = horizon

    @property
    def valid_states(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: np.ndarray, action: int, mean: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: np.ndarray) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def next_states(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def state_to_index(self, state: np.ndarray) -> int:
        raise NotImplementedError
