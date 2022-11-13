import gym
import numpy as np
from abc import ABC, abstractmethod


class MDP(gym.Env, ABC):
    def __init__(self, horizon):
        self.horizon = horizon

    @abstractmethod
    def reward(self, state: np.ndarray, action: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: np.ndarray) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def valid_states(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def next_states(self, state: np.ndarray, action: int) -> tuple[list[np.ndarray], list[float]]:
        raise NotImplementedError

