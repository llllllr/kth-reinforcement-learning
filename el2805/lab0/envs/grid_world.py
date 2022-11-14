import numpy as np
import gym
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from enum import IntEnum
from el2805.lab0.envs.mdp import MDP


class Move(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    NOP = 4

    def __str__(self):
        if self is Move.UP:
            s = "\u2191"
        elif self is Move.DOWN:
            s = "\u2193"
        elif self is Move.LEFT:
            s = "\u2190"
        elif self is Move.RIGHT:
            s = "\u2192"
        elif self is Move.NOP:
            s = "X"
        else:
            raise ValueError
        return s


class GridWorld(MDP, ABC):
    action_space = gym.spaces.Discrete(len(Move))

    def __init__(self, map_filepath: Path, horizon: Optional[int] = None, discount: Optional[float] = None):
        super().__init__(horizon, discount)
        self._states = None
        self._n_steps = None
        self._player_position = None
        self._player_start = None
        self._map = None
        self._load_map(map_filepath)
        assert isinstance(self._map, np.ndarray)
        self.observation_space = gym.spaces.MultiDiscrete(self._map.shape)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # update state
        previous_state = self._player_position
        new_state = self._next_state(previous_state, action)
        self._player_position = new_state
        self._n_steps += 1

        # calculate reward
        reward = self.reward(previous_state, action)

        # check end of episode
        done = self._episode_end()

        # additional info
        info = {}

        return self._player_position, reward, done, info

    def reset(self) -> np.ndarray:
        self._player_position = self._player_start
        self._n_steps = 0
        return self._player_position

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        map_ = self._map.copy()

        if mode == "human":
            x, y = self._player_position
            map_[x, y] = "P"
        elif mode == "policy":
            for s, action in enumerate(policy):
                state = self.states[s]
                action = Move(action)
                x, y = state
                map_[x, y] = str(action)
        else:
            raise ValueError

        print("=" * 4 * map_.shape[0])
        for i in range(map_.shape[0]):
            for j in range(map_.shape[1]):
                print(map_[i, j], end="\t")
            print()
        print("=" * 4 * map_.shape[0])

    def next_states(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        next_state = self._next_state(state, action)
        return np.asarray([next_state]), np.asarray([1])

    def _next_state(self, state, action):
        action = Move(action)
        if action not in self.valid_actions(state):
            raise ValueError(f"Invalid action {action}")

        x, y = state
        if action == Move.UP:
            x -= 1
        elif action == Move.DOWN:
            x += 1
        elif action == Move.LEFT:
            y -= 1
        elif action == Move.RIGHT:
            y += 1
        elif action == Move.NOP:
            pass
        else:
            raise ValueError(f"Invalid move {action}")
        state = np.asarray((x, y))
        return state

    @property
    def states(self) -> list[np.ndarray]:
        return self._states

    @abstractmethod
    def _episode_end(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _load_map(self, filepath: Path) -> None:
        raise NotImplementedError
