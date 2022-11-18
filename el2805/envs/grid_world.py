import numpy as np
import gym
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
from termcolor import colored
from el2805.envs.mdp import MDP


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


Position = tuple[int, int]


class GridWorld(MDP, ABC):
    action_space = gym.spaces.Discrete(len(Move))

    def __init__(self, map_filepath: Path, horizon: int | None = None, discount: float | None = None):
        super().__init__(horizon, discount)
        self._states = None
        self._n_steps = None
        self._current_state = None
        self._initial_state = None
        self._map = None
        self._load_map(map_filepath)
        assert isinstance(self._map, np.ndarray)
        self.observation_space = gym.spaces.MultiDiscrete(self._map.shape)

    @property
    def states(self) -> list[Position]:
        return self._states

    @abstractmethod
    def _terminal_state(self, state: Position) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _load_map(self, filepath: Path) -> None:
        raise NotImplementedError

    def step(self, action: int) -> tuple[Position, float, bool, dict]:
        # update state
        previous_state = self._current_state
        new_state = self._next_state(previous_state, action)
        self._current_state = new_state

        # calculate reward
        reward = self.discount**self._n_steps * self.reward(previous_state, action)

        # check end of episode
        self._n_steps += 1
        done = self._horizon_reached() or self._terminal_state(self._current_state)

        # additional info
        info = {}

        return self._current_state, reward, done, info

    def reset(self) -> Position:
        self._current_state = self._initial_state
        self._n_steps = 0
        return self._current_state

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        map_ = self._map.copy()
        if mode == "human":
            map_[self._current_state] = colored("P", color="blue")
        elif mode == "policy":
            for s, action in enumerate(policy):
                state = self.states[s]
                action = Move(action)
                map_[state] = str(action)
        else:
            raise ValueError
        self._render(map_)

    def next_states(self, state: Position, action: int) -> tuple[list[Position], np.ndarray]:
        next_state = self._next_state(state, action)
        return ([next_state]), np.asarray([1])  # deterministic

    def _next_state(self, state: Position, action: int) -> Position:
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
        state = (x, y)
        return state

    def _horizon_reached(self):
        horizon_reached = self._n_steps >= self.horizon if self.finite_horizon() else False
        return horizon_reached

    @staticmethod
    def _render(map_):
        print("=" * 8 * map_.shape[0])
        for i in range(map_.shape[0]):
            for j in range(map_.shape[1]):
                print(map_[i, j], end="\t")
            print()
        print("=" * 8 * map_.shape[0])
