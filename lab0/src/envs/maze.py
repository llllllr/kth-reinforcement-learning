import gym
import numpy as np
from enum import Enum
from typing import Union
from pathlib import Path
from mdp import MDP


class Cell(Enum):
    EMPTY = "o"
    WALL = "#"
    START = "A"
    GOAL = "B"

    def __str__(self):
        return self.value


class Move(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    NOP = 4

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return False

    def __str__(self):
        return self.name.lower()


class Maze(MDP):
    action_space = gym.spaces.Discrete(len(Move))
    _REWARD_STEP = -1

    def __init__(self, map_filepath: Union[str, Path], horizon: int):
        super().__init__(horizon)

        self.maze, self.player_start = self._load_maze(map_filepath)
        self.horizon = horizon
        self.observation_space = gym.spaces.MultiDiscrete(self.maze.shape)

        self._player_position = None
        self._n_steps = None

    def step(self, action: Union[Move, int]) -> tuple[tuple[int, int], float, bool, dict]:
        # calculate reward
        reward = self.reward(self._player_position, action)

        # update state
        action = Move(action)
        self._player_position = self._next_state(self._player_position, action)

        # check time horizon
        self._n_steps += 1
        done = self._n_steps >= self.horizon

        # additional info
        info = {}

        return self._player_position, reward, done, info

    def reset(self) -> tuple[int, int]:
        self._player_position = self.player_start
        self._n_steps = 0
        return self._player_position

    def render(self, mode: str = "human") -> None:
        print("=" * 2 * self.maze.shape[0])
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                output = "P" if self._player_position == (i, j) else self.maze[i, j]
                print(output, end=" ")
            print()
        print("=" * 2 * self.maze.shape[0])

    def reward(self, state: tuple[int, int], action: Union[Move, int]) -> float:
        x, y = state
        reward = 0 if (self.maze[x, y] is Cell.GOAL and action == Move.NOP) else self._REWARD_STEP
        return reward

    def valid_actions(self, state):
        valid_moves = {Move.NOP}

        x, y = state
        if self.maze[x, y] is not Cell.GOAL:
            x_tmp = x - 1
            if x_tmp >= 0 and self.maze[x_tmp, y] is not Cell.WALL:
                valid_moves.add(Move.UP)

            x_tmp = x + 1
            if x_tmp < self.maze.shape[0] and self.maze[x_tmp, y] is not Cell.WALL:
                valid_moves.add(Move.DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self.maze[x, y_tmp] is not Cell.WALL:
                valid_moves.add(Move.LEFT)

            y_tmp = y + 1
            if y_tmp < self.maze.shape[1] and self.maze[x, y_tmp] is not Cell.WALL:
                valid_moves.add(Move.RIGHT)

        return valid_moves

    def valid_states(self) -> list[tuple[int, int]]:
        valid_states = [(x, y) for x in range(self.maze.shape[0]) for y in range(self.maze.shape[1])
                        if self.maze[x, y] is not Cell.WALL]
        return valid_states

    def next_states(self, state: np.ndarray, action: int) -> tuple[list[tuple[int, int]], list[float]]:
        next_state = self._next_state(state, action)
        return [next_state], [1]

    def _next_state(self, state, action):
        if action not in self.valid_actions(self._player_position):
            raise ValueError(f"Invalid action {action}")

        x, y = state
        if action is Move.UP:
            x -= 1
        elif action is Move.DOWN:
            x += 1
        elif action is Move.LEFT:
            y -= 1
        elif action is Move.RIGHT:
            y += 1
        elif action is Move.NOP:
            pass
        else:
            raise ValueError(f"Invalid move {action}")
        state = (x, y)
        return state

    @staticmethod
    def _load_maze(filepath):
        with open(filepath) as f:
            lines = f.readlines()

        symbol_to_cell = {e.value: e for e in Cell}
        maze = np.asarray([[symbol_to_cell[symbol] for symbol in line[:-1]] for line in lines])

        player_start = np.asarray(maze == Cell.START).nonzero()
        maze[player_start] = Cell.EMPTY

        return maze, player_start
