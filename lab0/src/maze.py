import gym
import numpy as np
from enum import Enum


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


class Maze(gym.Env):
    action_space = gym.spaces.Discrete(len(Move))
    _REWARD_STEP = -1
    _REWARD_GOAL = 0

    def __init__(self, map_filepath, horizon):
        self.maze, self.player_start = self._load_maze(map_filepath)
        self.horizon = horizon
        self.observation_space = gym.spaces.MultiDiscrete(self.maze.shape)

        self._player_position = None
        self._n_steps = None

    def step(self, action):
        # check action
        action = Move(action)
        print(f"action: {action}")
        if not self._valid_action(action):
            raise ValueError(f"Invalid action {action}")

        # update state
        x, y = self._player_position
        if action is Move.UP:
            x -= 1
        elif action is Move.DOWN:
            x += 1
        elif action is Move.RIGHT:
            y += 1
        elif action is Move.LEFT:
            y -= 1
        elif action is Move.NOP:
            pass
        else:
            raise ValueError(f"Invalid move {action}")
        self._player_position = (x, y)

        # check new state
        if not self._valid_state():
            raise ValueError(f"Invalid action {action}")

        # calculate reward
        reward = self._REWARD_GOAL if self._goal_achieved() else self._REWARD_STEP

        # check time horizon
        self._n_steps += 1
        done = self._n_steps >= self.horizon

        # additional info
        info = {}

        return self._player_position, reward, done, info

    def reset(self):
        self._player_position = self.player_start
        self._n_steps = 0

    def render(self, mode="human"):
        print("=" * 2 * self.maze.shape[0])
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                output = "P" if self._player_position == (i, j) else self.maze[i, j]
                print(output, end=" ")
            print()
        print("=" * 2 * self.maze.shape[0])

    def _valid_action(self, action):
        x, y = self._player_position
        no_nop_in_goal = self.maze[x, y] is Cell.GOAL and action is not Move.NOP
        return not no_nop_in_goal

    def _valid_state(self):
        x, y = self._player_position
        out_of_map = not (0 <= x < self.maze.shape[0]) or not (0 <= y < self.maze.shape[1])
        on_wall = self.maze[x, y] is Cell.WALL
        return not out_of_map and not on_wall

    def _goal_achieved(self):
        x, y = self._player_position
        is_goal = self.maze[x, y] is Cell.GOAL
        return is_goal

    @staticmethod
    def _load_maze(filepath):
        with open(filepath) as f:
            lines = f.readlines()

        symbol_to_cell = {e.value: e for e in Cell}
        maze = np.asarray([[symbol_to_cell[symbol] for symbol in line[:-1]] for line in lines])

        player_start = np.asarray(maze == Cell.START).nonzero()
        maze[player_start] = Cell.EMPTY

        return maze, player_start
