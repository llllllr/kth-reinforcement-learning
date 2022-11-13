import gym
import numpy as np
from pathlib import Path
from enum import Enum, IntEnum
from mdp import MDP


class Cell(Enum):
    EMPTY = "0"
    WALL = "#"
    START = "A"
    GOAL = "B"
    DELAY_R1 = "R1"
    DELAY_R2 = "R2"

    @property
    def delay(self) -> int:
        if self is Cell.DELAY_R1:
            delay = 6
        elif self is Cell.DELAY_R2:
            delay = 1
        else:
            delay = 0
        return delay

    def __str__(self):
        return self.value


class Move(IntEnum):
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

    def __init__(self, map_filepath: Path, horizon: int):
        super().__init__(horizon)

        self.maze, self.player_start = self._load_maze(map_filepath)
        self.horizon = horizon
        self.observation_space = gym.spaces.MultiDiscrete(self.maze.shape)

        self._player_position = None
        self._n_steps = None

        self._valid_states = [np.asarray((x, y)) for x in range(self.maze.shape[0]) for y in range(self.maze.shape[1])
                              if self.maze[x, y] is not Cell.WALL]
        self._state_to_index = {tuple(state): s
                                for state, s in zip(self._valid_states, np.arange(len(self._valid_states)))}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # update state
        previous_state = self._player_position
        action = Move(action)
        new_state = self._next_state(previous_state, action)
        self._player_position = new_state

        # calculate reward
        reward = self.reward(previous_state, action, new_state)

        # check time horizon
        self._n_steps += 1
        done = self._n_steps >= self.horizon

        # additional info
        info = {}

        return self._player_position, reward, done, info

    def reset(self) -> np.ndarray:
        self._player_position = self.player_start
        self._n_steps = 0
        return self._player_position

    def render(self, mode: str = "human", policy: dict[int, int] = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        maze = self.maze.copy()

        if mode == "human":
            x, y = self._player_position
            maze[x, y] = "P"
        elif mode == "policy":
            for s, action in policy.items():
                state = self.valid_states[s]
                action = Move(action)
                x, y = state
                maze[x, y] = str(action)[0]
        else:
            raise ValueError

        print("=" * 4 * self.maze.shape[0])
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                print(maze[i, j], end="\t")
            print()
        print("=" * 4 * self.maze.shape[0])

    def reward(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            mean: bool = False
    ) -> float:
        assert action in self.valid_actions(state)
        x, y = state
        x_next, y_next = self._next_state(state, action)

        # pay attention: the reward when the next state is the goal must be 0
        # Otherwise, if we give a penalty, then with T=10 the agent is not encouraged to reach the goal.
        # Indeed, the total reward of reaching the goal (without staying there for at least 1 timestep) would be equal
        # to the reward of not reaching the goal.
        if self.maze[x, y] is Cell.GOAL or self.maze[x_next, y_next] is Cell.GOAL:
            reward = 0
        elif self.maze[x, y] is Cell.DELAY_R2 or Cell.DELAY_R1:
            delay = self.maze[x, y].delay
            if mean:
                reward = 0.5 * (1 + delay) * self._REWARD_STEP
            else:
                reward = self._REWARD_STEP if np.random.uniform() > 0.5 else delay * self._REWARD_STEP
        else:
            reward = self._REWARD_STEP
        return reward

    def valid_actions(self, state: np.ndarray) -> list[int]:
        valid_moves = [Move.NOP]

        x, y = state
        if self.maze[x, y] is not Cell.GOAL:
            x_tmp = x - 1
            if x_tmp >= 0 and self.maze[x_tmp, y] is not Cell.WALL:
                valid_moves.append(Move.UP)

            x_tmp = x + 1
            if x_tmp < self.maze.shape[0] and self.maze[x_tmp, y] is not Cell.WALL:
                valid_moves.append(Move.DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self.maze[x, y_tmp] is not Cell.WALL:
                valid_moves.append(Move.LEFT)

            y_tmp = y + 1
            if y_tmp < self.maze.shape[1] and self.maze[x, y_tmp] is not Cell.WALL:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def next_states(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        next_state = self._next_state(state, action)
        return np.asarray([next_state]), np.asarray([1])

    def state_to_index(self, state: np.ndarray) -> int:
        state = tuple(state)
        return self._state_to_index[state]

    @property
    def valid_states(self) -> list[np.ndarray]:
        return self._valid_states

    def _next_state(self, state, action):
        if action not in self.valid_actions(state):
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
        state = np.asarray((x, y))
        return state

    @staticmethod
    def _load_maze(filepath):
        with open(filepath) as f:
            lines = f.readlines()

        maze = np.asarray([[Cell(symbol) for symbol in line[:-1].split("\t")] for line in lines])
        player_start = np.asarray(maze == Cell.START).nonzero()
        player_start = np.asarray((player_start[0][0], player_start[1][0]))     # format as a state

        x, y = player_start
        maze[x, y] = Cell.EMPTY

        return maze, player_start
