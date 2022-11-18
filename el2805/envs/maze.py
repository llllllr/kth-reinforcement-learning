import numpy as np
from pathlib import Path
from enum import Enum
from el2805.envs.grid_world import GridWorld, Move, Position


class MazeCell(Enum):
    EMPTY = "0"
    WALL = "#"
    START = "A"
    EXIT = "B"
    DELAY_R1 = "R1"
    DELAY_R2 = "R2"

    @property
    def delay(self) -> int:
        if self is MazeCell.DELAY_R1:
            d = 6
        elif self is MazeCell.DELAY_R2:
            d = 1
        else:
            d = 0
        return d

    def __str__(self):
        return self.value


class Maze(GridWorld):
    _reward_step = -1
    _reward_exit = -_reward_step
    _probability_delay = 0.5

    def __init__(self, map_filepath: Path, horizon: int | None = None, discount: float | None = None):
        super().__init__(map_filepath, horizon, discount)

        self._states = [
            (x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])
            if self._map[x, y] is not MazeCell.WALL
        ]
        self._state_to_index = {state: s for state, s in zip(self._states, np.arange(len(self._states)))}

    def reward(self, state: Position, action: int, mean: bool = False) -> float:
        assert action in self.valid_actions(state)

        # terminal state (absorbing): nothing happens
        if self._terminal_state(state):
            reward = 0
        # main objective: minimize the time to exit <=> maximize the negative time to exit
        # => negative reward (penalty) at each step
        else:
            delay = self._map[state].delay
            reward_no_delay = self._reward_step
            reward_delay = (1 + delay) * self._reward_step

            # exit!
            # Pay attention: the reward when the exit is reached must be greater than the another walk step.
            # Otherwise, with T corresponding to the shortest path length, the agent is not encouraged to exit the maze.
            # Indeed, the total reward of exiting the maze (without staying there for at least 1 timestep) would be
            # equal to the reward of not exiting the maze.
            next_state = self._next_state(state, action)
            if self._terminal_state(next_state):
                reward_no_delay += self._reward_exit
                reward_delay += self._reward_exit
            if mean:
                reward = self._probability_delay * reward_delay + (1 - self._probability_delay) * reward_no_delay
            else:
                reward = self._rng.choice(
                    a=[reward_delay, reward_no_delay],
                    p=[self._probability_delay, 1 - self._probability_delay]
                )

        return reward

    def valid_actions(self, state: Position) -> list[Move]:
        valid_moves = [Move.NOP]

        if not self._terminal_state(state):
            x, y = state

            x_tmp = x - 1
            if x_tmp >= 0 and self._map[x_tmp, y] is not MazeCell.WALL:
                valid_moves.append(Move.UP)

            x_tmp = x + 1
            if x_tmp < self._map.shape[0] and self._map[x_tmp, y] is not MazeCell.WALL:
                valid_moves.append(Move.DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self._map[x, y_tmp] is not MazeCell.WALL:
                valid_moves.append(Move.LEFT)

            y_tmp = y + 1
            if y_tmp < self._map.shape[1] and self._map[x, y_tmp] is not MazeCell.WALL:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def state_index(self, state: Position) -> int:
        return self._state_to_index[state]

    def won(self):
        return self._terminal_state(self._current_state)

    def _terminal_state(self, state: Position) -> bool:
        exited = self._map[state] is MazeCell.EXIT
        return exited

    def _load_map(self, filepath: Path) -> None:
        with open(filepath) as f:
            lines = f.readlines()
        self._map = np.asarray([[MazeCell(symbol) for symbol in line[:-1].split("\t")] for line in lines])
        self._initial_state = np.asarray(self._map == MazeCell.START).nonzero()
        self._initial_state = (int(self._initial_state[0][0]), int(self._initial_state[1][0]))
