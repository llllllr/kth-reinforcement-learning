import numpy as np
from pathlib import Path
from enum import Enum
from typing import Optional
from el2805.lab0.envs.grid_world import GridWorld, Move, Position


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
            d = 6
        elif self is Cell.DELAY_R2:
            d = 1
        else:
            d = 0
        return d

    def __str__(self):
        return self.value


class Maze(GridWorld):
    _REWARD_STEP = -1
    _REWARD_GOAL = 1
    _DELAY_PROBABILITY = 0.5

    def __init__(self, map_filepath: Path, horizon: Optional[int] = None, discount: Optional[float] = None):
        super().__init__(map_filepath, horizon, discount)

        self._states = [
            (x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])
            if self._map[x, y] is not Cell.WALL
        ]
        self._state_to_index = {
            tuple(state): s for state, s in zip(self._states, np.arange(len(self._states)))
        }

    def reward(
            self,
            state: Position,
            action: int,
            mean: bool = False
    ) -> float:
        assert action in self.valid_actions(state)
        x, y = state
        x_next, y_next = self._next_state(state, action)

        if self._map[x, y] is Cell.GOAL:
            reward = 0
            # pay attention: the reward when the goal is initially reached must be greater than the other cases
            # Otherwise, with T corresponding to the shortest path, the agent is not encouraged to reach the goal.
            # Indeed, the total reward of reaching the goal (without staying there for at least 1 timestep) would be
            # equal to the reward of not reaching the goal.
            if self._map[x_next, y_next] is Cell.GOAL:
                reward += self._REWARD_GOAL
        else:
            delay = self._map[x, y].delay
            reward_no_delay = self._REWARD_STEP
            reward_delay = (1 + delay) * self._REWARD_STEP

            # pay attention: the reward when the goal is initially reached must be greater than the other cases
            # Otherwise, with T corresponding to the shortest path, the agent is not encouraged to reach the goal.
            # Indeed, the total reward of reaching the goal (without staying there for at least 1 timestep) would be
            # equal to the reward of not reaching the goal.
            if self._map[x_next, y_next] is Cell.GOAL:
                reward_no_delay += self._REWARD_GOAL
                reward_delay += self._REWARD_GOAL

            if mean:
                reward = self._DELAY_PROBABILITY * reward_delay + (1 - self._DELAY_PROBABILITY) * reward_no_delay
            else:
                reward = self._rng.choice(
                    a=[reward_delay, reward_no_delay],
                    p=[self._DELAY_PROBABILITY, 1-self._DELAY_PROBABILITY]
                )

        return reward

    def valid_actions(self, state: Position) -> list[Move]:
        valid_moves = [Move.NOP]

        x, y = state
        if self._map[x, y] is not Cell.GOAL:
            x_tmp = x - 1
            if x_tmp >= 0 and self._map[x_tmp, y] is not Cell.WALL:
                valid_moves.append(Move.UP)

            x_tmp = x + 1
            if x_tmp < self._map.shape[0] and self._map[x_tmp, y] is not Cell.WALL:
                valid_moves.append(Move.DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self._map[x, y_tmp] is not Cell.WALL:
                valid_moves.append(Move.LEFT)

            y_tmp = y + 1
            if y_tmp < self._map.shape[1] and self._map[x, y_tmp] is not Cell.WALL:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def state_to_index(self, state: Position) -> int:
        state = tuple(state)
        return self._state_to_index[state]

    def _done(self) -> bool:
        x, y = self._current_state
        goal_reached = self._map[x, y] is Cell.GOAL
        horizon_reached = self._n_steps >= self.horizon if self.horizon is not None else False
        return goal_reached or horizon_reached

    def _load_map(self, filepath: Path) -> None:
        with open(filepath) as f:
            lines = f.readlines()

        self._map = np.asarray([[Cell(symbol) for symbol in line[:-1].split("\t")] for line in lines])
        self._initial_state = np.asarray(self._map == Cell.START).nonzero()
        self._initial_state = (self._initial_state[0][0], self._initial_state[1][0])   # format as a state
