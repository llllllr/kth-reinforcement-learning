import gym
import numpy as np
import itertools as it
from pathlib import Path
from typing import Optional, Union
from el2805.lab0.envs import Maze
from el2805.lab0.envs.maze import Cell
from el2805.lab0.envs.grid_world import Move, Position

State = tuple[Position, Position]       # (player position, minotaur position)


class MazeMinotaur(Maze):
    def __init__(self, map_filepath: Path, horizon: Optional[int] = None, discount: Optional[float] = None):
        super().__init__(map_filepath, horizon, discount)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(self._map.shape),  # player
            gym.spaces.MultiDiscrete(self._map.shape)   # minotaur
        ))

        minotaur_states = [(x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])]
        player_states = [(x, y) for x, y in minotaur_states if self._map[x, y] is not Cell.WALL]
        self._states = it.product(player_states, minotaur_states)
        self._state_to_index = {    # TODO: does it work with tuple of tuple as a key?
            tuple(state): s for state, s in zip(self._states, np.arange(len(self._states)))
        }

    def reward(
            self,
            state: State,
            action: Move,
            mean: bool = False
    ) -> float:
        player_position, _ = state
        return super().reward(player_position, action)

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        # TODO: color player and minotaur in the rendering
        _, minotaur_position = self._current_state
        x, y = minotaur_position
        tmp = self._map[x, y]
        if mode == "human":
            self._map[x, y] = "M"
        super().render(mode, policy)
        self._map[x, y] = tmp

    def valid_actions(self, state: Union[State, Position]) -> list[Move]:
        if isinstance(state, tuple) and isinstance(state[0], int):
            valid_moves = super().valid_actions(state)
        else:
            print("state", state)
            player_position, minotaur_position = state
            if player_position == minotaur_position:
                valid_moves = [Move.NOP]
            else:
                valid_moves = super().valid_actions(player_position)
        return valid_moves

    def _next_state(self, state: State, action: Move) -> State:
        player_position, minotaur_position = state

        player_action = Move(action)
        if player_action not in self.valid_actions(state):
            raise ValueError(f"Invalid action {action}")
        minotaur_action = self._rng.choice(self._valid_minotaur_moves())

        player_position = super()._next_state(player_position, action)
        minotaur_position = super()._next_state(minotaur_position, minotaur_action)

        state = (player_position, minotaur_position)
        return state

    def _valid_minotaur_moves(self) -> list[Move]:
        player_position, minotaur_position = self._current_state
        valid_moves = []

        if self._done() or player_position == minotaur_position:
            valid_moves.append(Move.NOP)
        else:
            x, y = minotaur_position
            if x - 1 >= 0:
                valid_moves.append(Move.UP)
            if x + 1 < self._map.shape[0]:
                valid_moves.append(Move.DOWN)
            if y - 1 >= 0:
                valid_moves.append(Move.LEFT)
            if y + 1 < self._map.shape[1]:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def _load_map(self, filepath: Path) -> None:
        super()._load_map(filepath)
        minotaur_start = np.asarray(self._map == Cell.GOAL).nonzero()
        minotaur_start = (minotaur_start[0][0], minotaur_start[1][0])
        player_start = self._initial_state
        self._initial_state = (player_start, minotaur_start)
