import gym
import numpy as np
import itertools as it
from pathlib import Path
from termcolor import colored
from el2805.lab0.envs import Maze
from el2805.lab0.envs.maze import Cell
from el2805.lab0.envs.grid_world import Move, Position

State = tuple[Position, Position]       # (player position, minotaur position)


class MazeMinotaur(Maze):
    _REWARD_EATEN = -1

    def __init__(self, map_filepath: Path, horizon: int | None = None, discount: float | None = None):
        super().__init__(map_filepath, horizon, discount)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(self._map.shape),  # player
            gym.spaces.MultiDiscrete(self._map.shape)   # minotaur
        ))

        minotaur_states = [(x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])]
        player_states = [(x, y) for x, y in minotaur_states if self._map[x, y] is not Cell.WALL]
        self._states = list(it.product(player_states, minotaur_states))
        self._state_to_index = {state: s for state, s in zip(self._states, np.arange(len(self._states)))}

    def reward(
            self,
            state: State,
            action: Move,
            mean: bool = False
    ) -> float:
        assert action in self.valid_actions(state)

        if mean:
            next_states, transition_probabilities = self.next_states(state, action)
            rewards = []
            for next_state in next_states:
                reward = self._reward(state, next_state)
                rewards.append(reward)
            reward = transition_probabilities.dot(rewards)  # mean reward
        else:
            next_state = self._next_state(state, action)
            reward = self._reward(state, next_state)

        return reward

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        map_ = self._map.copy()
        if mode == "human":
            player_position, minotaur_position = self._current_state
            map_[player_position] = colored("P", color="blue")
            map_[minotaur_position] = colored("M", color="red")
        elif mode == "policy":
            for s, action in enumerate(policy):
                player_position, _ = self.states[s]
                action = Move(action)
                map_[player_position] = str(action)
        else:
            raise ValueError
        self._render(map_)

    def valid_actions(self, state: State | Position) -> list[Move]:
        if isinstance(state, tuple) and isinstance(state[0], int):
            valid_moves = super().valid_actions(state)
        else:
            player_position, minotaur_position = state
            if player_position == minotaur_position:
                valid_moves = [Move.NOP]
            else:
                valid_moves = super().valid_actions(player_position)
        return valid_moves

    def next_states(self, state: State, action: int) -> tuple[list[State], np.ndarray]:
        # the transition probability is non-deterministic, as the minotaur moves randomly
        # so, there is one next state for each possible minotaur move
        player_position, minotaur_position = state
        player_position_next = super()._next_state(player_position, action)
        valid_minotaur_moves = self._valid_minotaur_moves(state)
        next_states = []
        for minotaur_action in valid_minotaur_moves:
            minotaur_position_next = super()._next_state(minotaur_position, minotaur_action)
            next_states.append((player_position_next, minotaur_position_next))

        # same probability for all minotaur moves
        transition_probabilities = 1 / len(valid_minotaur_moves) * np.ones(len(valid_minotaur_moves))

        return next_states, transition_probabilities

    def won(self):
        player_position, minotaur_position = self._current_state
        exited = self._map[player_position] is Cell.EXIT and player_position is not minotaur_position
        return exited

    def _next_state(self, state: State, action: Move) -> State:
        player_position, minotaur_position = state

        player_action = Move(action)
        if player_action not in self.valid_actions(state):
            raise ValueError(f"Invalid action {action}")
        minotaur_move = self._rng.choice(self._valid_minotaur_moves(state))

        player_position = super()._next_state(player_position, action)
        minotaur_position = super()._next_state(minotaur_position, minotaur_move)

        state = (player_position, minotaur_position)
        return state

    def _reward(self, state: State, next_state: State) -> float:
        player_position, minotaur_position = state
        player_position_next, minotaur_position_next = next_state

        if self._map[player_position] is Cell.EXIT:
            reward = 0
            if player_position != minotaur_position:
                reward += self._REWARD_EATEN
        else:
            reward = self._REWARD_STEP

            # Pay attention: the reward when the exit is initially reached must be greater than the other
            # cases. Otherwise, with T corresponding to the shortest path, the agent is not encouraged to
            # exit the maze. Indeed, the total reward of exiting the maze (without staying there for at least
            # 1 timestep) would be equal to the reward of not exiting the maze.
            if self._map[player_position_next] is Cell.EXIT and player_position_next != minotaur_position_next:
                reward += self._REWARD_EXIT
        return reward

    def _terminal_state(self, state: State = None) -> bool:
        player_position, minotaur_position = state
        exited = super()._terminal_state(player_position)
        eaten = player_position == minotaur_position
        return exited or eaten

    def _valid_minotaur_moves(self, state: State) -> list[Move]:
        player_position, minotaur_position = state
        valid_moves = []

        x_player, y_player = player_position
        if self._map[x_player, y_player] is Cell.EXIT or player_position == minotaur_position:
            valid_moves.append(Move.NOP)
        else:
            x_minotaur, y_minotaur = minotaur_position
            if x_minotaur - 1 >= 0:
                valid_moves.append(Move.UP)
            if x_minotaur + 1 < self._map.shape[0]:
                valid_moves.append(Move.DOWN)
            if y_minotaur - 1 >= 0:
                valid_moves.append(Move.LEFT)
            if y_minotaur + 1 < self._map.shape[1]:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def _load_map(self, filepath: Path) -> None:
        super()._load_map(filepath)
        minotaur_start = np.asarray(self._map == Cell.EXIT).nonzero()
        minotaur_start = (minotaur_start[0][0], minotaur_start[1][0])
        player_start = self._initial_state
        self._initial_state = (player_start, minotaur_start)
