import gym
import numpy as np
import itertools as it
from pathlib import Path
from termcolor import colored
from el2805.envs.maze import Maze, Cell
from el2805.envs.grid_world import Move, Position

State = tuple[Position, Position]       # (player position, minotaur position)


class MinotaurMaze(Maze):
    _reward_exit = 1
    _probability_poison_death = 1 / 30

    def __init__(
            self,
            map_filepath: Path,
            horizon: int | None = None,
            discount: float | None = None,
            minotaur_nop: bool = False,
            poison: bool = False
    ):
        super().__init__(map_filepath, horizon, discount)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(self._map.shape),  # player
            gym.spaces.MultiDiscrete(self._map.shape)   # minotaur
        ))
        self.minotaur_nop = minotaur_nop
        self.poison = poison

        if self.finite_horizon():
            # important: since this is an additional objective, the worst-case penalty should be much lower than the
            # exit reward. Otherwise, the player might prioritize minimizing the average time to exit, resulting in a
            # lower probability of exiting alive.
            self._reward_step = - 0.0001
        else:
            # for discounted MDPs, we do not need this reward (see self._reward())
            self._reward_step = None

        # TODO: reduce state space by
        #   - considering the exit-alive configurations as a unique state
        #   - considering the eaten configurations as a unique state
        minotaur_states = [(x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])]
        player_states = [(x, y) for x, y in minotaur_states if self._map[x, y] is not Cell.WALL]
        self._states = list(it.product(player_states, minotaur_states))
        self._state_to_index = {state: s for state, s in zip(self._states, np.arange(len(self._states)))}

    def reward(self, state: State, action: Move, mean: bool = False) -> float:
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
        exited = self._map[player_position] is Cell.EXIT and player_position != minotaur_position
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

        # terminal state (absorbing): nothing happens
        if self._terminal_state(state):
            reward = 0
        # main objective: maximize probability of exiting alive before the time expires
        # <=> maximize reward by collecting the exit reward
        # => positive reward for exiting alive
        elif self._map[player_position_next] is Cell.EXIT and player_position_next != minotaur_position_next:
            reward = self._reward_exit
        # additional objective: don't waste time while you are alive
        # <=> minimize time to exit <=> maximize negative time to exit
        else:
            # for finite-horizon MDPs, we give negative reward (penalty) at each step
            # (e.g., better to arrive at t-1 than t, if both ways have the same success probability)
            if self.finite_horizon():
                reward = self._reward_step
            # for discounted MDPs, we do not need a penalty, as the player will already avoid wasting time due to the
            # discount factor (i.e., exiting in the future is worse than exiting now)
            else:
                reward = 0

        return reward

    def _horizon_reached(self) -> bool:
        # random time horizon geometrically distributed
        if self.poison:
            horizon_reached = self._rng.choice(
                a=[True, False],
                p=[self._probability_poison_death, 1 - self._probability_poison_death]
            )
        else:
            horizon_reached = super()._horizon_reached()
        return horizon_reached

    def _terminal_state(self, state: State) -> bool:
        player_position, minotaur_position = state
        exited = super()._terminal_state(player_position)
        eaten = player_position == minotaur_position
        return exited or eaten

    def _valid_minotaur_moves(self, state: State) -> list[Move]:
        player_position, minotaur_position = state
        valid_moves = []

        if self._map[player_position] is Cell.EXIT or player_position == minotaur_position:
            valid_moves.append(Move.NOP)
        else:
            x_minotaur, y_minotaur = minotaur_position
            if self.minotaur_nop:
                valid_moves.append(Move.NOP)
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
