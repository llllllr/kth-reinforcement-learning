import gym
import numpy as np
import itertools as it
from pathlib import Path
from enum import Enum, IntEnum
from termcolor import colored
from el2805.envs.maze import Maze, MazeCell
from el2805.envs.grid_world import Move, Position
from el2805.utils import random_decide


class Progress(IntEnum):
    EATEN = 0           # eaten
    WITHOUT_KEYS = 1    # keys not collected yet
    WITH_KEYS = 2       # keys collected (or not present in the maze)
    EXITED = 3          # exited alive


class MinotaurMazeCell(Enum):
    KEY = "C"

    def __str__(self):
        return self.value


# (player position, minotaur position, progress)
State = tuple[Position, Position, Progress]


class MinotaurMaze(Maze):
    _reward_key = 1
    _reward_exit = 1
    _probability_chase_move = 0.35
    _sentinel_position = (-1, -1)

    def __init__(
            self,
            map_filepath: Path,
            horizon: int | None = None,
            discount: float | None = None,
            minotaur_nop: bool = False,
            poison: bool = False,
            minotaur_chase: bool = False,
            keys_collected: bool = False
    ):
        super().__init__(map_filepath, horizon, discount)
        self.minotaur_nop = minotaur_nop
        self.poison = poison
        self.minotaur_chase = minotaur_chase
        self.keys = keys_collected

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(self._map.shape),  # player
            gym.spaces.MultiDiscrete(self._map.shape),  # minotaur
            gym.spaces.Discrete(n=len(Progress))        # progress (key not collected, key collected, exited, eaten)
        ))

        # E[T] = 1/(1-lambda) and T has geometric distribution => T~Geo(1-lambda)
        self._probability_poison_death = 1 - self.discount if self.poison else None
        assert not self.poison or self.discounted()     # poison only for discounted MDPs

        if self.finite_horizon():
            # important: since this is an additional objective, the worst-case penalty should be much lower than the
            # exit reward. Otherwise, the player might prioritize minimizing the average time to exit, resulting in a
            # lower probability of exiting alive.
            self._reward_step = - 0.0001
        else:
            # for discounted MDPs, we do not need this reward (see self._reward())
            self._reward_step = None

        self._states = self._generate_state_space()
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

    def _reward(self, state: State, next_state: State) -> float:
        _, _, progress = state
        _, _, next_progress = next_state

        # terminal state (absorbing): nothing happens
        if self._terminal_state(state):
            reward = 0
        # main objective: maximize probability of exiting alive before the time expires
        # <=> maximize reward by collecting the keys and exit reward
        # => positive reward for collecting keys and exiting alive
        elif next_progress is not Progress.EATEN and \
                progress is not Progress.WITH_KEYS and next_progress is Progress.WITH_KEYS:
            reward = self._reward_key
        elif next_progress is not Progress.EATEN and \
                progress is not Progress.EXITED and next_progress is Progress.EXITED:
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

    def next_states(self, state: State, action: int) -> tuple[list[State], np.ndarray]:
        transition_probabilities = {}
        valid_minotaur_moves = self._valid_minotaur_moves(state)

        if self._terminal_state(state):
            next_states = [state]
            transition_probabilities = np.asarray([1])
        else:
            # example: P('up') =
            # = P('random move', 'up') + P('deterministic move', 'up')
            # = P('random move') * P('up') + P('deterministic move') * P('up')
            # here we add the first part...
            probability_minotaur_move = 1 / len(valid_minotaur_moves)
            if self.minotaur_chase:
                probability_minotaur_move *= 1 - self._probability_chase_move
            for minotaur_move in valid_minotaur_moves:
                next_state = self._next_state(state, action, minotaur_move)
                if next_state in transition_probabilities:
                    transition_probabilities[next_state] += probability_minotaur_move
                else:
                    transition_probabilities[next_state] = probability_minotaur_move

            # ...and here the second part
            if self.minotaur_chase:
                minotaur_move = self._chase_minotaur_move(state)
                next_state = self._next_state(state, action, minotaur_move)
                transition_probabilities[next_state] += self._probability_chase_move

            next_states = list(transition_probabilities.keys())
            transition_probabilities = np.asarray(list(transition_probabilities.values()))

        return next_states, transition_probabilities

    def _next_state(self, state: State, action: int, minotaur_move: Move | None = None) -> State:
        player_position, minotaur_position, progress = state
        action = Move(action)
        if action not in self.valid_actions(state):
            raise ValueError(f"Invalid action {action}")

        if self._terminal_state(state):
            pass
        else:
            if minotaur_move is None:
                if self.minotaur_chase and random_decide(self._rng, self._probability_chase_move):
                    minotaur_move = self._chase_minotaur_move(state)
                else:
                    minotaur_move = self._rng.choice(self._valid_minotaur_moves(state))

            next_player_position = super()._next_state(player_position, action)
            next_minotaur_position = super()._next_state(minotaur_position, minotaur_move)

            if next_player_position == next_minotaur_position:
                progress = Progress.EATEN
                state = (self._sentinel_position, self._sentinel_position, progress)
            elif progress is Progress.WITH_KEYS and self._map[next_player_position] is MazeCell.EXIT:
                progress = Progress.EXITED
                state = (self._sentinel_position, self._sentinel_position, progress)
            elif self._map[next_player_position] is MinotaurMazeCell.KEY:
                state = (next_player_position, next_minotaur_position, Progress.WITH_KEYS)
            else:
                state = (next_player_position, next_minotaur_position, progress)

        return state

    def valid_actions(self, state: State | Position) -> list[Move]:
        if self._terminal_state(state):
            valid_moves = [Move.NOP]
        else:
            if isinstance(state, tuple) and isinstance(state[0], int):
                player_position = state  # called by parent class with only player position
            else:
                player_position, _, _ = state
            valid_moves = super().valid_actions(player_position)
        return valid_moves

    def _valid_minotaur_moves(self, state: State) -> list[Move]:
        player_position, minotaur_position, _ = state
        valid_moves = []

        if self._terminal_state(state):
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

    def _chase_minotaur_move(self, state: State) -> Move:
        if self._terminal_state(state):
            move = Move.NOP
        else:
            player_position, minotaur_position, _ = state
            x_player, y_player = player_position
            x_minotaur, y_minotaur = minotaur_position

            delta_x = x_player - x_minotaur
            delta_y = y_player - y_minotaur
            assert abs(delta_x) > 0 or abs(delta_y) > 0  # otherwise it should be eaten (terminal state)

            # strategy:
            # - move towards the player along the direction with smallest absolute delta (so as to block the path)
            # - if the smallest absolute delta is 0 (aligned along that direction), move along the other direction
            if delta_x != 0 and (delta_y == 0 or abs(delta_x) <= abs(delta_y)):
                if delta_x < 0:
                    move = Move.UP
                else:
                    move = Move.DOWN
            else:
                assert delta_y != 0 and (delta_x == 0 or abs(delta_x) > abs(delta_y))
                if delta_y < 0:
                    move = Move.LEFT
                else:
                    move = Move.RIGHT
        return move

    def _horizon_reached(self) -> bool:
        # random time horizon geometrically distributed
        if self.poison:
            horizon_reached = random_decide(self._rng, self._probability_poison_death)
        else:
            horizon_reached = super()._horizon_reached()
        return horizon_reached

    def _terminal_state(self, state: State | Position) -> bool:
        if isinstance(state, tuple) and isinstance(state[0], int):  # called by parent class
            terminal = False
        else:
            _, _, progress = state
            terminal = progress is Progress.EATEN or progress is Progress.EXITED
        return terminal

    def _generate_state_space(self) -> list[State]:
        # minotaur anywhere
        minotaur_states = [(x, y) for x in range(self._map.shape[0]) for y in range(self._map.shape[1])]

        # player not in walls
        player_states = [(x, y) for x, y in minotaur_states if self._map[x, y] is not MazeCell.WALL]

        # key collected or not
        keys_collected = [Progress.WITHOUT_KEYS, Progress.WITH_KEYS] if self.keys else [Progress.WITH_KEYS]

        # Cartesian product
        states = list(it.product(player_states, minotaur_states, keys_collected))

        # collapse terminal states to just one exit state and one eaten state
        def non_terminal_state(state):
            player_position, minotaur_position, _ = state
            eaten = player_position == minotaur_position
            exited = self._map[player_position] is MazeCell.EXIT  # even if eaten at the exit, we don't care here
            return not eaten and not exited
        states = [state for state in states if non_terminal_state(state)]
        states.append((self._sentinel_position, self._sentinel_position, Progress.EATEN))
        states.append((self._sentinel_position, self._sentinel_position, Progress.EXITED))

        return states

    def won(self):
        _, _, progress = self._current_state
        return progress.EXITED

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        map_ = self._map.copy()
        if mode == "human":
            player_position, minotaur_position, progress = self._current_state

            if progress is Progress.EATEN:
                print("LOSER...")
            elif progress is Progress.EXITED:
                print("WINNER!")
            else:
                if progress is Progress.WITHOUT_KEYS:
                    player_color = "red"
                elif progress is Progress.WITH_KEYS:
                    player_color = "green"
                else:
                    player_color = None
                map_[player_position] = colored("P", color=player_color)
                map_[minotaur_position] = colored("M", color="magenta")
                self._render(map_)
        elif mode == "policy":
            for s, action in enumerate(policy):
                player_position, _, _ = self.states[s]
                action = Move(action)
                map_[player_position] = str(action)
                self._render(map_)
        else:
            raise ValueError

    def _load_map(self, filepath: Path) -> None:
        with open(filepath) as f:
            lines = f.readlines()

        # create map
        self._map = np.asarray([[
            (MinotaurMazeCell(symbol) if symbol == MinotaurMazeCell.KEY.value else MazeCell(symbol))
            for symbol in line[:-1].split("\t")
        ] for line in lines])

        # get starting position of player and minotaur
        player_start = np.asarray(self._map == MazeCell.START).nonzero()
        player_start = (int(player_start[0][0]), int(player_start[1][0]))
        minotaur_start = np.asarray(self._map == MazeCell.EXIT).nonzero()
        minotaur_start = (int(minotaur_start[0][0]), int(minotaur_start[1][0]))

        # if there are no keys to collect in the map, start with keys
        keys_present = len(np.asarray(self._map == MinotaurMazeCell.KEY).nonzero()) == 0
        progress = Progress.WITHOUT_KEYS if keys_present else Progress.WITH_KEYS

        self._initial_state = (player_start, minotaur_start, progress)

    @property
    def states(self) -> list[State]:    # need to override only for avoiding warnings of type hints
        return self._states
