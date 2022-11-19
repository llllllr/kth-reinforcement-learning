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
            keys: bool = False
    ):
        self.keys = keys
        super().__init__(map_filepath, horizon, discount)
        self.minotaur_nop = minotaur_nop
        self.poison = poison
        self.minotaur_chase = minotaur_chase

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(self.map.shape),  # player
            gym.spaces.MultiDiscrete(self.map.shape),  # minotaur
            gym.spaces.Discrete(n=len(Progress))        # progress (key not collected, key collected, exited, eaten)
        ))

        # E[T] = 1/(1-lambda) and T has geometric distribution => T~Geo(1-lambda)
        self._probability_poison_death = 1 - self.discount if self.poison else 0
        assert not self.poison or self.discounted()     # poison only for discounted MDPs

        if not self.minotaur_chase:
            self._probability_chase_move = 0

        if self.finite_horizon():
            # important: since this is an additional objective, the worst-case penalty should be much lower than the
            # exit reward. Otherwise, the player might prioritize minimizing the average time to exit, resulting in a
            # lower probability of exiting alive.
            self._reward_step = - 0.0001
        else:
            # for discounted MDPs, we do not need this reward
            # indeed, the exit reward is discounted, so the player will not waste time
            self._reward_step = 0

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
        if self.terminal_state(state):
            reward = 0
        # main objective: maximize probability of exiting alive before the time expires
        # <=> maximize reward by collecting the keys and exit reward
        # => positive reward for collecting keys and exiting alive
        elif next_progress is not Progress.EATEN and \
                progress is Progress.WITHOUT_KEYS and next_progress is Progress.WITH_KEYS:
            reward = self._reward_key
        elif next_progress is not Progress.EATEN and \
                progress is not Progress.EXITED and next_progress is Progress.EXITED:
            reward = self._reward_exit
        # additional objective: don't waste time while you are alive
        # <=> minimize time to exit <=> maximize negative time to exit
        # notice that for discounted MDPs, the step penalty is set to 0 in the constructor
        else:
            reward = self._reward_step

        return reward

    def next_states(self, state: State, action: int) -> tuple[list[State], np.ndarray]:
        if self.terminal_state(state):
            next_states = [state]
            transition_probabilities = np.asarray([1])
        else:
            transition_probabilities = {}

            # example: P('up') =
            # = P('random move', 'up') + P('deterministic move', 'up')
            # = P('random move') * P('up') + P('deterministic move') * P('up')
            # here we add the first part...
            random_moves = self._valid_minotaur_moves(state, chase=False)
            probability_random_move = (1 - self._probability_chase_move) * 1 / len(random_moves)
            for minotaur_move in random_moves:
                next_state = self._next_state(state, action, minotaur_move)
                if next_state not in transition_probabilities:
                    transition_probabilities[next_state] = 0
                transition_probabilities[next_state] += probability_random_move

            if self.minotaur_chase:
                # ...and here the second part
                chase_moves = self._valid_minotaur_moves(state, chase=True)
                probability_chase_move = self._probability_chase_move * 1 / len(chase_moves)
                for minotaur_move in chase_moves:
                    next_state = self._next_state(state, action, minotaur_move)
                    transition_probabilities[next_state] += probability_chase_move

            next_states = list(transition_probabilities.keys())
            transition_probabilities = np.asarray(list(transition_probabilities.values()))

        return next_states, transition_probabilities

    def _next_state(self, state: State, action: int, minotaur_move: Move | None = None) -> State:
        player_position, minotaur_position, progress = state
        action = Move(action)
        if action not in self.valid_actions(state):
            raise ValueError(f"Invalid action {action}")

        if self.terminal_state(state):
            pass    # state stays the same (absorbing state)
        else:
            if minotaur_move is None:
                chase = self.minotaur_chase and random_decide(self._rng, self._probability_chase_move)
                valid_minotaur_moves = self._valid_minotaur_moves(state, chase=chase)
                minotaur_move = self._rng.choice(valid_minotaur_moves)

            next_player_position = super()._next_state(player_position, action)
            next_minotaur_position = super()._next_state(minotaur_position, minotaur_move)

            if next_player_position == next_minotaur_position:
                state = (self._sentinel_position, self._sentinel_position, Progress.EATEN)
            elif progress is Progress.WITH_KEYS and self.map[next_player_position] is MazeCell.EXIT:
                state = (self._sentinel_position, self._sentinel_position, Progress.EXITED)
            elif progress is Progress.WITHOUT_KEYS and self.map[next_player_position] is MinotaurMazeCell.KEY:
                state = (next_player_position, next_minotaur_position, Progress.WITH_KEYS)
            else:
                state = (next_player_position, next_minotaur_position, progress)

        return state

    def valid_actions(self, state: State | Position) -> list[Move]:
        if self.terminal_state(state):
            valid_moves = [Move.NOP]
        else:
            if isinstance(state, tuple) and isinstance(state[0], int):
                player_position = state  # called by parent class with only player position
            else:
                player_position, _, _ = state
            valid_moves = super().valid_actions(player_position)
        return valid_moves

    def _valid_minotaur_moves(self, state: State, chase: bool) -> list[Move]:
        player_position, minotaur_position, _ = state
        valid_moves = []

        if self.terminal_state(state):
            valid_moves.append(Move.NOP)
        else:
            x_minotaur, y_minotaur = minotaur_position
            if self.minotaur_nop:
                valid_moves.append(Move.NOP)
            if x_minotaur - 1 >= 0:
                valid_moves.append(Move.UP)
            if x_minotaur + 1 < self.map.shape[0]:
                valid_moves.append(Move.DOWN)
            if y_minotaur - 1 >= 0:
                valid_moves.append(Move.LEFT)
            if y_minotaur + 1 < self.map.shape[1]:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def _chase_minotaur_moves(self, state: State) -> list[Move]:
        chase_moves = []

        if self.terminal_state(state):
            chase_moves.append(Move.NOP)
        else:
            player_position, minotaur_position, _ = state
            x_player, y_player = player_position
            x_minotaur, y_minotaur = minotaur_position

            delta_x = x_player - x_minotaur
            delta_y = y_player - y_minotaur
            assert abs(delta_x) > 0 or abs(delta_y) > 0  # otherwise it should be eaten (terminal state)

            # move towards the player along the direction with smallest absolute delta
            # if the smallest absolute delta is 0 (aligned along that direction), move along the other direction
            # notice that there can be two moves resulting in the same distance
            if delta_x != 0 and (delta_y == 0 or abs(delta_x) <= abs(delta_y)):
                if delta_x < 0:
                    chase_moves.append(Move.UP)
                else:
                    chase_moves.append(Move.DOWN)
            if delta_y != 0 and (delta_x == 0 or abs(delta_y) <= abs(delta_x)):
                if delta_y < 0:
                    chase_moves.append(Move.LEFT)
                else:
                    chase_moves.append(Move.RIGHT)

        return chase_moves

    def _horizon_reached(self) -> bool:
        # random time horizon geometrically distributed
        if self.poison:
            horizon_reached = random_decide(self._rng, self._probability_poison_death)
        else:
            horizon_reached = super()._horizon_reached()
        return horizon_reached

    def terminal_state(self, state: State | Position) -> bool:
        if isinstance(state, tuple) and isinstance(state[0], int):  # called by parent class
            terminal = False
        else:
            _, _, progress = state
            terminal = progress is Progress.EATEN or progress is Progress.EXITED
        return terminal

    def _generate_state_space(self) -> list[State]:
        # minotaur anywhere
        minotaur_states = [(x, y) for x in range(self.map.shape[0]) for y in range(self.map.shape[1])]

        # player not in walls
        player_states = [(x, y) for x, y in minotaur_states if self.map[x, y] is not MazeCell.WALL]

        # key collected or not
        keys_collected = [Progress.WITHOUT_KEYS, Progress.WITH_KEYS] if self.keys else [Progress.WITH_KEYS]

        # Cartesian product
        states = list(it.product(player_states, minotaur_states, keys_collected))

        # collapse terminal states to just one exit state and one eaten state
        def non_terminal_state(state):
            player_position, minotaur_position, progress = state
            eaten = player_position == minotaur_position
            exited = progress is Progress.WITH_KEYS and self.map[player_position] is MazeCell.EXIT
            return not eaten and not exited
        states = [state for state in states if non_terminal_state(state)]
        states.append((self._sentinel_position, self._sentinel_position, Progress.EATEN))
        states.append((self._sentinel_position, self._sentinel_position, Progress.EXITED))

        return states

    def won(self):
        _, _, progress = self._current_state
        return progress is Progress.EXITED

    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None and policy.shape == self.map.shape)
        map_ = self.map.copy()
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
                    player_color = "yellow"
                else:
                    player_color = None
                map_[player_position] = colored("P", color=player_color)
                map_[minotaur_position] = colored("M", color="magenta")
        elif mode == "policy":
            for i in range(map_.shape[0]):
                for j in range(map_.shape[1]):
                    if self.map[i, j] is not MazeCell.WALL:
                        map_[i, j] = Move(policy[i, j])
        else:
            raise ValueError
        self._render(map_)

    def _load_map(self, filepath: Path) -> None:
        with open(filepath) as f:
            lines = f.readlines()

        # create map
        self.map = np.asarray([[
            (MinotaurMazeCell(symbol) if symbol == MinotaurMazeCell.KEY.value else MazeCell(symbol))
            for symbol in line[:-1].split("\t")
        ] for line in lines])

        # get starting position of player and minotaur
        player_start = np.asarray(self.map == MazeCell.START).nonzero()
        player_start = (int(player_start[0][0]), int(player_start[1][0]))
        minotaur_start = np.asarray(self.map == MazeCell.EXIT).nonzero()
        minotaur_start = (int(minotaur_start[0][0]), int(minotaur_start[1][0]))

        # if there are no keys to collect in the map, start with keys
        keys_present = len(np.asarray(self.map == MinotaurMazeCell.KEY).nonzero()) > 0
        if self.keys:
            assert keys_present
            progress = Progress.WITHOUT_KEYS
        else:
            progress = Progress.WITH_KEYS

        self._initial_state = (player_start, minotaur_start, progress)

    # need to override just to avoid warning of type hints
    @property
    def states(self) -> list[State]:
        return self._states

    def state_index(self, state: State) -> int:
        return self._state_to_index[state]
