import numpy as np
import gym
from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
from termcolor import colored
from el2805.envs.tabular_mdp import TabularMDP


class Move(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    NOP = 4

    def __str__(self):
        if self is Move.UP:
            s = "\u2191"
        elif self is Move.DOWN:
            s = "\u2193"
        elif self is Move.LEFT:
            s = "\u2190"
        elif self is Move.RIGHT:
            s = "\u2192"
        elif self is Move.NOP:
            s = "X"
        else:
            raise ValueError
        return s


Position = tuple[int, int]  # 用于指定一个名为 Position 的新类型。在这个声明中，Position 是一个元组类型，其中包含两个整数元素


class GridWorld(TabularMDP, ABC): # 继承自表格MDP
    action_space = gym.spaces.Discrete(len(Move))

    def __init__(self, map_filepath: Path, horizon: int | None = None):
        super().__init__(horizon)
        self._states = None
        self._n_steps = None
        self._current_state = None
        self._initial_state = None
        self.map = None
        self._load_map(map_filepath)
        assert isinstance(self.map, np.ndarray)
        self.observation_space = gym.spaces.MultiDiscrete(self.map.shape)

    @property
    def states(self) -> list[Position]:
        return self._states

    @abstractmethod
    def _load_map(self, filepath: Path) -> None:
        raise NotImplementedError

    # 把每个某个特定action apply之后, 
    def step(self, action: int) -> tuple[Position, float, bool, dict]:
        # update state
        previous_state = self._current_state
        new_state = self._next_state(previous_state, action)
        self._current_state = new_state

        # calculate reward , 对上一个state来计算reward
        reward = self.reward(previous_state, action)

        # check end of episode, 
        self._n_steps += 1
        done = self._horizon_reached() or self.terminal_state(self._current_state)

        # additional info, dict
        info = {}

        return self._current_state, reward, done, info

    def reset(self) -> Position:
        self._current_state = self._initial_state
        self._n_steps = 0
        return self._current_state

    # 这个方法用于渲染当前环境状态，以便用户可视化观察。
    def render(self, mode: str = "human", policy: np.ndarray = None) -> None:
        assert mode == "human" or (mode == "policy" and policy is not None)
        # map_是保存每个state对应的最优action对应的string的 array
        map_ = self.map.copy()
        if mode == "human":
            map_[self._current_state] = colored("P", color="blue")
        elif mode == "policy":
            for s, action in enumerate(policy):
                state = self.states[s]
                action = Move(action)
                map_[state] = str(action)
        else:
            raise ValueError
        self._render(map_)

    # 返回如果p是probabilistic transition, 那么某个state某个action之后的下一个 可能的states,以及对应的可能性
    # 返回一个元组, 第一个元素是下一个可能的states们, 第二个元素是 对应的probability.
    def next_states(self, state: Position, action: int) -> tuple[list[Position], np.ndarray]:
        next_state = self._next_state(state, action)
        return ([next_state]), np.asarray([1])  # deterministic

    def _next_state(self, state: Position, action: int) -> Position: # 对一个单个的state, take一个单个的action, 得到新state
        x, y = state
        if action == Move.UP:
            x -= 1
        elif action == Move.DOWN:
            x += 1
        elif action == Move.LEFT:
            y -= 1
        elif action == Move.RIGHT:
            y += 1
        elif action == Move.NOP:
            pass
        else:
            raise ValueError(f"Invalid move {action}")
        state = (x, y)
        return state

    def _horizon_reached(self):
        horizon_reached = self._n_steps >= self.horizon if self.finite_horizon() else False
        return horizon_reached

    @staticmethod
    def _render(map_):
        print("=" * 8 * map_.shape[0])
        for i in range(map_.shape[0]):
            for j in range(map_.shape[1]):
                print(map_[i, j], end="\t")
            print()
        print("=" * 8 * map_.shape[0])
