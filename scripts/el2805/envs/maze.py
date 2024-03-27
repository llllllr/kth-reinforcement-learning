import numpy as np
from pathlib import Path
from enum import Enum
from el2805.envs.grid_world import GridWorld, Move, Position


class MazeCell(Enum):  # 继承自Enumerate类, MazeCell.EMPTY 是枚举的members, name是empty, 0是value
    EMPTY = "0"
    WALL = "#"
    START = "A"
    EXIT = "B"
    DELAY_R1 = "R1"
    DELAY_R2 = "R2"

    @property    # 在类方法的前边加上@property装饰器, 代表是一个<只读/read-only>方法, 
    def delay(self) -> int:  # 定义一个函数/方法, -> marks the return function annotation indicating the function returns an int
        if self is MazeCell.DELAY_R1:
            d = 6
        elif self is MazeCell.DELAY_R2:
            d = 1
        else:
            d = 0
        return d

    def __str__(self):  # 对实例对象使用str(MazeCell)或者print(MazeCell)时使用, 结果是对应的枚举值
        return self.value


class Maze(GridWorld):
    _reward_step = -1
    _reward_exit = -_reward_step # 1
    _probability_delay = 0.5
    # 类构造器 接收参数名 map_filepath : 类型名, 参数名 horizon: 类型名(int|None) = 默认值None
    def __init__(self, map_filepath: Path, horizon: int | None = None):
        super().__init__(map_filepath, horizon)

        # range(self.map.shape[0]) 生成一个[0,1,2,...,行数-1]的可迭代对象
        # 这是一个嵌套的循环，用于遍历地图的所有可能的行列组合。结果是包含很多元组的列表 [ (0, 0), (0, 1)....(行数-1. 列数-1) ]
        self._states = [
            (x, y) for x in range(self.map.shape[0]) for y in range(self.map.shape[1])
            if self.map[x, y] is not MazeCell.WALL
        ]
        # 具体而言，它使用了 zip() 函数将两个可迭代对象合并在一起,组成一个元组，一个是存储了所有状态的列表 _states，
        # 另一个是使用 np.arange(len(self._states)) 生成的索引序列.
        # np.arange(len(self._states)) 生成的是一个从 0 开始、长度与 _states 列表相同的整数序列，用于表示状态在 _states 列表中的位置。
        # 用 {a:b for a,b in ...} 把元组的第一个第二个元素组合成一个键值对, 即(0,0) 对应index 0, (5, 6)对应index(6*7-1)=41
        self._state_to_index = {state: s for state, s in zip(self._states, np.arange(len(self._states)))}

    def step(self, action: int) -> tuple[Position, float, bool, dict]:
        # return 下一state, 当前state的reward, 是否到达horizon或者到达exit
        state, reward, done, info = super().step(action)
        won = self._won()   
        assert not (won and not done)
        info["won"] = won
        return state, reward, done, info

    def reward(self, state: Position, action: int, mean: bool = False) -> float:
        assert action in self.valid_actions(state)

        # terminal state (absorbing): nothing happens
        if self.terminal_state(state):
            reward = 0
        # main objective: minimize the time to exit <=> maximize the negative time to exit
        # => negative reward (penalty) at each step
        else:
            delay = self.map[state].delay
            reward_no_delay = self._reward_step  # self._reward_step = -1
            reward_delay = (1 + delay) * self._reward_step # 如果有delay=6, 那么就是reward就是-7

            # exit!
            # Pay attention: the reward when the exit is reached must be greater than the another walk step.
            # Otherwise, with T corresponding to the shortest path length, the agent is not encouraged to exit the maze.
            # Indeed, the total reward of exiting the maze (without staying there for at least 1 timestep) would be
            # equal to the reward of not exiting the maze.
            next_state = self._next_state(state, action)
            # 如果下个状态是结束状态, 那么还要加上 1,
            if self.terminal_state(next_state):
                reward_no_delay += self._reward_exit
                reward_delay += self._reward_exit
            if mean:
                #  如果mean, 那么  0.5* reward(-1 或者带delay的 -7) + 0.5*reward(不带delay 就是-1)
                reward = self._probability_delay * reward_delay + (1 - self._probability_delay) * reward_no_delay
            else:
                # 根据概率选择reward, 0.5的概率选择reward_delay, 0.5概率选择reward_no_delay
                reward = self._rng.choice(
                    a=[reward_delay, reward_no_delay],
                    p=[self._probability_delay, 1 - self._probability_delay]
                )

        return reward

    def valid_actions(self, state: Position) -> list[Move]: # 每个state ,对应一个能够执行的action列表
        valid_moves = [Move.NOP]

        if not self.terminal_state(state):
            x, y = state

            x_tmp = x - 1
            if x_tmp >= 0 and self.map[x_tmp, y] is not MazeCell.WALL:  # 如果up action之后的next_state在map范围内且不在墙上, 那就能执行up action
                valid_moves.append(Move.UP)

            x_tmp = x + 1
            if x_tmp < self.map.shape[0] and self.map[x_tmp, y] is not MazeCell.WALL:
                valid_moves.append(Move.DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self.map[x, y_tmp] is not MazeCell.WALL:
                valid_moves.append(Move.LEFT)

            y_tmp = y + 1
            if y_tmp < self.map.shape[1] and self.map[x, y_tmp] is not MazeCell.WALL:
                valid_moves.append(Move.RIGHT)

        return valid_moves

    def state_index(self, state: Position) -> int:
        return self._state_to_index[state]

    def _won(self):
        return self.terminal_state(self._current_state) # 调用方法terminal_state(state)

    def terminal_state(self, state: Position) -> bool:  # check current state是否是EXIT state
        exited = self.map[state] is MazeCell.EXIT
        return exited

    def _load_map(self, filepath: Path) -> None:
        with open(filepath) as f:   # 从文件中读取map(用A,B,0,#代表不同状态)
            lines = f.readlines()   # 读取所有行lines, 写入map变量中,  for symbol in line[:-1].split("\t"), 用[:-1]去除最后一个换行符, 用split["\t"]根据制表符将每行分割成一个列表.
                                    # 结果是 ['A', '0', '#', ....], 对每个元素进行MazeCell("A")操作:
                                    # 得到枚举类对应的枚举值, 例如 MazeCell.WALL 或者 MazeCell.EMPTY
        self.map = np.asarray([[MazeCell(symbol) for symbol in line[:-1].split("\t")] for line in lines])

        self._initial_state = np.asarray(self.map == MazeCell.START).nonzero() # 找到地图中起始点的坐标,类型是array, 即(0, 0)
        self._initial_state = (int(self._initial_state[0][0]), int(self._initial_state[1][0])) # 把array转化为一个元组? 
