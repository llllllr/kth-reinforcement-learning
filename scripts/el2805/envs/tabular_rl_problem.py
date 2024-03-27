import gym   
            # gym 提供了一个统一的接口,  提供了一系列标准化的环境，
            # 每个环境都代表了一个具体的问题或任务，例如经典的 CartPole、MountainCar、Atari 游戏等。
            # 这些环境包括了状态空间、动作空间、奖励函数等，用户可以根据需要选择合适的环境进行实验和测试。
from abc import ABC, abstractmethod
from typing import Any
from gym.utils.seeding import np_random


class TabularRLProblem(gym.Env, ABC):   # 表格的, discrete RL problem类, 继承自gym.Env 和abstract class ABC
    """Interface for a RL problem with discrete state and action spaces."""

    def __init__(self, horizon: int | None = None):
        """Initializes a RLProblem.

        :param horizon: time horizon, if None then the MDP has infinite horizon
        :type horizon: int, optional
        """
        self.horizon = horizon
        self._rng = None
        self.seed()     # 调用类方法seed(), self._rng, seed = np_random(seed) 通过np_random把self._rng属性赋值, 并放回一个seed变量

    @property  # 代表吗method可以用.方法名来想属性一样调用. 可以用调用属性的形式来调用方法,后面不需要加（）
    @abstractmethod
    def states(self) -> list[Any]:  # 返回包含valid states的列表, 
        """Returns the list of (valid) states in the state space. The invalid states are not included.

        :return: list of valid states
        :rtype: list[any]
        """
        raise NotImplementedError

    @abstractmethod         # 输入一个 state变量, 一个action(int), 输出 对应的reward,类型是float
    def reward(self, state: Any, action: int) -> float:
        """Returns reward received by taking a certain action in a certain state.

        :param state: current state
        :type state: any
        :param action: action taken in the current state
        :type action: int
        :return: reward sample (if mean=False) or mean reward (if mean=True)
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: Any) -> list[int]:  # 返回所有valid actions的列表
        """Returns the valid actions in a certain state.

        :param state: state for which to calculate the valid actions
        :type state: any
        :return: valid actions in the specified state
        :rtype: list[int]
        """
        raise NotImplementedError

    @abstractmethod
    def state_index(self, state: Any) -> int:
        """Returns the index of a certain state in the list of valid states. ????属于states列表中的第几个???

        :param state: state for which the index is desired
        :type state: any
        :return: index of the specified state in the list of valid states
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_state(self, state: Any) -> bool:
        """Returns whether the state is terminal or not.

        :param state: state to be analyzed
        :type state: Any
        :return: whether the state is terminal or not
        :rtype: bool
        """
        raise NotImplementedError

    def finite_horizon(self) -> bool:
        """Returns whether the MDP is finite horizon or not.

        :return: True if the MDP is finite horizon.
        :rtype: bool
        """
        return self.horizon is not None

    def seed(self, seed: int | None = None) -> list[int]: # 返回一个包含seed的列表?

        """Sets the seed of the environment's internal RNG.

        :param seed: seed
        :type seed: int, optional
        """
        self._rng, seed = np_random(seed)
        return [seed]
