import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from el2805.agents.rl_agent import RLAgent
from el2805.envs.rl_problem import RLProblem
from el2805.utils import random_decide


class QAgent(RLAgent, ABC):
    """Interface for a RL algorithm approximating the Q-function."""

    def __init__(
            self,
            env: RLProblem,
            learning_rate: float | str,
            epsilon: float | str,
            alpha: float | None = None,
            delta: float | None = None,
            seed: int | None = None
    ):
        """
        :param env: RL environment
        :type env: RLProblem
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param epsilon: probability of exploration (eps-greedy policy) or strategy to calculate it (e.g., "decay")
        :type epsilon: float or str
        :param alpha: parameter for the learning rate decay 1/n(s,a)**alpha
        :type alpha: float, optional
        :param delta: parameter for epsilon decay in eps-greedy policy
        :type delta: float, optional
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(env, learning_rate, seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.delta = delta
        self._exploration_decay = self.delta is not None

        if self.learning_rate != "decay":
            raise NotImplementedError
        else:
            assert self.alpha is not None

        assert (self.epsilon == "decay") is (self.delta is not None)

        # note: list of 1D ndarray and not 2D ndarray because the set of available actions for each state is different
        self._q = [np.zeros(len(self.env.valid_actions(state))) for state in self.env.states]
        self._n = [np.ones(len(self.env.valid_actions(state))) for state in self.env.states]

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Updates the Q-function from new observation(s).

        The parameters depend on the algorithm. For example:
        - Q-learning requires tuples (state, action, reward, next_state).
        - SARSA requires tuples (state, action, reward, next_state, next_action).
        """
        raise NotImplementedError

    def q(self, state: Any, action: int) -> float:
        """Returns the Q-function evaluated on the specified (state, action) pair. That is, Q(state,action).

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :return: Q-value of the specified (state,action) pair, Q(state,action)
        :rtype: float
        """
        s = self.env.state_index(state)
        a = self._action_index(state, action)
        q = self._q[s][a]
        return q

    def v(self, state: Any) -> float:
        """Returns the value function evaluated on the specified state. That is, V(state).

        :param state: state whose value is desired
        :type state: any
        :return: value of the specified state, V(state)
        :rtype: float
        """
        s = self.env.state_index(state)
        v = max(self._q[s])
        return v

    def compute_action(self, state: Any, episode: int, explore: bool = True) -> int:
        """Calculates the best action according to the agent's policy.

        :param state: state for which the action is desired
        :type state: any
        :param episode: episode
        :type episode: int
        :param explore: whether to allow exploration or not
        :type explore: bool, optional
        :return: best action according to the agent's policy
        :rtype: int
        """
        valid_actions = self.env.valid_actions(state)
        epsilon = 1 / (episode ** self.delta) if self.epsilon == "decay" else self.epsilon

        # eps-greedy policy: exploration mode with epsilon probability
        if explore and random_decide(self._rng, epsilon):
            action = self._rng.choice(valid_actions)
        else:
            s = self.env.state_index(state)
            a = self._q[s].argmax()
            action = valid_actions[a]

        return action

    def _action_index(self, state: Any, action: int) -> int:
        """Returns the index of a certain action in the list of valid actions in a certain state.

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :return: index of the specified action in the list of valid actions for the state
        :rtype: int
        """
        valid_actions = self.env.valid_actions(state)
        return valid_actions.index(action)

    def seed(self, seed: int | None = None) -> None:
        """Sets the seed of the agent's internal RNG.

        :param seed: seed
        :type seed: int, optional
        """
        self._rng = np.random.RandomState(seed)
