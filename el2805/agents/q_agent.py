import numpy as np
from abc import ABC
from typing import Any
from el2805.agents.rl_agent import RLAgent
from el2805.envs.rl_problem import RLProblem
from el2805.utils import random_decide


class QAgent(RLAgent, ABC):
    """Interface for a RL algorithm approximating the Q-function."""

    def __init__(
            self,
            env: RLProblem,
            learning_rate: str | float,
            epsilon: float,
            alpha: float | None = None,
            delta: float | None = None,
            seed: int | None = None
    ):
        """
        :param env: RL environment
        :type env: RLProblem
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param epsilon: parameter for eps-greedy policy (probability of exploration)
        :type epsilon: float
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

        # initialize to random values except for the terminal states, whose value must be 0
        # note: list of 1D ndarray and not 2D ndarray because the set of available actions for each state is different
        self._q = [self._rng.randn(len(self.env.valid_actions(state))) for state in self.env.states]
        terminal_states = [state for state in env.states if self.env.terminal_state(state)]
        for state in terminal_states:
            s = self.env.state_index(state)
            self._q[s][:] = 0
        self._n = [np.ones(len(self.env.valid_actions(state))) for state in self.env.states]

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

    def compute_action(self, state: Any, explore: bool = True) -> int:
        """Calculates the best action according to the agent's policy.

        :param state: state for which the action is desired
        :type state: any
        :param explore: whether to allow exploration or not
        :type explore: bool, optional
        :return: best action according to the agent's policy
        :rtype: int
        """
        valid_actions = self.env.valid_actions(state)

        # eps-greedy policy: exploration mode with epsilon probability
        if explore and random_decide(self._rng, self.epsilon):
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
