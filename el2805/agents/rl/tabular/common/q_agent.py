import numpy as np
from abc import ABC
from typing import Any
from el2805.agents.rl.common.rl_agent import RLAgent
from el2805.agents.rl.common.experience import Experience
from el2805.agents.rl.common.utils import get_epsilon
from el2805.environments import TabularRLProblem
from el2805.common.utils import random_decide


class QAgent(RLAgent, ABC):
    """Interface for a RL algorithm learning the Q-function with discrete state and action spaces."""

    def __init__(
            self,
            *,
            environment: TabularRLProblem,
            discount: float,
            learning_rate: float | str,
            alpha: float | None = None,
            epsilon: float | str,
            epsilon_max: float | None = None,
            epsilon_min: float | None = None,
            epsilon_decay_duration: int | None = None,
            delta: float | None = None,
            q_init: float = 0,
            seed: int | None = None
    ):
        """Initializes a tabular QAgent.

        :param environment: RL environment
        :type environment: TabularRLProblem
        :param discount: discount factor
        :type discount: float
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param alpha: parameter for the learning rate decay 1/n(s,a)**alpha
        :type alpha: float, optional
        :param epsilon: probability of exploration (eps-greedy policy) or strategy to calculate it (e.g., "decay")
        :type epsilon: float or str
        :param epsilon_max: initial probability of exploration (eps-greedy policy)
        :type epsilon_max: float, optional
        :param epsilon_min: final probability of exploration (eps-greedy policy)
        :type epsilon_min: float, optional
        :param epsilon_decay_duration: duration of epsilon decay in episodes (eps-greedy policy)
        :type epsilon_decay_duration: int, optional
        :param delta: parameter for epsilon decay in eps-greedy policy
        :type delta: float, optional
        :param q_init: value for Q-function initialization (except for terminal states)
        :type q_init: float, optional
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment, seed=seed)
        self.environment = environment  # to avoid warnings of type hints
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_duration
        self.delta = delta
        self.alpha = alpha
        self.q_init = q_init
        self._exploration_decay = self.delta is not None

        if self.learning_rate != "decay":
            raise NotImplementedError
        else:
            assert self.alpha is not None

        # Notes:
        # - List of 1D ndarray (not just 2D ndarray) because the set of available actions for each state is different.
        # - Q-values of terminal states must be initialized to 0, they will never be updated (see Sutton and Barto).
        #   This allows keeping the update formula the same also when the next state is terminal.
        self._q = [
            (self.q_init if not environment.terminal_state(state) else 0) *
            np.ones(len(self.environment.valid_actions(state)))
            for state in self.environment.states
        ]
        self._n = [np.zeros(len(self.environment.valid_actions(state))) for state in self.environment.states]
        self._last_experience = None

    def q(self, state: Any, action: int) -> float:
        """Returns the Q-function evaluated on the specified (state, action) pair. That is, Q(state,action).

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :return: Q-value of the specified (state,action) pair, Q(state,action)
        :rtype: float
        """
        s = self.environment.state_index(state)
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
        s = self.environment.state_index(state)
        v = max(self._q[s])
        return v

    def compute_action(
            self,
            state: Any,
            *,
            episode: int | None = None,
            explore: bool = True,
            **kwargs
    ) -> int:
        """Calculates the best action according to the agent's policy.

        :param state: state for which the action is desired
        :type state: any
        :param episode: episode. The first episode is supposed to be 1 (not 0).
        :type episode: int
        :param explore: whether to allow exploration or not
        :type explore: bool, optional
        :return: best action according to the agent's policy
        :rtype: int
        """
        _ = kwargs
        assert not (explore and episode is None)
        valid_actions = self.environment.valid_actions(state)

        # Compute epsilon according to exploration strategy
        if explore:
            epsilon = get_epsilon(
                epsilon=self.epsilon,
                episode=episode,
                epsilon_max=self.epsilon_max,
                epsilon_min=self.epsilon_min,
                epsilon_decay_duration=self.epsilon_decay_episodes,
                delta=self.delta
            )
        else:
            epsilon = None

        # Epsilon-greedy policy (or greedy policy if explore=False)
        if explore and random_decide(self._rng, epsilon):
            action = self._rng.choice(valid_actions)
        else:
            s = self.environment.state_index(state)
            v = self.v(state)
            a = self._rng.choice(np.asarray(self._q[s] == v).nonzero()[0])      # random among those with max Q-value
            action = valid_actions[a]

        return action

    def record_experience(self, experience: Experience) -> None:
        """Store a new experience, which is supposed to be used for training.

        :param experience: new experience to store
        :type experience: Experience
        """
        self._last_experience = experience

    def _action_index(self, state: Any, action: int) -> int:
        """Returns the index of a certain action in the list of valid actions in a certain state.

        :param state: state
        :type state: any
        :param action: action
        :type action: int
        :return: index of the specified action in the list of valid actions for the state
        :rtype: int
        """
        valid_actions = self.environment.valid_actions(state)
        return valid_actions.index(action)
