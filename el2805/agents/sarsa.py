import numpy as np
from typing import Any
from el2805.agents.rl_agent import RLAgent
from el2805.envs.rl_problem import RLProblem
from el2805.utils import random_decide


class SARSA(RLAgent):
    def __init__(
            self,
            env: RLProblem,
            learning_rate: str | float,
            epsilon: float,
            alpha: float | None = None,
            delta: float | None = None,
            seed: int | None = None
    ):
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
        self.q = [self._rng.randn(len(self.env.valid_actions(state))) for state in self.env.states]
        terminal_states = [state for state in env.states if self.env.terminal_state(state)]
        for state in terminal_states:
            s = self.env.state_index(state)
            self.q[s][:] = 0
        self._n = [np.ones(len(self.env.valid_actions(state))) for state in self.env.states]

    def update(self, state: Any, action: int, reward: float, next_state: Any, next_action: int) -> None:
        s = self.env.state_index(state)
        a = self._action_index(state, action)
        s_next = self.env.state_index(next_state)
        a_next = self._action_index(next_state, next_action)
        step_size = 1 / (self._n[s][a] ** self.alpha)

        self.q[s][a] += step_size * (reward + self.env.discount * self.q[s_next][a_next] - self.q[s][a])
        self._n[s][a] += 1

    def compute_action(self, state: Any, explore: bool = True) -> int:
        valid_actions = self.env.valid_actions(state)

        # eps-greedy policy: exploration mode with epsilon probability
        if explore and random_decide(self._rng, self.epsilon):
            action = self._rng.choice(valid_actions)
        else:
            s = self.env.state_index(state)
            a = self.q[s].argmax()
            action = valid_actions[a]

        return action

    def v(self, state):
        s = self.env.state_index(state)
        v = max(self.q[s])
        return v

    def _action_index(self, state: Any, action: int) -> int:
        valid_actions = self.env.valid_actions(state)
        return valid_actions.index(action)
