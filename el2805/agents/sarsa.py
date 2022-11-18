import numpy as np
from typing import Any
from el2805.agents.rl_agent import RLAgent
from el2805.envs.rl_problem import RLProblem
from el2805.utils import random_decide


class SARSA(RLAgent):
    def __init__(self, env: RLProblem, learning_rate: float, epsilon: float):
        super().__init__(env, learning_rate)
        self.epsilon = epsilon

        # list of 1D ndarray and not 2D ndarray because the set of available actions for each state is different
        self.q = [np.zeros(len(self.env.valid_actions(state))) for state in self.env.states]

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        next_action = self.compute_action(next_state, explore=True)

        s = self.env.state_index(state)
        a = self._action_index(state, action)
        s_next = self.env.state_index(next_state)
        a_next = self._action_index(state, next_action)

        self.q[s][a] += self.learning_rate * (reward + self.env.discount * self.q[s_next][a_next] - self.q[s][a])

    def compute_action(self, state: Any, explore: bool = False) -> int:
        valid_actions = self.env.valid_actions(state)

        # eps-greedy policy: exploration mode with epsilon probability
        if explore and random_decide(self._rng, self.epsilon):
            action = self._rng.choice(valid_actions)
        else:
            s = self.env.state_index(state)
            a = self.q[s].argmax()
            action = valid_actions[a]

        return action

    def _action_index(self, state: Any, action: int) -> int:
        valid_actions = self.env.valid_actions(state)
        return valid_actions[action]
