import numpy as np
from typing import Any
from el2805.agents.mdp.mdp_agent import MDPAgent
from el2805.environments.mdp import MDP


class ValueIteration(MDPAgent):
    def __init__(self, env: MDP, discount: float, precision: float):
        super().__init__(env, discount)
        self.discount = discount
        self.precision = precision
        self._v = np.zeros(len(self.env.states))     # V(s) for each s in S

    def solve(self) -> None:
        # value improvement
        n_states = len(self.env.states)
        delta = None
        while delta is None or delta > self.precision * (1 - self.discount) / self.discount:
            # update V(s)
            v_old = self._v.copy()
            for s, state in enumerate(self.env.states):
                q = np.asarray([self.q(state, action, self._v) for action in self.env.valid_actions(state)])
                self._v[s] = max(q)

            # calculate value improvement
            delta = np.linalg.norm(self._v - v_old, ord=np.inf)

        # store eps-optimal policy
        self.policy = np.zeros(n_states, dtype=np.int32)    # eps-optimal policy (stationary)
        for s, state in enumerate(self.env.states):
            valid_actions = self.env.valid_actions(state)
            q = np.asarray([self.q(state, action, self._v) for action in valid_actions])
            a_best = q.argmax()     # index of best action for valid actions in this state
            self.policy[s] = valid_actions[a_best]

    def compute_action(self, state: Any, **kwargs) -> int:
        assert self.policy is not None
        s = self.env.state_index(state)
        action = self.policy[s]
        return action

    def v(self, state: Any) -> float:
        s = self.env.state_index(state)
        v = self._v[s]
        return v
