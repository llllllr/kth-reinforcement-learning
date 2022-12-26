import numpy as np
from typing import Any
from el2805.agents.mdp.mdp_agent import MDPAgent
from el2805.envs.tabular_mdp import TabularMDP


class ValueIteration(MDPAgent):
    def __init__(self, environment: TabularMDP, discount: float, precision: float):
        super().__init__(environment=environment, discount=discount)
        self.discount = discount
        self.precision = precision
        self._v = np.zeros(len(self.environment.states))     # V(s) for each s in S

    def solve(self) -> None:
        # value improvement
        n_states = len(self.environment.states)
        delta = None
        while delta is None or delta > self.precision * (1 - self.discount) / self.discount:
            # update V(s)
            v_old = self._v.copy()
            for s, state in enumerate(self.environment.states):
                q = np.asarray([self.q(state, action, self._v) for action in self.environment.valid_actions(state)])
                self._v[s] = max(q)

            # calculate value improvement
            delta = np.linalg.norm(self._v - v_old, ord=np.inf)

        # store eps-optimal policy
        self.policy = np.zeros(n_states, dtype=np.int32)    # eps-optimal policy (stationary)
        for s, state in enumerate(self.environment.states):
            valid_actions = self.environment.valid_actions(state)
            q = np.asarray([self.q(state, action, self._v) for action in valid_actions])
            a_best = q.argmax()     # index of best action for valid actions in this state
            self.policy[s] = valid_actions[a_best]

    def compute_action(self, *, state: Any, **kwargs) -> int:
        _ = kwargs
        assert self.policy is not None
        s = self.environment.state_index(state)
        action = self.policy[s]
        return action

    def v(self, state: Any) -> float:
        s = self.environment.state_index(state)
        v = self._v[s]
        return v
