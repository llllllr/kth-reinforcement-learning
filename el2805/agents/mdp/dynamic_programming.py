import numpy as np
from typing import Any
from el2805.agents.mdp.common.mdp_agent import MDPAgent
from el2805.environments import TabularMDP


class DynamicProgramming(MDPAgent):
    def __init__(self, *, environment: TabularMDP):
        super().__init__(environment=environment)
        assert self.environment.finite_horizon()

    def solve(self) -> None:
        n_states = len(self.environment.states)
        u = np.zeros(n_states)
        self.policy = np.zeros((self.environment.horizon, n_states), dtype=np.int32)  # optimal policy (non-stationary)

        for t in range(self.environment.horizon - 1, -1, -1):
            last_time_step = t == self.environment.horizon - 1              # terminal case?
            u_next = u.copy() if not last_time_step else None       # u*_{t+1}

            for s, state in enumerate(self.environment.states):
                # Q_t(s,a) for each a in A_s
                valid_actions = self.environment.valid_actions(state)
                if last_time_step:
                    q = np.asarray([self.environment.reward(state, action, mean=True) for action in valid_actions])
                else:
                    q = np.asarray([self.q(state, action, u_next) for action in valid_actions])

                # u*_t(s)
                u[s] = max(q)

                # store optimal policy (non-stationary, optimal at this time step)
                a_best = q.argmax()     # index of best action for valid actions in this state
                self.policy[t, s] = valid_actions[a_best]

    def compute_action(self, *, state: Any, time_step: int, **kwargs) -> int:
        _ = kwargs
        assert self.policy is not None
        s = self.environment.state_index(state)
        action = self.policy[time_step, s]
        return action
