import numpy as np
from typing import Any
from el2805.agents.mdp_agent import MDPAgent
from el2805.envs.mdp import MDP


class DynamicProgramming(MDPAgent):
    def __init__(self, env: MDP):
        super().__init__(env)
        assert self.env.finite_horizon()

    def solve(self) -> None:
        n_states = len(self.env.states)
        u = np.zeros(n_states)
        self.policy = np.zeros((self.env.horizon, n_states), dtype=np.int32)    # optimal policy (non-stationary)

        for t in range(self.env.horizon-1, -1, -1):
            last_time_step = t == self.env.horizon - 1              # terminal case?
            u_next = u.copy() if not last_time_step else None       # u*_{t+1}

            for s, state in enumerate(self.env.states):
                # Q_t(s,a) for each a in A_s
                valid_actions = self.env.valid_actions(state)
                if last_time_step:
                    q = np.asarray([self.env.reward(state, action, mean=True) for action in valid_actions])
                else:
                    q = np.asarray([self.q(state, action, u_next) for action in valid_actions])

                # u*_t(s)
                u[s] = max(q)

                # store optimal policy (non-stationary, optimal at this time step)
                a_best = q.argmax()     # index of best action for valid actions in this state
                self.policy[t, s] = valid_actions[a_best]

    def compute_action(self, state: Any, time_step: int, **kwargs) -> int:
        assert self.policy is not None
        s = self.env.state_index(state)
        action = self.policy[time_step, s]
        return action
