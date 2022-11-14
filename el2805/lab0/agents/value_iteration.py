import numpy as np
from el2805.lab0.agents.mdp_agent import MDPAgent
from el2805.lab0.envs import MDP


class ValueIterationAgent(MDPAgent):
    def __init__(self, env: MDP, precision: float = 1e-2):
        super().__init__(env)
        self.precision = precision
        self._policy = None
        assert self.env.infinite_horizon()

    def policy(self) -> np.ndarray:
        return self._policy

    def solve(self) -> None:
        # value improvement
        n_states = len(self.env.states)
        v = np.zeros(n_states)  # V(s) for each s in S
        delta = None
        while delta is None or delta > self.precision * (1 - self.env.discount) / self.env.discount:
            # update V(s)
            v_old = v.copy()
            for s, state in enumerate(self.env.states):
                q = np.asarray([self.q(state, action, v) for action in self.env.valid_actions(state)])
                v[s] = max(q)

            # calculate value improvement
            delta = max(v - v_old)

        # store eps-optimal policy
        self._policy = np.zeros(n_states, dtype=np.int32)    # eps-optimal policy (stationary)
        for s, state in enumerate(self.env.states):
            valid_actions = self.env.valid_actions(state)
            q = np.asarray([self.q(state, action, v) for action in valid_actions])
            a_best = q.argmax()     # index of best action for valid actions in this state
            self._policy[s] = valid_actions[a_best]

    def compute_action(self, state: np.ndarray, time_step: int) -> int:
        assert self._policy is not None
        _ = time_step   # not used, infinite-horizon MDPs have a stationary optimal policy
        s = self.env.state_to_index(state)
        action = self._policy[s]
        return action
