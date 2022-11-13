import numpy as np
from el2805.lab0.agents.mdp_agent import MDPAgent
from el2805.lab0.envs import MDP


class ValueIterationAgent(MDPAgent):
    def __init__(self, env: MDP, precision: float = 1e-2):
        super().__init__(env)
        self.precision = precision
        assert self.env.horizon is None and self.env.discount < 1   # infinite-horizon MDP

    def solve(self):
        # (step 1) initialization
        states = self.env.valid_states
        v = np.zeros(len(states))   # V(s) for each s in S
        self.policy = {}

        # (step 2) value improvement
        delta = None
        while delta is None or delta > self.precision * (1 - self.env.discount) / self.env.discount:
            # (step 2.a) update V(s)
            v_old = v.copy()
            for s, state in enumerate(states):
                q = np.asarray([self.q(state, action, v) for action in self.env.valid_actions(state)])
                v[s] = max(q)

            # (step 2.b) calculate improvement
            delta = max(v - v_old)

        # (step 3) store policy
        for s, state in enumerate(states):
            valid_actions = self.env.valid_actions(state)
            q = np.asarray([self.q(state, action, v) for action in valid_actions])
            a_best = q.argmax()     # index of best action for valid actions in this state
            self.policy[s] = valid_actions[a_best]
