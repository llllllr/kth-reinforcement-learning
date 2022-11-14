import numpy as np
from el2805.lab0.agents.mdp_agent import MDPAgent
from el2805.lab0.envs import MDP


class DynamicProgrammingAgent(MDPAgent):
    def __init__(self, env: MDP):
        super().__init__(env)
        assert self.env.horizon is not None     # finite-horizon MDP

    def compute_action(self, state: np.ndarray):
        assert self.policy is not None
        s = self.env.state_to_index(state)
        action = self.policy[s]
        return action

    def solve(self):
        u = np.zeros(len(self.env.states))
        self.policy = {}

        # t = T
        for s, state in enumerate(self.env.states):
            # Q(s,a) for each a in A_s
            q = np.asarray([self.env.reward(state, action, mean=True) for action in self.env.valid_actions(state)])

            # u*(s)
            u[s] = max(q)

        # 1 <= t < T (backward)
        for t in range(self.env.horizon-1, 0, -1):
            u_next = u.copy()   # u_{t+1}, at the next time step
            for s, state in enumerate(self.env.states):
                # Q(s,a) for each a in A_s
                valid_actions = self.env.valid_actions(state)
                q = np.asarray([self.q(state, action, u_next) for action in valid_actions])

                # u*(s)
                u[s] = max(q)

                # t=1 => store policy
                # here we assume homogeneous MDPs, so the policy does not depend on the time, and we can read it from
                # the value function and Q-function (t=1)
                if t == 1:
                    a_best = q.argmax()     # index of best action for valid actions in this state
                    self.policy[s] = valid_actions[a_best]
