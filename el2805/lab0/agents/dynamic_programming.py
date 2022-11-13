import numpy as np
from el2805.lab0.envs import MDP


class DynamicProgrammingAgent:
    def __init__(self, env: MDP):
        self.env = env
        self.policy = None
        assert self.env.horizon < np.inf

    def solve(self):
        states = self.env.valid_states
        u = np.zeros(len(states))
        self.policy = {}

        # t = T
        for s, state in enumerate(states):
            # Q(s,a)
            # note that we ask for the mean reward instead of a reward sample, so as to support random rewards
            q_s = np.asarray([self.env.reward(state, action, mean=True) for action in self.env.valid_actions(state)])

            # u*(s)
            u[s] = max(q_s)

        # 1 <= t < T (backward)
        for t in range(self.env.horizon-1, 0, -1):
            for s, state in enumerate(states):
                # Q(s,a)
                valid_actions = self.env.valid_actions(state)
                q_s = np.zeros(len(valid_actions))
                for a, action in enumerate(valid_actions):
                    next_states, transition_probabilities = self.env.next_states(state, action)
                    s_next = [self.env.state_to_index(next_state) for next_state in next_states]
                    q_s[a] = self.env.reward(state, action, mean=True) + transition_probabilities.dot(u[s_next])

                # u*(s)
                u[s] = max(q_s)

                # t=1 => store policy
                # here we assume homogeneous MDPs, so the policy does not depend on the time and we can read it from
                # the V-function and Q-function (t=1)
                if t == 1:
                    a_s = q_s.argmax()  # index for valid actions in this state
                    self.policy[s] = valid_actions[a_s]

    def compute_action(self, state: np.ndarray):
        assert self.policy is not None
        s = self.env.state_to_index(state)
        action = self.policy[s]
        return action
