import numpy as np
from ..envs.mdp import MDP


class MazeDynamicProgramming:
    def __init__(self):
        pass

    def solve(self, env: MDP):
        states = env.valid_states()
        state_to_index = {state: s for state, s in zip(states, np.arange(len(states)))}
        u = np.zeros(len(states))

        # t = T
        for s, state in enumerate(states):
            # Q(s,a)
            q_s = np.asarray([env.reward(state, action) for action in env.valid_actions(state)])

            # u*(s)
            u[s] = q_s.max(initial=-np.inf)

        # 0 <= t < T (backward)
        for t in range(env.horizon-1, -1, -1):
            for s, state in enumerate(states):
                # Q(s,a)
                actions = env.valid_actions(state)
                q_s = np.zeros(len(actions))
                for a, action in enumerate(actions):
                    q_s[a] = env.reward(state, action)
                    for state_new, probability in env.next_states(state, action):
                        s_new = state_to_index[state_new]
                        q_s[a] += probability * u[s_new]

                # u*(s)
                u[s] = q_s.max(initial=-np.inf)

    def compute_action(self, state: np.ndarray):
        pass
