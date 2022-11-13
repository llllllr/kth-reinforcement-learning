import numpy as np
from el2805.lab0.agents.agent import Agent
from el2805.lab0.envs import MDP


class ValueIterationAgent(Agent):
    def __init__(self, env: MDP):
        super().__init__(env)
        self.policy = None

    def compute_action(self, state: np.ndarray):
        pass

    def solve(self):
        pass
