import numpy as np
from el2805.agents.agent import Agent


class RandomAgent(Agent):
    def compute_action(self, *, state: np.ndarray, **kwargs) -> int:
        _ = kwargs
        _ = state
        action = self.environment.action_space.sample()
        return action
