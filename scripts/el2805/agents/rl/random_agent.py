import numpy as np
from typing import Union
from el2805.agents.rl.rl_agent import RLAgent
from el2805.agents.rl.utils import Experience


class RandomAgent(RLAgent):
    def update(self) -> dict:
        pass

    def record_experience(self, experience: Experience) -> None:
        pass

    def compute_action(self, **kwargs) -> Union[int, np.array]:
        _ = kwargs
        action = self.environment.action_space.sample()
        return action
