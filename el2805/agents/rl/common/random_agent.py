import numpy as np
from typing import Union
from el2805.agents.rl.common.experience import Experience
from el2805.agents.rl.common.rl_agent import RLAgent


class RandomAgent(RLAgent):
    def update(self) -> dict:
        pass

    def record_experience(self, experience: Experience) -> None:
        pass

    def compute_action(self, *, state: np.ndarray, **kwargs) -> Union[int, np.array]:
        _ = kwargs
        _ = state
        action = self.environment.action_space.sample()
        return action
