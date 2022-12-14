from typing import Any
from el2805.agents.rl.tabular.q_agent import QAgent


class Sarsa(QAgent):
    def update(self, *, state: Any, action: int, reward: float, next_state: Any, next_action: int, **kwargs) -> None:
        _ = kwargs
        s = self.environment.state_index(state)
        a = self._action_index(state, action)
        s_next = self.environment.state_index(next_state)
        a_next = self._action_index(next_state, next_action)

        self._n[s][a] += 1
        step_size = 1 / (self._n[s][a] ** self.alpha)

        self._q[s][a] += step_size * (reward + self.discount * self._q[s_next][a_next] - self._q[s][a])
