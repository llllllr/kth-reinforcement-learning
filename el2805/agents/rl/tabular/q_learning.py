from el2805.agents.rl.tabular.q_agent import QAgent


class QLearning(QAgent):
    def update(self) -> dict:
        # Unpack last experience
        state = self._last_experience.state
        action = self._last_experience.action
        reward = self._last_experience.reward
        next_state = self._last_experience.next_state

        # Get indices
        s = self.environment.state_index(state)
        a = self._action_index(state, action)
        s_next = self.environment.state_index(next_state)

        # Update Q-function
        self._n[s][a] += 1
        step_size = 1 / (self._n[s][a] ** self.alpha)
        self._q[s][a] += step_size * (reward + self.discount * max(self._q[s_next]) - self._q[s][a])

        return {}
