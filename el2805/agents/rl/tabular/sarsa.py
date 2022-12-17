from el2805.agents.rl.tabular.common.q_agent import QAgent


class Sarsa(QAgent):
    def update(self) -> dict:
        # Unpack last experience
        episode = self._last_experience.episode
        state = self._last_experience.state
        action = self._last_experience.action
        reward = self._last_experience.reward
        next_state = self._last_experience.next_state

        # Compute next action
        next_action = self.compute_action(state=next_state, episode=episode)

        # Get indices
        s = self.environment.state_index(state)
        a = self._action_index(state, action)
        s_next = self.environment.state_index(next_state)
        a_next = self._action_index(next_state, next_action)

        # Update Q-function
        self._n[s][a] += 1
        step_size = 1 / (self._n[s][a] ** self.alpha)
        self._q[s][a] += step_size * (reward + self.discount * self._q[s_next][a_next] - self._q[s][a])

        return {}
