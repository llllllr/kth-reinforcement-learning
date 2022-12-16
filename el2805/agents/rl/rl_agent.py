import gym
import numpy as np
from abc import ABC, abstractmethod
from tqdm import trange
from collections import defaultdict
from el2805.agents.agent import Agent
from el2805.agents.rl.utils import Experience
from el2805.agents.utils import running_average


class RLAgent(Agent, ABC):
    """Interface for a RL algorithm."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            learning_rate: float | str,
            seed: int | None = None
    ):
        """Initializes a RLAgent.

        :param environment: RL environment
        :type environment: gym.Env
        :param discount: discount factor of the MDP
        :type discount: float
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment, discount=discount)
        self.learning_rate = learning_rate
        self._rng = None
        self.seed(seed)

    @abstractmethod
    def update(self) -> dict:
        """Updates the policy (or value function, or Q-function) from stored observations.

        :return: statistics for the update
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def record_experience(self, experience: Experience) -> None:
        """Store a new experience, which is supposed to be used for training.

        :param experience: new experience to store
        :type experience: Experience
        """
        raise NotImplementedError

    def _train_or_test(self, n_episodes: int, train: bool, render: bool) -> dict:
        history_stats = defaultdict(list)
        episodes = trange(1, n_episodes + 1, desc='Episode: ', leave=True)

        for episode in episodes:
            # Reset environment data and initialize variables
            done = False
            state = self.environment.reset()
            episode_reward = 0.
            episode_length = 0
            if render:
                self.environment.render()

            # Run episode
            while not done:
                # Interact with the environment
                action = self.compute_action(state=state, episode=episode, explore=train)
                next_state, reward, done, _ = self.environment.step(action)
                if render:
                    self.environment.render()

                # Update policy
                if train:
                    experience = Experience(
                        episode=episode,
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )
                    self.record_experience(experience)
                    update_stats = self.update()

                    # Update stats
                    for k, v in update_stats.items():
                        history_stats[k].append(v)
                episode_reward += reward
                episode_length += 1

                # Update state
                state = next_state

            # Update stats
            history_stats["episode_reward"].append(episode_reward)
            history_stats["episode_length"].append(episode_length)

            # Show progress
            avg_episode_length = running_average(history_stats["episode_length"])[-1]
            avg_episode_reward = running_average(history_stats["episode_reward"])[-1]
            episodes.set_description(
                f"Episode {episode} - "
                f"Reward: {episode_reward:.1f} - "
                f"Length: {episode_length} - "
                f"Avg reward: {avg_episode_reward:.1f} - "
                f"Avg length: {avg_episode_length:.1f}"
            )

        # TODO: early stopping?

        return history_stats

    def train(self, n_episodes: int) -> dict:
        stats = self._train_or_test(
            n_episodes=n_episodes,
            train=True,
            render=False
        )
        return stats

    def test(self, n_episodes: int, render: bool) -> dict:
        stats = self._train_or_test(
            n_episodes=n_episodes,
            train=False,
            render=render
        )
        return stats

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.RandomState(seed)
