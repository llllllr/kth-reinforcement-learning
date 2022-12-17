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
            epsilon: float | str,
            epsilon_max: float | None = None,
            epsilon_min: float | None = None,
            epsilon_decay_episodes: int | None = None,
            delta: float | None = None,
            seed: int | None = None
    ):
        """Initializes a RLAgent.

        :param environment: RL environment
        :type environment: gym.Env
        :param discount: discount factor of the MDP
        :type discount: float
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param epsilon: probability of exploration (eps-greedy policy) or strategy to schedule it
        :type epsilon: float or str
        :param epsilon_max: initial probability of exploration (eps-greedy policy)
        :type epsilon_max: float, optional
        :param epsilon_min: final probability of exploration (eps-greedy policy)
        :type epsilon_min: float, optional
        :param epsilon_decay_episodes: duration of epsilon decay in episodes (eps-greedy policy)
        :type epsilon_decay_episodes: int, optional
        :param delta: exponent in epsilon decay 1/(episode**delta) (eps-greedy policy)
        :type delta: float, optional
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment, discount=discount)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.delta = delta
        self._rng = None
        self.seed(seed)
        _ = self._get_epsilon(1)    # try epsilon to raise errors now if it is not correct

    @abstractmethod
    def update(self) -> dict:
        """Updates the policy (or value function, or Q-function) from stored observations.

        :return: statistics for the update
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def record_experience(self, experience: Experience) -> None:
        """Stores a new experience, which is supposed to be used for training.

        :param experience: new experience to store
        :type experience: Experience
        """
        raise NotImplementedError

    def train(self, n_episodes: int, early_stopping_reward: float | None = None) -> dict:
        """Trains the RL agent for the specified number of episodes.

        :param n_episodes: number of training episodes
        :type n_episodes: int
        :param early_stopping_reward: average reward considered as problem solved
        :type early_stopping_reward: float, optional
        :return: dictionary of training stats (per episode or per time step, depending on the metric)
        :rtype: dict
        """
        stats = self._train_or_test(
            n_episodes=n_episodes,
            train=True,
            early_stopping_reward=early_stopping_reward
        )
        return stats

    def test(self, n_episodes: int, render: bool) -> dict:
        """Tests the RL agent for the specified number of episodes.

        :param n_episodes: number of test episodes
        :type n_episodes: int
        :return: dictionary of test stats (per episode or per time step, depending on the metric)
        :rtype: dict
        """
        stats = self._train_or_test(
            n_episodes=n_episodes,
            train=False,
            render=render
        )
        return stats

    def seed(self, seed: int | None = None) -> None:
        """Sets the seed of the agent's internal RNG.

        :param seed: seed
        :type seed: int, optional
        """
        self._rng = np.random.RandomState(seed)

    def _train_or_test(
            self,
            n_episodes: int,
            train: bool,
            render: bool = False,
            early_stopping_reward: float | None = None
    ) -> dict:
        assert not (train and render)
        stats = defaultdict(list)
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
                        stats[k].append(v)
                episode_reward += reward
                episode_length += 1

                # Update state
                state = next_state

            # Update stats
            stats["episode_reward"].append(episode_reward)
            stats["episode_length"].append(episode_length)

            # Show progress
            avg_episode_length = running_average(stats["episode_length"])[-1]
            avg_episode_reward = running_average(stats["episode_reward"])[-1]
            episodes.set_description(
                f"Episode {episode} - "
                f"Reward: {episode_reward:.1f} - "
                f"Length: {episode_length} - "
                f"Avg reward: {avg_episode_reward:.1f} - "
                f"Avg length: {avg_episode_length:.1f}"
            )

            if early_stopping_reward is not None and avg_episode_reward >= early_stopping_reward:
                print("Early stopping: environment solved!")
                break

        return stats

    def _get_epsilon(self, episode: int) -> float:
        if isinstance(self.epsilon, float):
            epsilon = self.epsilon
        elif self.epsilon == "delta":
            epsilon = 1 / (episode ** self.delta)
        elif self.epsilon == "linear":
            epsilon = max(
                self.epsilon_min,
                self.epsilon_max -
                (self.epsilon_max - self.epsilon_min) * (episode - 1) / (self.epsilon_decay_episodes - 1)
            )
        elif self.epsilon == "exponential":
            epsilon = max(
                self.epsilon_min,
                self.epsilon_max *
                (self.epsilon_min / self.epsilon_max) ** ((episode - 1) / (self.epsilon_decay_episodes - 1))
            )
        else:
            raise NotImplementedError
        return epsilon
