import gym
import numpy as np
import torch
from abc import ABC, abstractmethod
from tqdm import trange
from collections import defaultdict
from el2805.agents.agent import Agent
from el2805.agents.utils import running_average
from el2805.agents.rl.utils import Experience
from pathlib import Path

class RLAgent(Agent, ABC):
    """Interface for a RL algorithm."""

    def __init__(self, environment: gym.Env, seed: int | None = None):
        """Initializes a RLAgent.

        :param environment: RL environment
        :type environment: gym.Env
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment)
        self._rng = None
        self.seed(seed)

    def seed(self, seed: int | None) -> None:
        """Sets the seed of the agent's internal RNG.

        :param seed: seed
        :type seed: int, optional
        """
        self._rng = np.random.RandomState(seed)
        self.environment.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def reset(self, seed: int | None = None) -> None:
        """Resets the agent and optionally sets a new seed.

        :param seed: seed
        :type seed: int, optional
        """
        self.seed(seed)

    @abstractmethod
    def update(self) -> dict:
        """Updates the policy (or value function, or Q-function) from stored observations. This function is called
        after each interaction with the environment.

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

    def train(self, n_episodes: int, early_stop_reward: float | None = None) -> dict:
        """Trains the RL agent for the specified number of episodes.

        :param n_episodes: number of training episodes
        :type n_episodes: int
        :param early_stop_reward: average reward considered as problem solved
        :type early_stop_reward: float, optional
        :return: dictionary of training stats (per episode or per time step, depending on the metric)
        :rtype: dict
        """
        stats = self._train_or_test(
            n_episodes=n_episodes,
            train=True,
            early_stop_reward=early_stop_reward
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



    def _train_or_test(
            self,
            n_episodes: int,
            train: bool,
            render: bool = False,
            early_stop_reward: float | None = None
    ) -> dict:
        assert not (train and render)
        stats = defaultdict(list)
        episodes = trange(1, n_episodes + 1, desc='Episode: ', leave=True)
        max_reward = 0
        max_episode = 1
        
        for episode in episodes:
            # Reset environment data and initialize variables
            done = False
            ref_accels, state = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            episode_diff_between_ref_a_and_traj_a = [abs(ref_accels[episode_length] - state[0])]
            episode_posi_from_zero = [abs(state[2])]

            if render:
                self.environment.render()

            # Run the whole episode, from T=0 to T=1000
            while not done:
                # Interact with the environment
                action = self.compute_action(state=state, episode=episode, explore=train)
                next_state, reward, done = self.environment.step(action)
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
                    self.record_experience(experience)  #     def record_experience(self, experience: Experience) -> None: self._episodic_buffer.append(experience)
                    #  stats["critic_loss"] = [loss for episode1_epoche1, loss for episode1_epoche2 ,..... loss for epoche10]
                    #  stats["actor_loss"].append(actor_loss.item()), 
                    #  type: dict: ['loss'] -> list of loss in one epoche.
                    # in update()-function, onlyif episode is done, then do 10 times updates for params in both NNs
                    update_stats = self.update()

                    # Update stats, enumerate key and value in dict
                    for k, v in update_stats.items():
                        if isinstance(v, list):
                            stats[k].extend(v)
                        else:
                            stats[k].append(v) # update one-episode statistics in the total-statistics
                
                episode_reward += reward # sum the reward for each step in environment
                episode_length += 1

                # Update state
                state = next_state

                episode_diff_between_ref_a_and_traj_a += abs(ref_accels[episode_length] - state[0])
                episode_posi_from_zero += abs(state[2])
                
            if episode == 1:
                max_reward = episode_reward
            if episode_reward > max_reward:
                max_reward = episode_reward
                max_episode = episode
                torch.save(self.actor, Path(__file__).parent.parent.parent.parent.parent / "results_version1_backup_to_0327_3"  / "actor_with_max_reward.pth")
                torch.save(self.critic, Path(__file__).parent.parent.parent.parent.parent / "results_version1_backup_to_0327_3"  / "critic_with_max_reward.pth")

            # Update stats
            
            stats["episode_reward"].append(episode_reward.item())
            stats["episode_length"].append(episode_length)
            stats["episode_diff_between_ref_a_and_traj_a"].append((episode_diff_between_ref_a_and_traj_a/episode_length).item())
            stats["episode_posi_from_zero"].append((episode_posi_from_zero/episode_length).item())
            assert len(stats["episode_diff_between_ref_a_and_traj_a"]) == len(stats["episode_length"])

            # Show progress
            avg_episode_length = running_average(stats["episode_length"])[-1]
            # print(stats["episode_reward"]), return array in list:  [array([-2160091.79174874])]
            # avg_episode_reward = running_average(np.concatenate(stats["episode_reward"]))[-1]
            avg_episode_reward = running_average(stats["episode_reward"])[-1]
            avg_ref = running_average(stats["episode_diff_between_ref_a_and_traj_a"])[-1]

            episodes.set_description(
                f"Episode {episode} - "
                f"Reward: {episode_reward.item():.1f} - "
                f"Length: {episode_length} - "
                f"Avg reward: {avg_episode_reward:.1f} - "
                f"Avg length: {avg_episode_length:.1f} - "
                f"Max reward Episode: {max_episode}"
            )

            if early_stop_reward is not None and avg_episode_reward >= early_stop_reward:
                print("Early stopping: environment solved!")
                break

        return stats
