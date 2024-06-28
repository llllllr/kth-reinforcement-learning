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
import matplotlib.pyplot as plt

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

    def train(self, result_dir_name: str, n_episodes: int, early_stop_reward: float | None = None ) -> dict:
        """Trains the RL agent for the specified number of episodes.

        :param n_episodes: number of training episodes
        :type n_episodes: int
        :param early_stop_reward: average reward considered as problem solved
        :type early_stop_reward: float, optional
        :return: dictionary of training stats (per episode or per time step, depending on the metric)
        :rtype: dict
        """
        stats = self._train_or_test(
            result_dir_name=result_dir_name,
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
            result_dir_name: str,
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
        # figure, axes = plt.subplots(nrows=3, ncols=int(n_episodes/100))

        for episode in episodes:
            # Reset environment data and initialize variables
            done = False
            ref_spec_force, ref_ang_vel, state, state_unwrapped = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            episode_diff_between_ref_f_and_traj_f = [abs(ref_spec_force[episode_length] - state_unwrapped[5])]
            episode_posi_from_zero = [abs(state_unwrapped[2])]

            episode_diff_ang_vel = [abs(ref_ang_vel[episode_length] - state_unwrapped[3])]
            episode_angle_from_zero = [abs(state_unwrapped[4])]

            if render:
                self.environment.render()

            # Run the whole episode, from T=0 to T=1000
            while not done:
                # Interact with the environment
                action = self.compute_action(state=state, episode=episode, explore=train)
                next_state, next_state_unwrapped, reward, done = self.environment.step(action)
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
                    #  stats["critic_loss"] = [loss for episode1_epoche1, loss for episode1_epoche2 ,.....loss for epoche10]同一道轨迹, 训练十次
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

                episode_diff_between_ref_f_and_traj_f += abs(ref_spec_force[episode_length] -  next_state_unwrapped[5])
                episode_posi_from_zero += abs( next_state_unwrapped[2])
                episode_diff_ang_vel += abs(ref_ang_vel[episode_length] -  next_state_unwrapped[3])
                episode_angle_from_zero += abs( next_state_unwrapped[4])

                
            if episode == 1:
                max_reward = episode_reward
            if episode_reward > max_reward:
                max_reward = episode_reward
                max_episode = episode
                current_path = Path(__file__).parent.parent.parent.parent.parent

                torch.save(self.actor, current_path / result_dir_name / "actor_with_max_reward.pth")
                torch.save(self.critic, current_path / result_dir_name / "critic_with_max_reward.pth")
            '''
            if episode % 100 == 0:
                done = False
                ref_accels, state = self.environment.reset()
                test_reward = 0
                traj_acc = [state[0]]
                traj_vel = [state[1]]
                traj_pos = [state[2]]
                while not done:
                    action = self.compute_action_test(state=state)
                    next_state, reward, done = self.environment.step(action)
                    traj_acc.append(next_state[0])
                    traj_vel.append(next_state[1])
                    traj_pos.append(next_state[2])

                    test_reward += reward
                    state = next_state

                ax = axes[0, int(episode/100) - 1]
                t = np.arange(0, 10.0, 0.01)
                ax.plot(t, ref_accels[:len(t)], color='blue', label='reference')
                ax.plot(t, traj_acc[:len(t)], color='red', label='real_accl')
                ax.set_xlabel('Time')
                ax.set_ylabel('Acceleration')
                ax.set_title('Reward: '+str(round(test_reward)))
                ax.grid(True)
                ax.legend()
                
                ax = axes[1, int(episode/100) - 1]
                ax.plot(t, traj_vel[:len(t)])
                ax.set_xlabel('Time')
                ax.set_ylabel('Velocity')
                ax.grid(True)

                ax = axes[2, int(episode/100) - 1]
                ax.plot(t, traj_pos[:len(t)])
                ax.set_xlabel('Time')
                ax.set_ylabel('Position')
                ax.grid(True)
'''
            # if episode % 10 == 0:
            #     print()
            # Update stats
            
            stats["episode_reward"].append(episode_reward.item())
            stats["episode_length"].append(episode_length)
            stats["episode_diff_spec_force"].append((episode_diff_between_ref_f_and_traj_f/episode_length).item())
            stats["episode_posi_from_zero"].append((episode_posi_from_zero/episode_length).item())

            stats["episode_diff_ang_vel"].append((episode_diff_ang_vel/episode_length).item())
            stats["episode_angle_from_zero"].append((episode_angle_from_zero/episode_length).item())

            assert len(stats["episode_diff_ang_vel"]) == len(stats["episode_length"])

            # Show progress
            avg_episode_length = running_average(stats["episode_length"])[-1]
            # print(stats["episode_reward"]), return array in list:  [array([-2160091.79174874])]
            # avg_episode_reward = running_average(np.concatenate(stats["episode_reward"]))[-1]
            avg_episode_reward = running_average(stats["episode_reward"])[-1]
            # avg_ref = running_average(stats["episode_diff_between_ref_a_and_traj_a"])[-1]

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
        # figure.savefig(result_dir_name / "test_during_train.png")
        # plt.show()

        figure2, axes2 = plt.subplots(nrows=4, ncols=5)
        for i in range(5):
            base_env = self.environment.unwrapped
            done = False
            ref_f, ref_ang_vel, state = base_env.reset()
            test_reward = 0

            traj_f = [state[5]]
            traf_ang_vel = [state[3]]
            traj_pos = [state[2]]
            traj_angle = [state[4]]


            while not done:
                action = self.compute_action_test(state=state)
                next_state, reward, done = base_env.step(action)
                traj_f.append(next_state[5])
                traf_ang_vel.append(next_state[3])
                traj_pos.append(next_state[2])
                traj_angle.append(next_state[4])

                test_reward += reward
                state = next_state

            ax = axes2[0, i]
            t = np.arange(0, 10.0, 0.01)
            ax.plot(t, ref_f[:len(t)], color='blue', label='reference')
            ax.plot(t, traj_f[:len(t)], color='red', label='real_accl')
            ax.set_xlabel('Time')
            ax.set_ylabel('Spec_Force')
            ax.set_title('Rew:'+str(round(test_reward)))
            ax.grid(True)
            ax.legend()
            
            ax = axes2[1, i]
            ax.plot(t, ref_ang_vel[:len(t)], color='blue', label='reference')
            ax.plot(t, traj_angle[:len(t)], color='red', label='real_accl')
            ax.set_xlabel('Time')
            ax.set_ylabel('Angular_Velocity')
            ax.grid(True)

            ax = axes2[2, i]
            ax.plot(t, traj_pos[:len(t)])
            ax.set_xlabel('Time')
            ax.set_ylabel('x Position')
            ax.grid(True)
            
            ax = axes2[3, i]
            ax.plot(t, traj_angle[:len(t)])
            ax.set_xlabel('Time')
            ax.set_ylabel('angle along y')
            ax.grid(True)

        figure2.savefig(result_dir_name / "test_after_train.png")
        plt.show()
        return stats
