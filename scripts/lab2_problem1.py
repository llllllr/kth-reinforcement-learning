# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#
# Modified by: [Franco Ruggeri - fruggeri@kth.se]

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from el2805.utils import running_average


def main():
    env = gym.make('LunarLander-v2')    # with discrete actions
    env.reset()

    # Parameters
    n_episodes = 100                                # number of episodes
    discount_factor = 0.95                          # value of the discount factor
    running_avg_window_len = 50                     # running average of 50 episodes
    n_actions = env.action_space.n                  # number of available actions
    state_dim = len(env.observation_space.high)     # State dimensionality

    # Data to compute the average episodic reward and the average number of steps per episode
    episode_rewards = []        # total reward per episode
    episode_lengths = []        # number of steps per episode

    # Random agent initialization
    agent = RandomAgent(n_actions)

    # Training process
    episodes = trange(n_episodes, desc='Episode: ', leave=True)
    for i in episodes:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()
        episode_reward = 0.
        episode_length = 0

        # Run episode
        while not done:
            action = agent.forward(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            episode_length += 1

        # Append episode reward and total number of steps
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Updates the tqdm update bar with fresh information
        average_reward = running_average(episode_rewards, running_avg_window_len)[-1]
        average_n_steps = running_average(episode_lengths, running_avg_window_len)[-1]
        episodes.set_description(
            f"Episode {i} - "
            f"Reward/Steps: {episode_reward:.1f}/{episode_length} - "
            f"Avg. Reward/Steps: {average_reward:.1f}/{average_n_steps}"
        )
    env.close()

    # Plot rewards
    x = np.arange(1, n_episodes+1)
    figure, axes = plt.subplots()
    axes.plot(x, episode_rewards, label='Episode reward')
    axes.plot(x, running_average(episode_rewards, running_avg_window_len), label='Avg. episode reward')
    axes.set_xlabel('Episodes')
    axes.set_ylabel('Episode reward')
    axes.set_title('Episode rewards vs Episodes')
    axes.legend()
    figure.show()

    # Plot number of steps
    figure, axes = plt.subplots()
    axes.plot(x, episode_lengths, label='Steps per episode')
    axes.plot(x, running_average(episode_lengths, running_avg_window_len), label='Avg. number of steps per episode')
    axes.set_xlabel('Episodes')
    axes.set_ylabel('Episode length')
    axes.set_title('Episode lengths vs Episodes')
    axes.legend()
    figure.show()


if __name__ == "__main__":
    main()
