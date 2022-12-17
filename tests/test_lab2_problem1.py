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

import unittest
import numpy as np
import gym
import torch
from tqdm import trange
from pathlib import Path


class DQNTestCase(unittest.TestCase):
    model_path = Path(__file__).parent.parent / "results" / "lab2" / "problem1" / "neural-network-1.pth"

    def test_trained_model(self):
        n_episodes = 50
        confidence_pass = 50

        environment = gym.make('LunarLander-v2')
        environment.reset()
        model = torch.load(self.model_path)

        episode_rewards = []
        episodes = trange(n_episodes, desc='Episode: ', leave=True)
        for episode in episodes:
            episodes.set_description(f"Episode {episode}")
            done = False
            state = environment.reset()
            episode_reward = 0.
            while not done:
                state = torch.as_tensor(
                    data=state.reshape((1,) + state.shape),
                    dtype=torch.float32
                )
                q_values = model(state)
                action = q_values.argmax().item()
                next_state, reward, done, _ = environment.step(action)
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            environment.close()

        avg_reward = np.mean(episode_rewards)
        confidence = np.std(episode_rewards) * 1.96 / np.sqrt(n_episodes)
        self.assertTrue(avg_reward - confidence >= confidence_pass)


if __name__ == '__main__':
    unittest.main()
