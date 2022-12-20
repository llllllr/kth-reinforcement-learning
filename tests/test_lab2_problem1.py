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
from el2805.agents.rl.deep import DQN
from el2805.agents.rl.deep.utils import get_device


class DQNTestCase(unittest.TestCase):
    seed = 1
    environment = None

    def setUp(self):
        self.environment = gym.make('LunarLander-v2')
        self.environment.seed(self.seed)

    def tearDown(self):
        self.environment.close()

    def test_training(self):
        # Hyper-parameters
        seed = 1
        n_episodes = 1000
        discount = .99
        epsilon = "exponential"
        epsilon_max = .99
        epsilon_min = .05
        epsilon_decay_episodes = int(.9 * n_episodes)
        learning_rate = 5e-4
        batch_size = 64
        replay_buffer_size = 10000
        replay_buffer_min = int(.2 * replay_buffer_size)
        target_update_period = replay_buffer_size // batch_size
        hidden_layer_sizes = [64, 64]
        hidden_layer_activation = "relu"
        gradient_max_norm = 2
        cer = True
        dueling = True
        early_stopping_reward = 200

        # Agent
        agent = DQN(
            environment=self.environment,
            discount=discount,
            learning_rate=learning_rate,
            replay_buffer_size=replay_buffer_size,
            replay_buffer_min=replay_buffer_min,
            batch_size=batch_size,
            target_update_period=target_update_period,
            epsilon=epsilon,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
            epsilon_decay_duration=epsilon_decay_episodes,
            gradient_max_norm=gradient_max_norm,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_activation=hidden_layer_activation,
            cer=cer,
            dueling=dueling,
            device=get_device(),
            seed=self.seed
        )
        agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)

        def compute_action(state):
            action = agent.compute_action(state, explore=False)
            return action

        self._test(compute_action)

    def test_saved_model(self):
        model_path = Path(__file__).parent.parent / "results" / "lab2" / "problem1" / "neural-network-1.pth"
        model = torch.load(model_path)

        def compute_action(state):
            state = torch.as_tensor(
                data=state.reshape((1,) + state.shape),
                dtype=torch.float32
            )
            q_values = model(state)
            action = q_values.argmax().item()
            return action

        self._test(compute_action)

    def _test(self, compute_action):
        n_episodes = 50
        confidence_pass = 50

        episode_rewards = []
        episodes = trange(n_episodes, desc='Episode: ', leave=True)
        for episode in episodes:
            episodes.set_description(f"Episode {episode}")
            done = False
            state = self.environment.reset()
            episode_reward = 0.
            while not done:
                action = compute_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            self.environment.close()

        # Assumption: episode reward has Gaussian distribution
        # Goal: estimate the mean value by taking the sample mean
        # Problem: how close the sample mean is from the true mean value?
        #
        # Confidence level: 0.95
        # Confidence interval: (sample_mean - confidence, sample_mean + confidence)
        # Confidence: confidence = q_0.975 * std_reward / sqrt(n)
        #
        # See "Philosophy of Science and Research Methodology" course
        avg_reward = np.mean(episode_rewards)
        confidence = np.std(episode_rewards) * 1.96 / np.sqrt(n_episodes)
        self.assertTrue(avg_reward - confidence >= confidence_pass)


if __name__ == '__main__':
    unittest.main()
