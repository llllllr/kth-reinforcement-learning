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
from el2805.agents.rl.deep import PPO
from el2805.agents.rl.deep.common.utils import get_device


# TODO: fix and avoid duplicated code
class PPOTestCase(unittest.TestCase):
    seed = 1
    environment = None

    def setUp(self):
        self.environment = gym.make('LunarLanderContinuous-v2')
        self.environment.seed(self.seed)

    def tearDown(self):
        self.environment.close()

    def test_training(self):
        # Hyper-parameters
        seed = 1
        n_episodes = 1000
        discount = .99
        n_epochs_per_update = 1
        critic_learning_rate = 1e-3
        critic_n_hidden_layers = 2
        critic_hidden_layer_size = 64
        critic_activation = "relu"
        actor_learning_rate = 1e-3
        actor_n_hidden_layers = 2
        actor_hidden_layer_size = 64
        actor_activation = "relu"
        objective_clipping_eps = .1
        gradient_max_norm = 2
        early_stopping_reward = 200

        # Agent
        agent = PPO(
            environment=self.environment,
            discount=discount,
            n_epochs_per_update=n_epochs_per_update,
            critic_learning_rate=critic_learning_rate,
            critic_n_hidden_layers=critic_n_hidden_layers,
            critic_hidden_layer_sizes=critic_hidden_layer_size,
            critic_hidden_layer_activation=critic_activation,
            actor_learning_rate=actor_learning_rate,
            actor_n_hidden_layers=actor_n_hidden_layers,
            actor_hidden_layer_sizes=actor_hidden_layer_size,
            actor_hidden_layer_activation=actor_activation,
            objective_clipping_eps=objective_clipping_eps,
            gradient_max_norm=gradient_max_norm,
            device=get_device(),
            seed=seed
        )
        agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)

        def compute_action(state):
            action = agent.compute_action(state, explore=False)
            return action

        self._test(compute_action)

    def test_saved_model(self):
        model_path = Path(__file__).parent.parent / "results" / "lab2" / "problem3" / "neural-network-3-actor.pth"
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
        confidence_pass = 125

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
