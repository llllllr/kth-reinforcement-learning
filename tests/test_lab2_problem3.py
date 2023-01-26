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
from pathlib import Path
from el2805.agents.rl import PPO
from el2805.agents.rl.utils import get_device
from tests.utils import test


class PPOTestCase(unittest.TestCase):
    seed = 1
    environment = None
    rng = None

    def setUp(self):
        self.environment = gym.make('LunarLanderContinuous-v2')
        self.environment.seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

    def tearDown(self):
        self.environment.close()

    def test_training(self):
        # Hyper-parameters
        n_episodes = 1600
        discount = .99
        n_epochs_per_update = 10
        epsilon = .2
        critic_learning_rate = 1e-3
        critic_hidden_layer_sizes = [400, 200]
        critic_hidden_layer_activation = "relu"
        actor_learning_rate = 1e-5
        actor_shared_hidden_layer_sizes = [400]
        actor_mean_hidden_layer_sizes = [200]
        actor_var_hidden_layer_sizes = [200]
        actor_hidden_layer_activation = "relu"
        gradient_max_norm = 1
        early_stopping_reward = 250

        # Agent
        agent = PPO(
            environment=self.environment,
            discount=discount,
            n_epochs_per_step=n_epochs_per_update,
            critic_learning_rate=critic_learning_rate,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_hidden_layer_activation=critic_hidden_layer_activation,
            actor_learning_rate=actor_learning_rate,
            actor_shared_hidden_layer_sizes=actor_shared_hidden_layer_sizes,
            actor_mean_hidden_layer_sizes=actor_mean_hidden_layer_sizes,
            actor_var_hidden_layer_sizes=actor_var_hidden_layer_sizes,
            actor_hidden_layer_activation=actor_hidden_layer_activation,
            epsilon=epsilon,
            gradient_max_norm=gradient_max_norm,
            device=get_device(),
            seed=self.seed
        )
        agent.train(n_episodes=n_episodes, early_stop_reward=early_stopping_reward)

        def compute_action(state):
            action = agent.compute_action(state, explore=True)
            return action

        test(self, self.environment, compute_action)

    def test_saved_model(self):
        model_path = Path(__file__).parent.parent / "results" / "lab2" / "problem3" / "neural-network-3-actor.pth"
        model = torch.load(model_path)

        def compute_action(state):
            with torch.no_grad():
                state = torch.as_tensor(
                    data=state.reshape((1,) + state.shape),
                    dtype=torch.float64
                )
                mean, var = model(state)
                mean, var = mean.reshape(-1), var.reshape(-1)
                action = torch.normal(mean, torch.sqrt(var))
                action = action.numpy()
            return action

        test(self, self.environment, compute_action)


if __name__ == '__main__':
    unittest.main()
