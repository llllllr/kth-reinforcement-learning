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
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se
#
# Modified by: [Franco Ruggeri - fruggeri@kth.se]

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from el2805.agents.rl import RLAgent
from el2805.agents.rl.fn_approximation.dqn import QNetwork
from utils import get_device


class Agent(RLAgent):
    _replay_buffer_size = 128
    _batch_size = 3

    def __init__(self, *, environment, device):
        super().__init__(environment=environment, discount=1, learning_rate=1e-3)

        n_state_features = len(environment.observation_space.low)
        self._n_actions = environment.action_space.n

        self.device = device
        self.neural_network = QNetwork(
            input_size=n_state_features,
            output_size=self._n_actions,
            n_hidden_layers=1,
            hidden_layer_size=8,
            activation="relu"
        ).to(device)

        self._replay_buffer = deque(maxlen=self._replay_buffer_size)
        self._optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)

    def update(self) -> dict:
        stats = {}
        if len(self._replay_buffer) < self._batch_size:
            return stats

        # Clean up gradients
        self._optimizer.zero_grad()

        # Sample mini-batch of experiences
        experience_indices = self._rng.choice(len(self._replay_buffer), size=self._batch_size)
        experience_batch = [self._replay_buffer[i] for i in experience_indices]

        # Forward pass
        states = torch.as_tensor(np.asarray([e.state for e in experience_batch]), device=self.device)
        actions = torch.as_tensor([e.action for e in experience_batch], device=self.device)
        outputs = self.neural_network(states)
        assert outputs.shape == (self._batch_size, self._n_actions)
        outputs = outputs[torch.arange(self._batch_size), actions]
        assert outputs.shape == (self._batch_size,)
        z = outputs
        y = torch.zeros(len(z))
        loss = torch.nn.functional.mse_loss(z, y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), max_norm=1)
        self._optimizer.step()

        # Save stats
        stats["loss"] = loss.item()
        return stats

    def record_experience(self, experience):
        self._replay_buffer.append(experience)

    def compute_action(self, state, **kwargs):
        _ = kwargs
        state = torch.tensor(state.reshape((1,) + state.shape))
        output = self.neural_network(state)
        assert output.shape[0] == 1
        action = output.argmax().item()
        return action


def main():
    environment = gym.make("CartPole-v0")
    agent = Agent(
        environment=environment,
        device=get_device()
    )
    history_stats = agent.train(n_episodes=100)
    environment.close()

    figure, axes = plt.subplots()
    axes.plot(np.arange(1, len(history_stats["loss"])+1), history_stats["loss"])
    axes.set_xlabel("time step")
    axes.set_ylabel("loss")
    figure.show()


if __name__ == "__main__":
    main()
