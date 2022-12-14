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

import numpy as np
import gym
import torch
from collections import deque
from el2805.agents.rl import RLAgent
from el2805.utils import Experience, NeuralNetwork
from utils import get_device


class Agent(RLAgent):
    _replay_buffer_size = 128
    _batch_size = 3

    def __init__(self, environment, neural_network, device):
        super().__init__(environment=environment, discount=1, learning_rate=1e-3)

        self.device = device
        self.neural_network = neural_network.to(device)

        self._n_actions = environment.action_space.n
        self._replay_buffer = deque(maxlen=self._replay_buffer_size)
        self._optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)

        assert self.neural_network.input_size == len(environment.observation_space.low)
        assert self.neural_network.output_size == self._n_actions

    def update(self):
        if len(self._replay_buffer) < self._batch_size:
            print("Not enough experience, skipping update...")
            return

        # clean up gradients
        self._optimizer.zero_grad()

        # sample mini-batch of experiences
        experience_indices = self._rng.choice(len(self._replay_buffer), size=self._batch_size)
        experience_batch = [self._replay_buffer[i] for i in experience_indices]

        # forward pass
        states = torch.as_tensor(np.asarray([e.state for e in experience_batch]), device=self.device)
        actions = torch.as_tensor([e.action for e in experience_batch], device=self.device)
        outputs = self.neural_network(states)
        assert outputs.shape == (self._batch_size, self._n_actions)
        outputs = outputs[torch.arange(self._batch_size), actions]
        assert outputs.shape == (self._batch_size,)
        z = outputs
        y = torch.zeros(len(z))
        loss = torch.nn.functional.mse_loss(z, y)

        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), max_norm=1)
        self._optimizer.step()

    def compute_action(self, state):
        state = torch.tensor(state.reshape((1,) + state.shape))
        output = self.neural_network(state)
        assert output.shape[0] == 1
        action = output.argmax().item()
        return action

    def record_experience(self, experience):
        self._replay_buffer.append(experience)


def main():
    environment = gym.make('CartPole-v0')           # Create a CartPole environment

    n_state_features = len(environment.observation_space.low)
    n_actions = environment.action_space.n
    neural_network = NeuralNetwork(
        input_size=n_state_features,
        output_size=n_actions,
        hidden_layer_size=8,
        n_hidden_layers=1
    )
    agent = Agent(environment=environment, neural_network=neural_network, device=get_device())

    for episode in range(5):
        state = environment.reset()                 # Reset environment, returns initial state
        done = False                                # Boolean variable used to indicate if an episode terminated

        while not done:
            environment.render()                    # Render the environment (DO NOT USE during training of the labs...)

            # action = np.random.randint(m)   # Pick a random integer between [0, m-1]
            action = agent.compute_action(state)

            # The next line takes permits you to take an action in the RL environment
            # env.step(action) returns 4 variables:
            # (1) next state; (2) reward; (3) done variable; (4) additional stuff
            next_state, reward, done, _ = environment.step(action)

            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            agent.record_experience(experience)
            agent.update()

            state = next_state

    environment.close()  # Close all the windows


if __name__ == "__main__":
    main()
