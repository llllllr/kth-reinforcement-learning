import gym
import numpy as np
import torch
from collections import deque
from copy import deepcopy
from el2805.agents.rl.rl_agent import RLAgent
from el2805.utils import Experience, NeuralNetwork, random_decide


class DQN(RLAgent):
    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            epsilon: float | str,
            epsilon_max: float | None = None,
            epsilon_min: float | None = None,
            epsilon_decay_duration: int | None = None,
            q_network: NeuralNetwork,
            learning_rate: float,
            batch_size: int,
            replay_buffer_size: int,
            warmup_steps: int,
            target_update_frequency: int,
            gradient_clipping_value: float,
            cer: bool,
            dueling: bool,
            device: str
    ):
        super().__init__(environment=environment, discount=discount, learning_rate=learning_rate)

        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_duration = epsilon_decay_duration
        self.q_network = q_network
        self.replay_buffer_size = replay_buffer_size
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.gradient_clipping_value = gradient_clipping_value
        self.cer = cer
        self.dueling = dueling
        self.device = device

        self._target_q_network = deepcopy(self.q_network)
        self._n_actions = environment.action_space.n
        self._replay_buffer = deque(maxlen=replay_buffer_size)
        self._optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self._n_updates = 0

        self.q_network = self.q_network.to(self.device)
        self._target_q_network = self._target_q_network.to(self.device)

        assert self.q_network.input_size == len(environment.observation_space.low)
        assert self.q_network.output_size == self._n_actions

        if self.epsilon != "decay" and not isinstance(self.epsilon, float):
            raise NotImplementedError
        if self.epsilon == "decay":
            assert self.epsilon_max is not None and \
                   self.epsilon_min is not None and \
                   self.epsilon_decay_duration is not None

        if self.dueling:
            raise NotImplementedError

    def update(self, **kwargs) -> None:
        _ = kwargs

        if len(self._replay_buffer) < self.warmup_steps:
            print("Not enough experience, skipping update...")
            return

        # Enable training mode
        self.q_network.train(mode=True)

        # Clean up gradients
        self._optimizer.zero_grad()

        # Sample mini-batch of experiences
        experience_indices = self._rng.choice(len(self._replay_buffer), size=self.batch_size)
        experience_batch = [self._replay_buffer[i] for i in experience_indices]
        if self.cer:
            experience_batch[-1] = self._replay_buffer[-1]

        # Unpack experiences
        states = torch.as_tensor(
            data=np.asarray([e.state for e in experience_batch]),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.as_tensor(
            data=[e.action for e in experience_batch],
            dtype=torch.long,
            device=self.device
        )
        next_states = torch.as_tensor(
            data=np.asarray([e.next_state for e in experience_batch]),
            dtype=torch.float32,
            device=self.device
        )
        rewards = torch.as_tensor(
            data=[e.reward for e in experience_batch],
            dtype=torch.float32,
            device=self.device
        )
        dones = torch.as_tensor(
            data=[e.done for e in experience_batch],
            dtype=torch.bool,
            device=self.device
        )

        # Compute targets
        q_next = self._target_q_network(next_states)        # Q(s',a)
        assert q_next.shape == (self.batch_size, self._n_actions)
        targets = rewards + dones * self.discount * q_next.max(axis=1).values
        assert targets.shape == (self.batch_size,)

        # Forward pass
        q = self._target_q_network(states)                  # Q(s,a)
        assert q.shape == (self.batch_size, self._n_actions)
        q = q[torch.arange(self.batch_size), actions]       # Q(s,a*), where a* is the action taken in the experience
        assert q.shape == (self.batch_size,)
        loss = torch.nn.functional.mse_loss(targets, q)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.gradient_clipping_value)
        self._optimizer.step()

        # Update target network
        self._n_updates += 1
        if (self._n_updates % self.target_update_frequency) == 0:
            self._target_q_network = deepcopy(self.q_network)

        # Disable training mode
        self.q_network.train(mode=False)

    def compute_action(self, *, state: np.ndarray, episode: int, explore: bool = True, **kwargs) -> int:
        _ = kwargs

        # Calculate epsilon
        if explore and self.epsilon == "decay":     # if explore=False, we don't care about epsilon
            epsilon = max(
                self.epsilon_min,
                self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (episode-1) / (self.epsilon_decay_duration - 1)
            )
        else:
            epsilon = self.epsilon

        # Epsilon-greedy policy (or greedy policy if explore=False)
        if explore and random_decide(self._rng, epsilon):   # exploration (probability eps)
            action = self._rng.choice(self._n_actions)
        else:                                               # exploitation (probability 1-eps)
            state = torch.as_tensor(
                data=state.reshape((1,) + state.shape),
                dtype=torch.float32,
                device=self.device
            )
            q_values = self.q_network(state)
            assert q_values.shape[0] == 1
            action = q_values.argmax().item()

        return action

    def record_experience(self, experience: Experience) -> None:
        self._replay_buffer.append(experience)
