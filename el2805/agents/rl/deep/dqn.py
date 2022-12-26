import gym
import numpy as np
import torch
from collections import deque
from copy import deepcopy
from el2805.agents.rl.rl_agent import RLAgent
from el2805.agents.rl.utils import Experience, get_epsilon
from el2805.agents.rl.deep.utils import MultiLayerPerceptron
from el2805.utils import random_decide


class DQN(RLAgent):
    """DQN (Deep Q-Network) agent."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            epsilon: float | str,
            epsilon_max: float | None = None,
            epsilon_min: float | None = None,
            epsilon_decay_duration: int | None = None,
            delta: float | None = None,
            learning_rate: float,
            batch_size: int,
            replay_buffer_size: int,
            replay_buffer_min: int,
            target_update_period: int,
            gradient_max_norm: float,
            hidden_layer_sizes: list[int],
            hidden_layer_activation: str,
            cer: bool,
            dueling: bool,
            device: str,
            seed: int | None = None
    ):
        """Initializes a DQN agent.
        
        :param environment: RL environment
        :type environment: gym.Env
        :param discount: discount factor of the MDP
        :type discount: float
        :param epsilon: probability of exploration (eps-greedy policy) or strategy to schedule it
        :type epsilon: float or str
        :param epsilon_max: initial probability of exploration (eps-greedy policy)
        :type epsilon_max: float, optional
        :param epsilon_min: final probability of exploration (eps-greedy policy)
        :type epsilon_min: float, optional
        :param epsilon_decay_duration: duration of epsilon decay in episodes (eps-greedy policy)
        :type epsilon_decay_duration: int, optional
        :param delta: exponent in epsilon decay 1/(episode**delta) (eps-greedy policy)
        :type delta: float, optional
        :param learning_rate: learning rate (e.g., 1e-3) or learning rate method (e.g., "decay")
        :type learning_rate: float or str
        :param batch_size: batch size
        :type batch_size: int
        :param replay_buffer_size: size of experience replay buffer 
        :type replay_buffer_size: int
        :param replay_buffer_min: minimum number of experiences in the experience replay buffer to update the Q-network
        :type replay_buffer_min: int
        :param target_update_period: period for refreshing target network, expressed in number of steps
        :type target_update_period: int
        :param gradient_max_norm: maximum norm used for gradient clipping
        :type gradient_max_norm: float
        :param hidden_layer_sizes: number of neurons in each hidden layer of the Q-network
        :type hidden_layer_sizes: list[int]
        :param hidden_layer_activation: activation function for hidden layers in the Q-network
        :type hidden_layer_activation: str
        :param cer: enables CER (combined experience replay)
        :type cer: bool
        :param dueling: enables dueling DQN
        :type dueling: bool
        :param device: device where to store and run neural networks (e.g., "cpu")
        :type device: str
        :param seed: seed
        :type seed: int, optional
        """
        super().__init__(environment=environment, seed=seed)
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_duration
        self.delta = delta
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_min = replay_buffer_min
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.gradient_max_norm = gradient_max_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activation = hidden_layer_activation
        self.cer = cer
        self.dueling = dueling
        self.device = device

        assert isinstance(environment.observation_space, gym.spaces.Box)
        state_dim = len(environment.observation_space.low)
        assert isinstance(environment.action_space, gym.spaces.Discrete)
        self._n_actions = environment.action_space.n

        self.q_network = QNetwork(
            state_dim=state_dim,
            n_actions=self._n_actions,
            hidden_layer_sizes=self.hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            dueling=self.dueling
        ).double().to(self.device)

        self._target_q_network = deepcopy(self.q_network).to(self.device)
        self._replay_buffer = deque(maxlen=self.replay_buffer_size)
        self._optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self._n_updates = 0

    def update(self) -> dict:
        stats = {}

        # Skip update if buffer does not contain enough experiences
        if len(self._replay_buffer) < self.replay_buffer_min:
            return stats

        # Enable training mode
        self.q_network.train()

        # Sample mini-batch of experiences
        experience_indices = self._rng.choice(len(self._replay_buffer), size=self.batch_size)
        experience_batch = [self._replay_buffer[i] for i in experience_indices]
        if self.cer:
            experience_batch[-1] = self._replay_buffer[-1]

        # Unpack experiences
        states = torch.as_tensor(
            data=np.asarray([e.state for e in experience_batch]),
            dtype=torch.float64,
            device=self.device
        )
        actions = torch.as_tensor(
            data=np.asarray([e.action for e in experience_batch]),
            dtype=torch.long,
            device=self.device
        )
        next_states = torch.as_tensor(
            data=np.asarray([e.next_state for e in experience_batch]),
            dtype=torch.float64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            data=[e.reward for e in experience_batch],
            dtype=torch.float64,
            device=self.device
        )
        dones = torch.as_tensor(
            data=[e.done for e in experience_batch],
            dtype=torch.bool,
            device=self.device
        )

        # Compute targets
        with torch.no_grad():
            q_next = self._target_q_network(next_states)    # Q(s',a)
            targets = rewards + dones.logical_not() * self.discount * q_next.max(axis=1).values

        # Forward pass
        q = self.q_network(states)                          # Q(s,a)
        q = q[torch.arange(self.batch_size), actions]       # Q(s,a*), where a* is the action taken in the experience
        loss = torch.nn.functional.mse_loss(targets, q)

        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.gradient_max_norm)
        self._optimizer.step()

        # Update target network
        self._n_updates = (self._n_updates + 1) % self.target_update_period
        if self._n_updates == 0:
            self._target_q_network = deepcopy(self.q_network)

        # Disable training mode
        self.q_network.eval()

        # Save stats
        stats["loss"] = loss.item()
        return stats

    def record_experience(self, experience: Experience) -> None:
        self._replay_buffer.append(experience)

    def compute_action(
            self,
            state: np.ndarray,
            *,
            episode: int | None = None,
            explore: bool = True,
            **kwargs
    ) -> int:
        _ = kwargs
        assert not (explore and episode is None)

        # Compute epsilon according to exploration strategy
        if explore:
            epsilon = get_epsilon(
                epsilon=self.epsilon,
                episode=episode,
                epsilon_max=self.epsilon_max,
                epsilon_min=self.epsilon_min,
                epsilon_decay_duration=self.epsilon_decay_episodes,
                delta=self.delta
            )
        else:
            epsilon = None

        # Epsilon-greedy policy (or greedy policy if explore=False)
        if explore and random_decide(self._rng, epsilon):   # exploration (probability eps)
            action = self._rng.choice(self._n_actions)
        else:                                               # exploitation (probability 1-eps)
            with torch.no_grad():
                state = torch.as_tensor(
                    data=state.reshape((1,) + state.shape),
                    dtype=torch.float64,
                    device=self.device
                )
                q = self.q_network(state)
                assert q.shape == (1, self._n_actions)
                action = q.argmax().item()

        return action


class QNetwork(torch.nn.Module):
    def __init__(
            self,
            *,
            state_dim: int,
            n_actions: int,
            hidden_layer_sizes: list[int],
            hidden_layer_activation: str,
            dueling: bool
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activation = hidden_layer_activation
        self.dueling = dueling

        self._hidden_layers = MultiLayerPerceptron(
            input_size=self.state_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            include_top=False
        )

        input_size = self.hidden_layer_sizes[-1]
        if self.dueling:
            self._v_layer = torch.nn.Linear(input_size, 1)
            self._advantage_layer = torch.nn.Linear(input_size, self.n_actions)
            self._output_layer = None
        else:
            self._v_layer = None
            self._advantage_layer = None
            self._output_layer = torch.nn.Linear(input_size, self.n_actions)

    def forward(self, x):
        x = x.to(torch.float64)
        x = self._hidden_layers(x)
        if self.dueling:
            v = self._v_layer(x)
            advantage = self._advantage_layer(x)
            avg_advantage = advantage.mean(dim=1, keepdim=True)
            q = v + advantage - avg_advantage
        else:
            q = self._output_layer(x)
        return q
