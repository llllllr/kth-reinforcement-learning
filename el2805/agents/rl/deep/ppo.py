import gym
import numpy as np
import torch
from collections import defaultdict
from el2805.agents.rl.common.rl_agent import RLAgent
from el2805.agents.rl.common.experience import Experience
from el2805.agents.rl.deep.common.fc_network import FCNetwork


class PPO(RLAgent):
    """PPO (Proximal Policy Optimization) agent."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            batch_size: int,
            n_epochs_per_update: int,
            critic_learning_rate: float,
            critic_n_hidden_layers: int,
            critic_hidden_layer_size: int,
            critic_activation: str,
            actor_learning_rate: float,
            actor_n_hidden_layers: int,
            actor_hidden_layer_size: int,
            actor_activation: str,
            objective_clipping_eps: float,
            gradient_clipping_max_norm: float,
            device: str,
            seed: int | None = None
    ):
        super().__init__(environment=environment, seed=seed)
        self.discount = discount
        self.batch_size = batch_size
        self.n_epochs_per_update = n_epochs_per_update
        self.critic_learning_rate = critic_learning_rate
        self.critic_n_hidden_layers = critic_n_hidden_layers
        self.critic_hidden_layer_size = critic_hidden_layer_size
        self.critic_activation = critic_activation
        self.actor_learning_rate = actor_learning_rate
        self.actor_n_hidden_layers = actor_n_hidden_layers
        self.actor_hidden_layer_size = actor_hidden_layer_size
        self.actor_activation = actor_activation
        self.objective_clipping_eps = objective_clipping_eps
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.device = device

        assert isinstance(environment.observation_space, gym.spaces.Box)
        n_state_features = len(environment.observation_space.low)
        if not isinstance(environment.action_space, gym.spaces.Box):
            raise NotImplementedError
        action_dim = len(environment.action_space.low)

        self.critic = FCNetwork(
            input_size=n_state_features,
            n_hidden_layers=self.critic_n_hidden_layers,
            hidden_layer_size=self.critic_hidden_layer_size,
            activation=self.critic_activation,
            output_size=1,
            include_top=True
        ).to(self.device)

        self.actor = FCNetwork(
            input_size=n_state_features,
            n_hidden_layers=self.actor_n_hidden_layers,
            hidden_layer_size=self.actor_hidden_layer_size,
            activation=self.actor_activation,
            output_size=action_dim,
            include_top=True
        ).to(self.device)

        self._critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self._actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self._episodic_buffer = []

    def update(self) -> dict:
        stats = defaultdict(list)

        # Skip update if the episode has not terminated
        if not self._episodic_buffer[-1].done:
            return stats

        # Unpack experiences
        rewards = np.as_array([e.reward for e in self._episodic_buffer])
        states = torch.as_tensor(
            data=np.asarray([e.state for e in self._episodic_buffer]),
            dtype=torch.float32,
            device=self.device
        )

        # Compute Monte Carlo targets
        g = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + self.discount * discounted_reward
            g.append(discounted_reward)
        g = g[::-1]     # reverse
        g = torch.as_tensor(np.asarray(g), device=self.device)

        # Compute actions from old policy
        pi_old = self.actor(states).detach()

        for _ in range(self.n_epochs_per_update):
            # Forward pass by critic
            v = self.critic(states)
            critic_loss = torch.nn.functional.mse_loss(g, v)

            # Backward pass by critic
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_clipping_max_norm)
            self._critic_optimizer.step()

            # Forward pass by actor
            # TODO: check dimension, I need to get two-dimensional loss
            pi = self.actor(states)
            psi = g - v
            r = pi / pi_old
            r_clipped = max(1 - self.objective_clipping_eps, min(r, 1 + self.objective_clipping_eps))
            actor_loss = - torch.mean(torch.min(r * psi, r_clipped * psi))  # TODO: this is for sure wrong for dimensions

            # Backward pass by actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_clipping_max_norm)
            self._actor_optimizer.step()

            # Save stats
            stats["critic_loss"].append(critic_loss.item())
            stats["actor_loss"].append(actor_loss.item())

        # Clear episodic buffer for new episode
        self._episodic_buffer = []

        return stats

    def record_experience(self, experience: Experience) -> None:
        self._episodic_buffer.append(experience)

    def compute_action(
            self,
            state: np.ndarray,
            *,
            explore: bool = True,
            **kwargs
    ) -> int:
        pass
