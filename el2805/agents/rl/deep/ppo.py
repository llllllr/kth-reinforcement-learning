import gym
import numpy as np
import torch
from collections import defaultdict
from el2805.agents.rl.common.rl_agent import RLAgent
from el2805.agents.rl.common.experience import Experience
from el2805.agents.rl.deep.common.multi_layer_perceptron import MultiLayerPerceptron
from el2805.agents.rl.deep.common.utils import normal_pdf


class PPO(RLAgent):
    """PPO (Proximal Policy Optimization) agent."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            n_epochs_per_update: int,
            critic_learning_rate: float,
            critic_hidden_layer_sizes: list[int],
            critic_hidden_layer_activation: str,
            actor_learning_rate: float,
            actor_shared_hidden_layer_sizes: list[int],
            actor_mean_hidden_layer_sizes: list[int],
            actor_var_hidden_layer_sizes: list[int],
            actor_hidden_layer_activation: str,
            objective_clipping_eps: float,
            gradient_max_norm: float,
            device: str,
            seed: int | None = None
    ):
        super().__init__(environment=environment, seed=seed)
        self.discount = discount
        self.n_epochs_per_update = n_epochs_per_update
        self.critic_learning_rate = critic_learning_rate
        self.critic_hidden_layer_sizes = critic_hidden_layer_sizes
        self.critic_hidden_layer_activation = critic_hidden_layer_activation
        self.actor_learning_rate = actor_learning_rate
        self.actor_shared_hidden_layer_sizes = actor_shared_hidden_layer_sizes
        self.actor_mean_hidden_layer_sizes = actor_mean_hidden_layer_sizes
        self.actor_var_hidden_layer_sizes = actor_var_hidden_layer_sizes
        self.actor_hidden_layer_activation = actor_hidden_layer_activation
        self.objective_clipping_eps = objective_clipping_eps
        self.gradient_max_norm = gradient_max_norm
        self.device = device

        assert isinstance(environment.observation_space, gym.spaces.Box)
        state_dim = len(environment.observation_space.low)
        assert isinstance(environment.action_space, gym.spaces.Box)
        self._action_dim = len(environment.action_space.low)

        self.critic = PPOCritic(
            state_dim=state_dim,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
            hidden_layer_activation=self.critic_hidden_layer_activation
        ).to(self.device)

        self.actor = PPOActor(
            state_dim=state_dim,
            action_dim=self._action_dim,
            shared_hidden_layer_sizes=self.actor_shared_hidden_layer_sizes,
            mean_hidden_layer_sizes=self.actor_mean_hidden_layer_sizes,
            var_hidden_layer_sizes=self.actor_var_hidden_layer_sizes,
            hidden_layer_activation=self.actor_hidden_layer_activation
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
        rewards = np.asarray([e.reward for e in self._episodic_buffer])
        states = torch.as_tensor(
            data=np.asarray([e.state for e in self._episodic_buffer]),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.as_tensor(
            data=[e.action for e in self._episodic_buffer],
            dtype=torch.float32,
            device=self.device
        )

        # Compute Monte Carlo targets
        g = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + self.discount * discounted_reward
            g.append(discounted_reward)
        g = torch.as_tensor(
            np.asarray(g[::-1]),    # reverse
            dtype=torch.float32,
            device=self.device
        )

        # Compute actions from old policy
        mean, var = self.actor(states)
        pi_old = normal_pdf(actions, mean, var).prod(dim=1)     # assumption: independent action dimensions
        pi_old = pi_old.detach()

        for _ in range(self.n_epochs_per_update):
            # Forward pass by critic
            v = self.critic(states)
            v = v.reshape(-1)
            critic_loss = torch.nn.functional.mse_loss(g, v)

            # Backward pass by critic
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            # TODO: if it works without gradient clipping, remove it from constructor
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_clipping_max_norm)
            self._critic_optimizer.step()

            # Forward pass by actor
            v = v.detach()      # fix critic, we don't want to update the critic with the backward pass on the actor
            psi = g - v
            mean, var = self.actor(states)
            pi = normal_pdf(actions, mean, var).prod(dim=1)     # assumption: independent action dimensions
            r = pi / pi_old
            r_clipped = r.clip(min=1-self.objective_clipping_eps, max=1+self.objective_clipping_eps)
            actor_loss = - torch.mean(torch.minimum(r * psi, r_clipped * psi))

            # Backward pass by actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_clipping_max_norm)
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
    ) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(
                data=state.reshape((1,) + state.shape),
                dtype=torch.float32,
                device=self.device
            )
            mean, var = self.actor(state)
            assert mean.shape == (1, self._action_dim) and var.shape == (1, self._action_dim)
            mean, var = mean.reshape(-1), var.reshape(-1)
            action = self._rng.normal(mean, var) if explore else mean
            action = action.clip(min=self.environment.action_space.low, max=self.environment.action_space.high)

        return action


class PPOCritic(MultiLayerPerceptron):
    def __init__(
            self,
            *,
            state_dim: int,
            hidden_layer_sizes: list[int],
            hidden_layer_activation: str,
    ):
        super().__init__(
            input_size=state_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_activation=hidden_layer_activation,
            output_size=1,
            include_top=True
        )


class PPOActor(torch.nn.Module):
    def __init__(
            self,
            *,
            state_dim: int,
            action_dim: int,
            shared_hidden_layer_sizes: list[int],
            mean_hidden_layer_sizes: list[int],
            var_hidden_layer_sizes: list[int],
            hidden_layer_activation: str
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_hidden_layer_sizes = shared_hidden_layer_sizes
        self.mean_hidden_layer_sizes = mean_hidden_layer_sizes
        self.var_hidden_layer_sizes = var_hidden_layer_sizes
        self.hidden_layer_activation = hidden_layer_activation

        self._shared_hidden_layers = MultiLayerPerceptron(
            input_size=self.state_dim,
            hidden_layer_sizes=self.shared_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            include_top=False
        )
        input_size = self.shared_hidden_layer_sizes[-1]

        self._mean_hidden_layers = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.mean_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,
            output_layer_activation="tanh",
            include_top=True
        )

        self._var_hidden_layers = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.var_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,    # assumption: independent action dimensions
            output_layer_activation="sigmoid",
            include_top=True
        )

    def forward(self, x):
        x = self._shared_hidden_layers(x)
        mean = self._mean_hidden_layers(x)
        var = self._var_hidden_layers(x)
        return mean, var

