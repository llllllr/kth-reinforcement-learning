import gym
import numpy as np
import torch
from collections import defaultdict
from el2805.agents.rl.rl_agent import RLAgent
from el2805.agents.rl.utils import Experience
from el2805.agents.rl.deep.utils import MultiLayerPerceptron, normal_pdf


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
            policy_ratio_clip_range: float,
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
        self.policy_ratio_clip_range = policy_ratio_clip_range
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
        ).double().to(self.device)

        self.actor = PPOActor(
            state_dim=state_dim,
            action_dim=self._action_dim,
            shared_hidden_layer_sizes=self.actor_shared_hidden_layer_sizes,
            mean_hidden_layer_sizes=self.actor_mean_hidden_layer_sizes,
            var_hidden_layer_sizes=self.actor_var_hidden_layer_sizes,
            hidden_layer_activation=self.actor_hidden_layer_activation
        ).double().to(self.device)

        self._critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self._actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self._episodic_buffer = []

    def update(self) -> dict:
        stats = defaultdict(list)

        # Skip update if the episode has not terminated
        if not self._episodic_buffer[-1].done:
            return stats

        # Unpack experiences
        n = len(self._episodic_buffer)
        rewards = torch.as_tensor(
            data=np.asarray([e.reward for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )
        states = torch.as_tensor(
            data=np.asarray([e.state for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )
        actions = torch.as_tensor(
            data=np.asarray([e.action for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )

        # Compute Monte Carlo targets
        g = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.discount * discounted_reward
            g.append(discounted_reward)
        g = torch.as_tensor(
            np.asarray(g[::-1]),    # reverse
            dtype=torch.float64,
            device=self.device
        )
        assert g.shape == (n,)

        with torch.no_grad():
            # Compute advantages
            v = self.critic(states)
            assert v.shape == (n, 1)
            v = v.reshape(-1)
            psi = g - v

            # Compute action probabilities from old policy
            mean, var = self.actor(states)
            assert mean.shape == (n, self._action_dim) and var.shape == (n, self._action_dim)
            pi_old = normal_pdf(actions, mean, var).prod(dim=1)     # assumption: independent action dimensions
            assert pi_old.shape == (n,)

        for _ in range(self.n_epochs_per_update):
            # Forward pass by critic
            v = self.critic(states)
            assert v.shape == (n, 1)
            v = v.reshape(-1)
            critic_loss = torch.nn.functional.mse_loss(g, v)

            # Backward pass by critic
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_max_norm)
            self._critic_optimizer.step()

            # Forward pass by actor
            mean, var = self.actor(states)
            assert mean.shape == (n, self._action_dim) and var.shape == (n, self._action_dim)
            pi = normal_pdf(actions, mean, var).prod(dim=1)     # assumption: independent action dimensions
            assert pi.shape == (n,)
            r = pi / pi_old
            assert r.shape == (n,)
            r_clipped = r.clip(min=1-self.policy_ratio_clip_range, max=1+self.policy_ratio_clip_range)
            assert r_clipped.shape == (n,)
            actor_loss = - torch.minimum(r * psi, r_clipped * psi).mean()

            # Backward pass by actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_max_norm)
            self._actor_optimizer.step()

            # Save stats
            stats["critic_loss"].append(critic_loss.item())
            stats["actor_loss"].append(actor_loss.item())

        # Clear episodic buffer for new episode
        self._episodic_buffer = []

        return stats

    def record_experience(self, experience: Experience) -> None:
        self._episodic_buffer.append(experience)

    def compute_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(
                data=state.reshape((1,) + state.shape),
                dtype=torch.float64,
                device=self.device
            )
            mean, var = self.actor(state)
            mean, var = mean.reshape(-1), var.reshape(-1)
            action = torch.normal(mean, torch.sqrt(var))
            action = action.numpy()
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

        self._shared_layers = MultiLayerPerceptron(
            input_size=self.state_dim,
            hidden_layer_sizes=self.shared_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            include_top=False
        )
        input_size = self.shared_hidden_layer_sizes[-1]

        # The randomized policy is modeled as a multi-variate Gaussian distribution
        # Key assumption: independent action dimensions
        # Therefore, the randomized policy is parametrized with mean and variance of each action dimension
        self._mean_head = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.mean_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,
            output_layer_activation="tanh",
            include_top=True
        )
        self._var_head = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.var_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,    # assumption: independent action dimensions
            output_layer_activation="sigmoid",
            include_top=True
        )

    def forward(self, x):
        x = self._shared_layers(x)
        mean = self._mean_head(x)
        var = self._var_head(x)
        return mean, var
