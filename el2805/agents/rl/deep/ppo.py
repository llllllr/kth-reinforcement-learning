import gym
from el2805.agents.rl.common.rl_agent import RLAgent
from el2805.agents.rl.deep.common.fc_network import FCNetwork
from el2805.agents.rl.deep.common.fc_network import Experience


class PPO(RLAgent):
    """PPO (Proximal Policy Optimization) agent."""

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
            gradient_clipping_max_norm: float,
            n_hidden_layers: int,
            hidden_layer_size: int,
            activation: str,
            cer: bool,
            dueling: bool,
            device: str,
            seed: int | None = None
    ):
        super().__init__(

        )

        self.critic = FCNetwork(
            input_size=n_state_features,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            output_size=1,
            include_top=True
        )

        self.actor = FCNetwork(
            input_size=n_state_features,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            output_size=n_actions,
            include_top=True
        )

    def update(self) -> dict:
        pass

    def record_experience(self, experience: Experience) -> None:
        pass

    def compute_action(self, **kwargs) -> int:
        pass
