import gym
import torch
from pathlib import Path
from el2805.agents.rl.deep import PPO
from el2805.agents.rl.deep.utils import get_device
from utils import plot_training_stats, test_rl_agent


def train_ppo(results_dir, agent_path):
    # Hyper-parameters
    seed = 1
    n_episodes = 1600
    discount = .99
    n_epochs_per_update = 10
    critic_learning_rate = 1e-3
    critic_hidden_layer_sizes = [400, 200]
    critic_hidden_layer_activation = "relu"
    actor_learning_rate = 1e-5
    actor_shared_hidden_layer_sizes = [400]
    actor_mean_hidden_layer_sizes = [200]
    actor_var_hidden_layer_sizes = [200]
    actor_hidden_layer_activation = "relu"
    policy_ratio_clip_range = .9
    gradient_max_norm = 1
    early_stopping_reward = 250

    # Environment
    environment = gym.make("LunarLanderContinuous-v2")

    # Agent
    agent = PPO(
        environment=environment,
        discount=discount,
        n_epochs_per_update=n_epochs_per_update,
        critic_learning_rate=critic_learning_rate,
        critic_hidden_layer_sizes=critic_hidden_layer_sizes,
        critic_hidden_layer_activation=critic_hidden_layer_activation,
        actor_learning_rate=actor_learning_rate,
        actor_shared_hidden_layer_sizes=actor_shared_hidden_layer_sizes,
        actor_mean_hidden_layer_sizes=actor_mean_hidden_layer_sizes,
        actor_var_hidden_layer_sizes=actor_var_hidden_layer_sizes,
        actor_hidden_layer_activation=actor_hidden_layer_activation,
        policy_ratio_clip_range=policy_ratio_clip_range,
        gradient_max_norm=gradient_max_norm,
        device=get_device(),
        seed=seed
    )

    # Train agent
    training_stats = agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)
    agent.save(agent_path)
    torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    plot_training_stats(training_stats, results_dir)


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem3"
    results_dir.mkdir(parents=True, exist_ok=True)
    agent_path = results_dir / "ppo.pickle"

    train_ppo(results_dir, agent_path)
    test_rl_agent(agent_path)


if __name__ == "__main__":
    main()
