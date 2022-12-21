import gym
import torch
from pathlib import Path
from el2805.agents.rl.deep import DQN
from el2805.agents.rl.deep.utils import get_device
from utils import plot_training_stats, test_rl_agent


def train_dqn(results_dir, agent_path):
    # Hyper-parameters
    seed = 1
    n_train_episodes = 1000
    discount = .99
    epsilon = "exponential"
    epsilon_max = .99
    epsilon_min = .05
    epsilon_decay_episodes = int(.9 * n_train_episodes)
    learning_rate = 5e-4
    batch_size = 64
    replay_buffer_size = 10000
    replay_buffer_min = int(.2 * replay_buffer_size)
    target_update_period = replay_buffer_size // batch_size
    hidden_layer_sizes = [64, 64]
    hidden_layer_activation = "relu"
    gradient_max_norm = 1
    cer = True
    dueling = True
    early_stopping_reward = 250

    # Environment
    environment = gym.make("LunarLander-v2")
    environment.seed(seed)

    # Agent
    agent = DQN(
        environment=environment,
        discount=discount,
        learning_rate=learning_rate,
        replay_buffer_size=replay_buffer_size,
        replay_buffer_min=replay_buffer_min,
        batch_size=batch_size,
        target_update_period=target_update_period,
        epsilon=epsilon,
        epsilon_max=epsilon_max,
        epsilon_min=epsilon_min,
        epsilon_decay_duration=epsilon_decay_episodes,
        gradient_max_norm=gradient_max_norm,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_layer_activation=hidden_layer_activation,
        cer=cer,
        dueling=dueling,
        device=get_device(),
        seed=seed
    )

    # Train agent
    training_stats = agent.train(n_episodes=n_train_episodes, early_stopping_reward=early_stopping_reward)
    agent.save(agent_path)
    torch.save(agent.q_network, results_dir / "neural-network-1.pth")
    plot_training_stats(training_stats, results_dir)

    # Test agent
    agent.test(n_episodes=50, render=True)


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem1"
    results_dir.mkdir(parents=True, exist_ok=True)
    agent_path = results_dir / "dqn.pickle"
    train_dqn(results_dir, agent_path)


if __name__ == "__main__":
    main()
