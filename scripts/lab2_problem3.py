import gym
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from el2805.agents.rl.deep import PPO
from el2805.agents.rl.deep.utils import get_device
from el2805.agents.rl import RandomAgent
from utils import plot_training_stats, plot_bar, plot_surface, print_and_write_line

n_train_episodes = 1600
n_test_episodes = 50
early_stopping_reward = 250

agent_config = {
    "seed": 1,
    "environment": gym.make("LunarLanderContinuous-v2"),
    "discount": .99,
    "n_epochs_per_step": 10,
    "epsilon": .2,
    "critic_learning_rate": 1e-3,
    "critic_hidden_layer_sizes": [400, 200],
    "critic_hidden_layer_activation": "relu",
    "actor_learning_rate": 1e-5,
    "actor_shared_hidden_layer_sizes": [400],
    "actor_mean_hidden_layer_sizes": [200],
    "actor_var_hidden_layer_sizes": [200],
    "actor_hidden_layer_activation": "relu",
    "gradient_max_norm": 1,
    "device": get_device(),
}


def part_c(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train agent
    agent = PPO(**agent_config)
    training_stats = agent.train(n_episodes=n_train_episodes, early_stopping_reward=early_stopping_reward)

    # Save results
    agent.save(agent_path)
    torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    plot_training_stats(training_stats, results_dir)


def part_e2(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    figures = None
    for discount in [0.5, 0.99, 1]:
        # Update config
        agent_config_ = deepcopy(agent_config)
        agent_config_["discount"] = discount

        # Train agent
        agent = PPO(**agent_config_)
        training_stats = agent.train(n_episodes=n_train_episodes, early_stopping_reward=early_stopping_reward)

        # Save results
        figures = plot_training_stats(
            stats=training_stats,
            results_dir=results_dir,
            label=rf"$\gamma$={discount:.2f}",
            figures=figures
        )


def part_e3(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    figures = None
    for epsilon in [0.01, 0.2, 0.5]:
        # Update config
        agent_config_ = deepcopy(agent_config)
        agent_config_["epsilon"] = epsilon

        # Train agent
        agent = PPO(**agent_config_)
        training_stats = agent.train(n_episodes=n_train_episodes, early_stopping_reward=early_stopping_reward)

        # Save results
        figures = plot_training_stats(
            stats=training_stats,
            results_dir=results_dir,
            label=rf"$\epsilon$={epsilon:.2f}",
            figures=figures
        )


def part_f(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    agent = PPO.load(agent_path)

    # Prepare grid of states
    y = torch.linspace(start=0, end=1.5, steps=100, dtype=torch.float64)
    w = torch.linspace(start=-np.pi, end=np.pi, steps=100, dtype=torch.float64)
    y_grid, w_grid = torch.meshgrid(y, w, indexing="ij")
    n_states = len(y_grid.reshape(-1))
    state_dim = len(agent.environment.observation_space.low)
    states = torch.zeros((n_states, state_dim), dtype=torch.float64)
    states[:, 1] = y_grid.reshape(-1)
    states[:, 4] = w_grid.reshape(-1)

    # Predict with actor and critic
    with torch.no_grad():
        v = agent.critic(states)
        v = v.reshape(y_grid.shape)
        mean, _ = agent.actor(states)
        mean_side_engine = mean[:, 1]
        mean_side_engine = mean_side_engine.reshape(y_grid.shape)

    # Plot results
    plot_surface(
        x=y_grid,
        y=w_grid,
        z=mean_side_engine,
        x_label="height",
        y_label="angle",
        z_label=r"$\mu_{\theta,2}(s)$",
        filepath=results_dir / "actor.pdf"
    )
    plot_surface(
        x=y_grid,
        y=w_grid,
        z=v,
        x_label="height",
        y_label="angle",
        z_label=r"$V_{\omega}(s)$",
        filepath=results_dir / "critic.pdf"
    )


def part_g(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_ppo = PPO.load(agent_path)
    agent_random = RandomAgent(agent_ppo.environment, seed=agent_config["seed"])
    agents = [agent_ppo, agent_random]
    agent_names = ["ppo", "random"]

    avg_episode_rewards = []
    for agent_name, agent in zip(agent_names, agents):
        test_stats = agent.test(n_episodes=n_test_episodes, render=False)
        avg_episode_reward = np.mean(test_stats["episode_reward"])
        avg_episode_rewards.append(avg_episode_reward)

    plot_bar(
        heights=avg_episode_rewards,
        x_tick_labels=agent_names,
        y_label="avg. episode reward",
        filepath=results_dir / "episode_reward.pdf"
    )


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem3"
    agent_path = results_dir / "part_c" / "ppo.pickle"

    print("Part (c)")
    part_c(results_dir / "part_c", agent_path)
    print()

    print("Part (e2)")
    part_e2(results_dir / "part_e2")
    print()

    print("Part (e3)")
    part_e3(results_dir / "part_e3")
    print()

    print("Part (f)")
    part_f(results_dir / "part_f", agent_path)
    print()

    print("Part (g)")
    part_g(results_dir / "part_g", agent_path)
    print()


if __name__ == "__main__":
    main()
