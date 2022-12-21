import gym
import torch
from pathlib import Path
from copy import deepcopy
from el2805.agents.rl.deep import PPO
from el2805.agents.rl.deep.utils import get_device
from utils import plot_training_stats

n_episodes = 1600
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


def part_c(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train agent
    agent = PPO(**agent_config)
    training_stats = agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)

    # Save results
    agent.save(results_dir / "ppo.pickle")
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
        training_stats = agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)

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
        training_stats = agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)

        # Save results
        figures = plot_training_stats(
            stats=training_stats,
            results_dir=results_dir,
            label=rf"$\epsilon$={epsilon:.2f}",
            figures=figures
        )


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem3"

    print("Part (c)")
    part_c(results_dir / "part_c")
    print()

    print("Part (e2)")
    part_e2(results_dir / "part_e2")
    print()

    print("Part (e3)")
    part_e3(results_dir / "part_e3")
    print()


if __name__ == "__main__":
    main()
