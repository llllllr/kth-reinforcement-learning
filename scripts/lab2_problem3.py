import gym
import torch
from pathlib import Path
from copy import deepcopy
from el2805.agents.rl.deep import PPO
from el2805.agents.rl.deep.utils import get_device
from utils import plot_training_stats, analyze_lunar_lander_agent, compare_rl_agent_with_random

SEED = 1
N_TRAIN_EPISODES = 1600
EARLY_STOP_REWARD = 250
AGENT_CONFIG = {
    "seed": SEED,
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
    agent = PPO(**AGENT_CONFIG)
    training_stats = agent.train(
        n_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD
    )

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
        agent_config_ = deepcopy(AGENT_CONFIG)
        agent_config_["discount"] = discount

        # Train agent
        agent = PPO(**agent_config_)
        training_stats = agent.train(n_episodes=N_TRAIN_EPISODES, early_stop_reward=EARLY_STOP_REWARD)

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
        agent_config_ = deepcopy(AGENT_CONFIG)
        agent_config_["epsilon"] = epsilon

        # Train agent
        agent = PPO(**agent_config_)
        training_stats = agent.train(n_episodes=N_TRAIN_EPISODES, early_stop_reward=EARLY_STOP_REWARD)

        # Save results
        figures = plot_training_stats(
            stats=training_stats,
            results_dir=results_dir,
            label=rf"$\epsilon$={epsilon:.2f}",
            figures=figures
        )


def part_f(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)

    def v(agent, states):
        v_ = agent.critic(states)
        return v_

    def mean_side_engine(agent, states):
        mean, _ = agent.actor(states)
        mean_side_engine_ = mean[:, 1]
        return mean_side_engine_

    analyze_lunar_lander_agent(
        agent_path=agent_path,
        agent_function=v,
        z_label=r"$V_{\omega}(s)$",
        filepath=results_dir / "critic.pdf"
    )

    analyze_lunar_lander_agent(
        agent_path=agent_path,
        agent_function=mean_side_engine,
        z_label=r"$\mu_{\theta,2}(s)$",
        filepath=results_dir / "actor.pdf"
    )


def part_g(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    compare_rl_agent_with_random(
        agent_path=agent_path,
        agent_name="ppo",
        n_episodes=50,
        seed=SEED,
        results_dir=results_dir
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
