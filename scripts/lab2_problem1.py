import gym
import torch
from pathlib import Path
from el2805.agents.rl.deep import DQN
from el2805.agents.rl.deep.utils import get_device
from utils import plot_training_stats, analyze_lunar_lander_agent, analyze_hyperparameter, compare_rl_agent_with_random

N_TRAIN_EPISODES = 1000
SEED = 1
EARLY_STOP_REWARD = 250
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
AGENT_CONFIG = {
    "seed": 1,
    "environment": gym.make("LunarLander-v2"),
    "discount": .99,
    "epsilon": "exponential",
    "epsilon_max": .99,
    "epsilon_min": .05,
    "epsilon_decay_duration": int(.9 * N_TRAIN_EPISODES),
    "learning_rate": 5e-4,
    "batch_size": BATCH_SIZE,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
    "replay_buffer_min": int(.2 * REPLAY_BUFFER_SIZE),
    "target_update_period": REPLAY_BUFFER_SIZE // BATCH_SIZE,
    "hidden_layer_sizes": [64, 64],
    "hidden_layer_activation": "relu",
    "gradient_max_norm": 1,
    "cer": True,
    "dueling": False,
    "device": get_device(),
}


def task_c(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train agent
    agent = DQN(**AGENT_CONFIG)
    training_stats = agent.train(
        n_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD
    )

    # Save results
    agent.save(agent_path)
    torch.save(agent.q_network, results_dir / "neural-network-1.pth")
    plot_training_stats(training_stats, results_dir)


def task_e2(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    analyze_hyperparameter(
        agent_class=DQN,
        agent_config=AGENT_CONFIG,
        hyperparameter_name="discount",
        hyperparameter_values=[0.5, 0.99, 1],
        n_train_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD,
        results_dir=results_dir
    )


def task_e3(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Effect of number of episodes
    # results_dir_tmp = results_dir / "n_episodes"
    # results_dir_tmp.mkdir(parents=True, exist_ok=True)
    # agent = DQN(**AGENT_CONFIG)
    # training_stats = agent.train(
    #     n_episodes=5000,
    #     early_stop_reward=None
    # )
    # plot_training_stats(training_stats, results_dir_tmp)

    # Effect of memory size
    results_dir_tmp = results_dir / "replay_buffer_size"
    results_dir_tmp.mkdir(parents=True, exist_ok=True)
    analyze_hyperparameter(
        agent_class=DQN,
        agent_config=AGENT_CONFIG,
        hyperparameter_name="replay_buffer_size",
        hyperparameter_values=[1000, 10000, 100000],
        n_train_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD,
        results_dir=results_dir_tmp
    )


def task_f(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    agent = DQN.load(agent_path)

    def v(states):
        q = agent.q_network(states)
        v_ = q.max(dim=1).values
        return v_

    def policy(states):
        q = agent.q_network(states)
        actions_ = q.argmax(dim=1)
        return actions_

    analyze_lunar_lander_agent(
        agent_function=v,
        environment=agent.environment,
        z_label=r"$V_{\theta}(s)$",
        filepath=results_dir / "value_function.pdf"
    )

    analyze_lunar_lander_agent(
        agent_function=policy,
        environment=agent.environment,
        z_label=r"$\pi_{\theta}(s)$",
        filepath=results_dir / "policy.pdf"
    )


def task_g(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    compare_rl_agent_with_random(
        agent_path=agent_path,
        agent_name="dqn",
        n_episodes=50,
        seed=SEED,
        results_dir=results_dir
    )


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem1"
    agent_path = results_dir / "task_c" / "dqn.pickle"

    print("Task (c)")
    task_c(results_dir / "task_c", agent_path)
    print()

    print("Task (e2)")
    task_e2(results_dir / "task_e2")
    print()

    print("Task (e3)")
    task_e3(results_dir / "task_e3")
    print()

    print("Task (f)")
    task_f(results_dir / "task_f", agent_path)
    print()

    print("Task (g)")
    task_g(results_dir / "task_g", agent_path)
    print()


if __name__ == "__main__":
    main()
