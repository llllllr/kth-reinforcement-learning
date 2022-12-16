import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from el2805.agents.rl import DQN
from el2805.agents.utils import running_average
from utils import get_device

TRAIN = True


def main():
    # Prepare results dir
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem1"
    results_dir.mkdir(parents=True, exist_ok=True)
    agent_path = results_dir / "dqn.pickle"
    model_path = results_dir / "neural-network-1.pth"

    # Environment
    environment = gym.make('LunarLander-v2')  # with discrete actions
    environment.reset()

    if TRAIN:
        # Hyper-parameters
        n_episodes = 500
        discount = .99
        epsilon = "linear_decay"
        epsilon_max = .99
        epsilon_min = .05
        epsilon_decay_duration = int(.9 * n_episodes)
        learning_rate = 1e-3
        batch_size = 256
        replay_buffer_size = 100000
        warmup_steps = batch_size
        target_update_frequency = 200
        n_hidden_layers = 1
        hidden_layer_size = 64
        activation = "tanh"
        gradient_clipping_value = 2
        early_stopping_reward = 200

        # Agent
        agent = DQN(
            environment=environment,
            discount=discount,
            learning_rate=learning_rate,
            replay_buffer_size=replay_buffer_size,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency,
            epsilon=epsilon,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
            epsilon_decay_duration=epsilon_decay_duration,
            gradient_clipping_value=gradient_clipping_value,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            cer=True,
            dueling=False,
            device=get_device()
        )

        # Train and save agent
        training_stats = agent.train(n_episodes=n_episodes, early_stopping_reward=early_stopping_reward)
        agent.save(agent_path, only_nn=False)
        agent.save(model_path, only_nn=True)

        # Plot training stats
        for metric_name, metric_values in training_stats.items():
            metric_name_readable = metric_name.replace("_", " ")
            x = np.arange(1, len(metric_values)+1)
            x_label = "episode" if metric_name.startswith("episode") else "time step"

            figure, axes = plt.subplots()
            axes.plot(x, metric_values, label=metric_name_readable)
            axes.plot(x, running_average(metric_values), label=f"avg {metric_name_readable}")
            axes.set_xlabel(x_label)
            axes.set_ylabel(metric_name_readable)
            axes.legend()
            figure.savefig(results_dir / f"{metric_name}.pdf")
            figure.show()
    else:
        n_episodes = 10
        agent = DQN.load(agent_path)
        agent.test(n_episodes=n_episodes, render=True)

    environment.close()


if __name__ == "__main__":
    main()
