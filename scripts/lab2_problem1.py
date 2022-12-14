import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from el2805.agents.rl import DQN
from el2805.utils import Experience, NeuralNetwork, running_average
from utils import get_device


def main():
    train = True

    # Prepare results dir
    results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem1"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "neural-network-1.pth"

    # Hyper-parameters
    n_episodes = 1000
    discount = .99
    epsilon_max = .99
    epsilon_min = .05
    epsilon_decay_duration = int(.95 * n_episodes)
    learning_rate = 1e-4
    batch_size = 64
    replay_buffer_size = 5000
    warmup_steps = batch_size
    target_update_frequency = 200
    n_hidden_layers = 1
    hidden_layer_size = 64
    activation = "tanh"
    gradient_clipping_value = 1
    running_avg_window_len = 50
    episode_avg_reward_stop = 200

    # Environment
    environment = gym.make('LunarLander-v2')    # with discrete actions
    environment.reset()
    n_actions = environment.action_space.n
    n_state_features = len(environment.observation_space.high)

    # Agent
    if train:
        q_network = NeuralNetwork(
            input_size=n_state_features,
            output_size=n_actions,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation
        )
    else:
        q_network = torch.load(model_path)
    agent = DQN(
        environment=environment,
        q_network=q_network,
        discount=discount,
        learning_rate=learning_rate,
        replay_buffer_size=replay_buffer_size,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        epsilon="decay",
        epsilon_max=epsilon_max,
        epsilon_min=epsilon_min,
        epsilon_decay_duration=epsilon_decay_duration,
        gradient_clipping_value=gradient_clipping_value,
        cer=True,
        dueling=False,
        device=get_device()
    )

    # Train
    episodes = trange(1, n_episodes + 1, desc='Episode: ', leave=True)
    episode_rewards = []  # total reward per episode
    episode_lengths = []  # number of steps per episode
    for episode in episodes:
        # Reset environment data and initialize variables
        done = False
        state = environment.reset()
        episode_reward = 0.
        episode_length = 0
        if not train:
            environment.render()

        # Run episode
        while not done:
            # Interact with the environment
            action = agent.compute_action(state=state, episode=episode)
            next_state, reward, done, _ = environment.step(action)
            episode_reward += reward
            state = next_state
            episode_length += 1

            # Update policy
            if train:
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                agent.record_experience(experience)
                agent.update()
            else:
                environment.render()

        # Save stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Updates the tqdm update bar with fresh information
        average_episode_reward = running_average(episode_rewards, running_avg_window_len)[-1]
        average_episode_length = running_average(episode_lengths, running_avg_window_len)[-1]
        episodes.set_description(
            f"Episode {episode} - "
            f"Reward/Steps: {episode_reward:.1f}/{episode_length} - "
            f"Avg. Reward/Steps: {average_episode_reward:.1f}/{average_episode_length}"
        )

        # Early stopping
        if average_episode_reward > episode_avg_reward_stop:
            n_episodes = episode
            break

    environment.close()

    if train:
        # Save model
        torch.save(agent.q_network, model_path)

        # Plot rewards
        episodes = np.arange(1, n_episodes + 1)
        figure, axes = plt.subplots()
        axes.plot(episodes, episode_rewards, label='Episode reward')
        axes.plot(episodes, running_average(episode_rewards, running_avg_window_len), label='Avg. episode reward')
        axes.set_xlabel('Episodes')
        axes.set_ylabel('Episode reward')
        axes.legend()
        figure.savefig(results_dir / "episode_reward.pdf")
        figure.show()

        # Plot number of steps
        figure, axes = plt.subplots()
        axes.plot(episodes, episode_lengths, label='Steps per episode')
        axes.plot(
            episodes, running_average(episode_lengths, running_avg_window_len),
            label='Avg. number of steps per episode'
        )
        axes.set_xlabel('Episodes')
        axes.set_ylabel('Episode length')
        axes.legend()
        figure.savefig(results_dir / "episode_length.pdf")
        figure.show()





if __name__ == "__main__":
    main()
