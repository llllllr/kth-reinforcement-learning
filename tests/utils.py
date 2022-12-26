import numpy as np
from tqdm import trange


def test(test_case, environment, compute_action):
    n_episodes = 50
    confidence_pass = 50

    episode_rewards = []
    episodes = trange(n_episodes, desc='Episode: ', leave=True)
    for episode in episodes:
        episodes.set_description(f"Episode {episode}")
        done = False
        state = environment.reset()
        episode_reward = 0.
        while not done:
            action = compute_action(state)
            next_state, reward, done, _ = environment.step(action)
            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)
        environment.close()

    # Assumption: episode reward has Gaussian distribution
    # Goal: estimate the mean value by taking the sample mean
    # Problem: how close the sample mean is from the true mean value?
    #
    # Confidence level: 0.95
    # Confidence interval: (sample_mean - confidence, sample_mean + confidence)
    # Confidence: confidence = q_0.975 * std_reward / sqrt(n)
    #
    # See "Philosophy of Science and Research Methodology" course
    avg_reward = np.mean(episode_rewards)
    confidence = np.std(episode_rewards) * 1.96 / np.sqrt(n_episodes)
    test_case.assertTrue(
        expr=avg_reward - confidence >= confidence_pass,
        msg=f"Avg reward ({avg_reward}) - Confidence ({confidence}) < Confidence pass ({confidence_pass})"
    )
