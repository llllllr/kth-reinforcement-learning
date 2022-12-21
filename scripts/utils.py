import numpy as np
import torch
import matplotlib.pyplot as plt
from el2805.envs import Maze, PluckingBerries, MinotaurMaze
from el2805.envs.grid_world import Move
from el2805.agents.rl import RLAgent, RandomAgent
from el2805.agents.rl.utils import Experience
from el2805.agents.utils import running_average


def best_maze_path(env, agent):
    assert type(env) == Maze or type(env) == PluckingBerries
    best_path_ = np.full(len(env.states), fill_value=Move.NOP)
    done = False
    time_step = 0
    env.seed(1)
    state = env.reset()
    while not done:
        action = agent.compute_action(state=state, time_step=time_step)
        s = env.state_index(state)
        best_path_[s] = Move(action)
        state, _, done, _ = env.step(action)
        time_step += 1
    return best_path_


def minotaur_maze_exit_probability(environment, agent):
    assert type(environment) == MinotaurMaze
    n_episodes = 10000
    n_wins = 0
    for episode in range(1, n_episodes+1):
        done = False
        time_step = 0
        environment.seed(episode)
        state = environment.reset()
        while not done:
            action = agent.compute_action(state=state, time_step=time_step, explore=False)
            state, _, done, info = environment.step(action)
            time_step += 1
            n_wins += info["won"]
    exit_probability = n_wins / n_episodes
    return exit_probability


def train_rl_agent_one_episode(environment, agent, episode):
    done = False
    state = environment.reset()
    while not done:
        action = agent.compute_action(state=state, episode=episode)
        next_state, reward, done, _ = environment.step(action)

        experience = Experience(
            episode=episode,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        agent.record_experience(experience)
        agent.update()

        state = next_state


def analyze_lunar_lander_agent(agent_path, agent_function, z_label, filepath):
    agent = RLAgent.load(agent_path)

    # Prepare grid of states
    w = torch.linspace(start=-torch.pi, end=torch.pi, steps=100, dtype=torch.float64)
    y = torch.linspace(start=0, end=1.5, steps=100, dtype=torch.float64)
    w_grid, y_grid = torch.meshgrid(w, y, indexing="ij")
    n_states = len(y_grid.reshape(-1))
    state_dim = len(agent.environment.observation_space.low)
    states = torch.zeros((n_states, state_dim), dtype=torch.float64)
    states[:, 1] = y_grid.reshape(-1)
    states[:, 4] = w_grid.reshape(-1)

    # Agent output
    with torch.no_grad():
        z = agent_function(agent, states)
        z_grid = z.reshape(y_grid.shape)

    # Plot results
    figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
    plot = axes.plot_surface(w_grid, y_grid, z_grid, cmap="coolwarm")
    axes.set_xlabel("angle")
    axes.set_ylabel("height")
    axes.set_zlabel(z_label)
    figure.colorbar(plot, location="left")
    figure.savefig(filepath)
    figure.show()


def compare_rl_agent_with_random(agent_path, agent_name, n_episodes, seed, results_dir):
    # Test agents
    agent = RLAgent.load(agent_path)
    agent_random = RandomAgent(agent.environment, seed=seed)
    agents = [agent, agent_random]
    agent_names = [agent_name, "random"]
    avg_episode_rewards = []
    for agent_name, agent in zip(agent_names, agents):
        test_stats = agent.test(n_episodes=n_episodes, render=False)
        avg_episode_reward = np.mean(test_stats["episode_reward"])
        avg_episode_rewards.append(avg_episode_reward)

    # Plot results
    figure, axes = plt.subplots()
    axes.bar_label(axes.bar(
        x=np.arange(len(avg_episode_rewards)),
        height=avg_episode_rewards,
        tick_label=agent_names
    ))
    axes.set_ylabel("avg. episode reward")
    figure.savefig(results_dir / f"{agent_name}_vs_random.pdf")
    figure.show()


def print_and_write_line(filepath, output, mode):
    print(output)
    with open(filepath, mode=mode) as f:
        f.write(output + "\n")


def plot_training_stats(stats, results_dir, label=None, figures=None):
    if figures is None:
        figures = {metric_name: plt.subplots()[0] for metric_name in stats.keys()}
    else:
        for metric_name in stats.keys():
            assert metric_name in figures

    for metric_name, metric_values in stats.items():
        figure = figures[metric_name]
        assert len(figure.axes) == 1
        axes = figure.axes[0]

        metric_name_readable = metric_name.replace("_", " ")
        x = np.arange(1, len(metric_values) + 1)
        x_label = "episode" if metric_name.startswith("episode") else "update"

        plots = axes.plot(x, running_average(metric_values), label=label)
        color = plots[0].get_color()
        axes.plot(x, metric_values, alpha=.2, color=color)
        axes.set_xlabel(x_label)
        axes.set_ylabel(metric_name_readable)
        if label is not None:
            axes.legend()

        figure.savefig(results_dir / f"{metric_name}.pdf")
        figure.show()

    return figures
