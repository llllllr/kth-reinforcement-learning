import numpy as np
import matplotlib.pyplot as plt
from el2805.envs import Maze, PluckingBerries, MinotaurMaze
from el2805.envs.grid_world import Move
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


def plot_bar(heights, x_tick_labels, y_label, filepath):
    figure, axes = plt.subplots()
    axes.bar_label(axes.bar(
        x=np.arange(len(heights)),
        height=heights,
        tick_label=x_tick_labels
    ))
    axes.set_ylabel(y_label)
    figure.savefig(filepath)
    figure.show()


def plot_surface(x, y, z, x_label, y_label, z_label, filepath):
    figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
    axes.plot_surface(x, y, z, cmap="coolwarm_r")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_zlabel(z_label)
    figure.savefig(filepath)
    figure.show()
