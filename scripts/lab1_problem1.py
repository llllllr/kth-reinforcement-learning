import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from el2805.envs import MinotaurMaze
from el2805.agents import DynamicProgramming, ValueIteration, QLearning, SARSA
from utils import print_and_write_line, minotaur_maze_exit_probability

SEED = 1


def part_c(map_filepath, results_dir):
    results_dir = results_dir / "part_c"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MinotaurMaze(map_filepath=map_filepath, horizon=20)
    agent = DynamicProgramming(env=env)
    agent.solve()

    done = False
    time_step = 0
    env.seed(1)
    state = env.reset()
    env.render()
    while not done:
        action = agent.compute_action(state, time_step)
        state, _, done, _ = env.step(action)
        time_step += 1
        env.render()


def part_d(map_filepath, results_dir):
    results_dir = results_dir / "part_d"
    results_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots()
    write_mode = "w"
    for minotaur_nop in [False, True]:
        print(f"Minotaur NOP: {minotaur_nop}")
        horizons = np.arange(1, 31)

        # trick: instead of solving for every min_horizon<=T<=max_horizon, we solve only for T=max_horizon
        # then, we read the results by hacking the policy to consider the last T time steps
        max_horizon = horizons[-1]
        env = MinotaurMaze(map_filepath=map_filepath, horizon=max_horizon, minotaur_nop=minotaur_nop)
        agent = DynamicProgramming(env=env)
        agent.solve()
        full_policy = agent.policy.copy()

        exit_probabilities = []
        for horizon in horizons:
            agent.policy = full_policy[max_horizon - horizon:]  # trick
            env.horizon = horizon

            exit_probability = minotaur_maze_exit_probability(env, agent)
            exit_probabilities.append(exit_probability)

            print_and_write_line(
                filepath=results_dir / "results.txt",
                output=f"T={horizon} -> P('exit alive')={exit_probability}",
                mode=write_mode
            )
            write_mode = "a"    # append after the first time

        label = ("with " if minotaur_nop else "w/o ") + "stay move"
        axes.plot(horizons, exit_probabilities, label=label)
    axes.set_xlabel("T")
    axes.set_ylabel(r"$\mathbb{P}$('exit alive')")
    axes.set_xticks(horizons[4::5])
    axes.legend()
    figure.savefig(results_dir / "probability_exit.pdf")
    figure.show()


def part_f(map_filepath, results_dir):
    results_dir = results_dir / "part_f"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MinotaurMaze(map_filepath=map_filepath, discount=29/30, poison=True)
    agent = ValueIteration(env=env, precision=1e-2)
    agent.solve()

    exit_probability = minotaur_maze_exit_probability(env, agent)
    print_and_write_line(
        filepath=results_dir / "results.txt",
        output=f"P('exit alive'|'poisoned')={exit_probability}",
        mode="w"
    )
    print()


def part_bonus(map_filepath, results_dir):
    results_dir = results_dir / "part_bonus"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MinotaurMaze(map_filepath=map_filepath, discount=49/50, poison=True, minotaur_chase=True, keys=True)
    start_state = env.reset()
    n_episodes = 50000

    # baseline: Value Iteration
    agent = ValueIteration(env=env, precision=1e-2)
    agent.solve()
    v = agent.v(start_state)
    values_baseline = np.full(n_episodes, v)
    x = np.arange(1, n_episodes+1)

    ########################
    # part (i): Q-learning #
    ########################
    figure, axes = plt.subplots()
    for epsilon, alpha in zip(
            [0.1, 0.5, 0.1, 0.1],
            [2/3, 2/3, 0.6, 0.9]
    ):
        label = rf"$\epsilon$={epsilon:.2f}, $\alpha$={alpha:.2f}"
        agent = QLearning(env=env, learning_rate="decay", alpha=alpha, epsilon=epsilon, seed=SEED)
        env.seed(SEED)
        values = []
        for episode in trange(1, n_episodes+1, desc=f"Q-learning - {label}"):
            done = False
            state = env.reset()
            while not done:
                action = agent.compute_action(state=state, episode=episode)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
            v = agent.v(start_state)
            values.append(v)

        axes.plot(x, values, label=label)
    axes.plot(x, values_baseline, label="VI")
    axes.set_xlabel("number of episodes")
    axes.set_ylabel(r"V($s_0$)")
    axes.legend()
    figure.savefig(results_dir / "q_learning.pdf")
    figure.show()

    ####################
    # part (j): SARSA #
    ####################
    figure, axes = plt.subplots()
    for epsilon, delta in zip(
            [0.2, 0.1, "decay", "decay"],
            [None, None, 0.6, 0.9]
    ):
        if epsilon != "decay":
            label = rf"$\epsilon$={epsilon:.2f}"
        else:
            label = rf"$\delta$={delta:.2f}"

        agent = SARSA(env=env, learning_rate="decay", alpha=2/3, epsilon=epsilon, delta=delta, seed=SEED)
        env.seed(SEED)
        values = []
        for episode in trange(1, n_episodes+1, desc=f"SARSA - {label}"):
            done = False
            state = env.reset()
            action = agent.compute_action(state=state, episode=episode)
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = agent.compute_action(state=next_state, episode=episode)
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            v = agent.v(start_state)
            values.append(v)

        axes.plot(x, values, label=label)
    axes.plot(x, values_baseline, label="VI")
    axes.set_xlabel("number of episodes")
    axes.set_ylabel(r"V($s_0$)")
    axes.legend()
    figure.savefig(results_dir / "sarsa.pdf")
    figure.show()


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab1" / "problem1"
    map_filepath = Path(__file__).parent.parent / "data" / "maze_minotaur.txt"
    map_filepath_key = Path(__file__).parent.parent / "data" / "maze_minotaur_key.txt"

    print("Part (c)")
    part_c(map_filepath, results_dir)
    print()

    print("Part (d)")
    part_d(map_filepath, results_dir)
    print()

    print("Part (f)")
    part_f(map_filepath, results_dir)
    print()

    print("Part BONUS")
    part_bonus(map_filepath_key, results_dir)
    print()


if __name__ == "__main__":
    main()
