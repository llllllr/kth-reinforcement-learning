import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from el2805.lab1.envs import MinotaurMaze
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent
from utils import print_and_write_line

MAP_FILEPATH = Path(__file__).parent.parent / "data" / "maze_minotaur.txt"


def part_c(results_dir):
    results_dir = results_dir / "part_c"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MinotaurMaze(map_filepath=MAP_FILEPATH, horizon=20)
    agent = DynamicProgrammingAgent(env)
    agent.solve()

    done = False
    time_step = 0
    env.seed(1)
    state = env.reset()
    env.render()
    while not done:
        action = agent.compute_action(state, time_step)
        state, reward, done, _ = env.step(action)
        time_step += 1
        env.render()


def part_d(results_dir):
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
        env = MinotaurMaze(map_filepath=MAP_FILEPATH, horizon=max_horizon, minotaur_nop=minotaur_nop)
        agent = DynamicProgrammingAgent(env)
        agent.solve()
        full_policy = agent.policy.copy()

        exit_probabilities = []
        for horizon in horizons:
            agent.policy = full_policy[max_horizon - horizon:]  # trick
            env.horizon = horizon

            n_episodes = 10000
            n_wins = 0
            for i in range(n_episodes):
                done = False
                time_step = 0
                env.seed(i)
                state = env.reset()
                while not done:
                    action = agent.compute_action(state, time_step)
                    state, reward, done, _ = env.step(action)
                    time_step += 1
                n_wins += 1 if env.won() else 0

            exit_probability = n_wins / n_episodes
            exit_probabilities.append(exit_probability)
            print_and_write_line(
                filepath=results_dir / "results.txt",
                output=f"T={horizon} -> P('exit alive')={exit_probability}",
                mode=write_mode
            )
            write_mode = "a"    # append after the first time

        label = ("with " if minotaur_nop else "w/o ") + "stay move"
        axes.plot(horizons, exit_probabilities, label=label)
        axes.set_xlabel(R"T")
        axes.set_ylabel(r"$\mathbb{P}$('exit alive')")
        axes.set_xticks(horizons[4::5])
    axes.legend()
    figure.savefig(results_dir / "probability_exit.pdf")


def part_f(results_dir):
    results_dir = results_dir / "part_f"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MinotaurMaze(map_filepath=MAP_FILEPATH, discount=29/30, poison=True)
    agent = ValueIterationAgent(env)
    agent.solve()

    n_episodes = 10000
    n_wins = 0
    for i in range(n_episodes):
        done = False
        time_step = 0
        env.seed(i)
        state = env.reset()
        while not done:
            action = agent.compute_action(state, time_step)
            state, reward, done, _ = env.step(action)
            time_step += 1
        n_wins += 1 if env.won() else 0

    exit_probability = n_wins / n_episodes
    print_and_write_line(
        filepath=results_dir / "results.txt",
        output=f"P('exit alive'|'poisoned')={exit_probability}",
        mode="w"
    )
    print()


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab1" / "problem1"

    print("Part (c)")
    part_c(results_dir)
    print()

    print("Part (d)")
    part_d(results_dir)
    print()

    print("Part (f)")
    part_f(results_dir)
    print()


if __name__ == "__main__":
    main()
