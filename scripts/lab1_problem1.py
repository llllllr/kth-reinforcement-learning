import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from el2805.lab1.envs import MazeMinotaur
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent

MAP_FILEPATH = Path(__file__).parent.parent / "data" / "maze_minotaur.txt"


def part_c(results_dir):
    results_dir = results_dir / "part_c"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MazeMinotaur(map_filepath=MAP_FILEPATH, horizon=20)
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

    exit_probabilities = []
    horizons = np.arange(1, 31)
    for horizon in horizons:
        env = MazeMinotaur(map_filepath=MAP_FILEPATH, horizon=horizon)
        agent = DynamicProgrammingAgent(env)
        agent.solve()

        n_episodes = 10000
        n_wins = 0
        for _ in range(n_episodes):
            done = False
            time_step = 0
            env.seed(1)
            state = env.reset()
            while not done:
                action = agent.compute_action(state, time_step)
                state, reward, done, _ = env.step(action)
                time_step += 1
            n_wins += 1 if env.won() else 0

        exit_probability = n_wins / n_episodes
        exit_probabilities.append(exit_probability)
        print(f"T={horizon} -> {exit_probability}")

    figure, axes = plt.subplots()
    axes.plot(horizons, exit_probabilities)
    axes.set_xlabel(R"T")
    axes.set_ylabel(r"$\mathbb{P}$('exit alive')")
    figure.savefig(results_dir / "exit_probabilities.pdf")


def main():
    results_dir = Path(__file__).parent.parent / "results" / "lab1" / "problem1"

    print("Part (c)")
    part_c(results_dir)
    print()

    print("Part (d)")
    part_d(results_dir)
    print()

    # print("Value iteration - Stationary policy")
    # env = MazeMinotaur(map_filepath=map_filepath, discount=29/30)
    # agent = ValueIterationAgent(env)
    # agent.solve()
    # env.render(mode="policy", policy=agent._policy)
    # print()


if __name__ == "__main__":
    main()
