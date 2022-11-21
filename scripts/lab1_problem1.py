import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from el2805.envs import MinotaurMaze
from el2805.envs.grid_world import Move
from el2805.envs.maze import MazeCell
from el2805.envs.minotaur_maze import Progress
from el2805.agents import DynamicProgramming, ValueIteration, QLearning, SARSA
from utils import print_and_write_line, minotaur_maze_exit_probability, train_rl_agent_one_episode

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

    for t in [0, agent.policy.shape[0]-1]:
        policy = agent.policy[t]
        map_policy = np.zeros(env.map.shape)
        minotaur_position = np.asarray(env.map == MazeCell.EXIT).nonzero()
        minotaur_position = int(minotaur_position[0][0]), int(minotaur_position[1][0])

        for i in range(env.map.shape[0]):
            for j in range(env.map.shape[1]):
                try:
                    state = ((i, j), minotaur_position, Progress.WITH_KEYS)
                    s = env.state_index(state)
                    map_policy[i, j] = policy[s]
                except KeyError:
                    map_policy[i, j] = Move.NOP

        print()
        print(f"Dynamic programming - Minotaur at exit and t={t+1}")
        env.render(mode="policy", policy=map_policy)


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

    expected_life = 30
    env = MinotaurMaze(map_filepath=map_filepath, probability_poison_death=1/expected_life)
    agent = ValueIteration(env=env, discount=1-1/expected_life, precision=1e-2)
    agent.solve()

    exit_probability = minotaur_maze_exit_probability(env, agent)
    print_and_write_line(
        filepath=results_dir / "results.txt",
        output=f"P('exit alive'|'poisoned')={exit_probability}",
        mode="w"
    )
    print()


def part_ij(map_filepath, results_dir):
    results_dir = results_dir / "part_ij"
    results_dir.mkdir(parents=True, exist_ok=True)

    expected_life = 50
    discount = 1 - 1/expected_life
    n_episodes = 100000

    env = MinotaurMaze(
        map_filepath=map_filepath,
        minotaur_chase=True,
        keys=True,
        probability_poison_death=0  # important: we can sample better with infinite horizon
    )

    # baseline: Value Iteration
    start_state = env.reset()
    agent = ValueIteration(env=env, discount=discount, precision=1e-2)
    agent.solve()
    v = agent.v(start_state)
    values_baseline = np.full(n_episodes, v)
    x = np.arange(1, n_episodes+1)

    filename = "part_j3"     # TODO: adjust the filename according to algorithm selected
    figure, axes = plt.subplots()

    for delta, in zip(        # TODO: put here the hyper-parameters under study
        [0.55, 0.65, 0.75, 0.85, 0.95]
    ):
        # TODO: adjust the label for the plot legend according to hyper-parameters under study
        label = rf"$\epsilon={delta:.2f}"

        # agent = QLearning(    # TODO: select the algorithm by commenting/uncommenting
        agent = SARSA(
            env=env,
            learning_rate="decay",
            discount=discount,
            alpha=2/3,
            epsilon="decay",    # TODO: adjust the parameters according to hyper-parameters under study
            delta=delta,
            q_init=0.01,
            seed=SEED
        )

        env.seed(SEED)
        values = []
        for episode in trange(1, n_episodes+1, desc=label):
            train_rl_agent_one_episode(env, agent, episode)
            v = agent.v(start_state)
            values.append(v)

        axes.plot(x, values, label=label)
    axes.plot(x, values_baseline, label="VI")
    axes.set_xlabel("number of episodes")
    axes.set_ylabel(r"V($s_0$)")
    axes.legend()
    figure.savefig(results_dir / f"{filename}.pdf")
    figure.show()


def part_k(map_filepath, results_dir):
    results_dir = results_dir / "part_k"
    results_dir.mkdir(parents=True, exist_ok=True)

    expected_life = 50
    probability_poison_death = 1/expected_life
    discount = 1 - 1/expected_life

    env = MinotaurMaze(
        map_filepath=map_filepath,
        minotaur_chase=True,
        keys=True,
        probability_poison_death=probability_poison_death
    )

    agent_vi = ValueIteration(env=env, discount=discount, precision=1e-2)

    agent_q_learning = QLearning(
        env=env,
        learning_rate="decay",
        discount=discount,
        alpha=2/3,
        epsilon=0.1,
        delta=None,
        q_init=0.01,
        seed=SEED
    )

    agent_sarsa = SARSA(
        env=env,
        learning_rate="decay",
        discount=discount,
        alpha=2/3,
        epsilon=0.1,
        delta=None,
        q_init=0.01,
        seed=SEED
    )

    write_mode = "w"
    for agent, agent_name in zip(
        [agent_vi, agent_q_learning, agent_sarsa],
        ["vi", "q_learning", "sarsa"]
    ):
        # train
        n_episodes = 50000
        if agent_name != "vi":
            for episode in trange(1, n_episodes+1, desc=agent_name):
                train_rl_agent_one_episode(env, agent, episode)
        else:
            agent.solve()

        # test
        exit_probability = minotaur_maze_exit_probability(env, agent)
        print_and_write_line(
            filepath=results_dir / "results.txt",
            output=f"{agent_name}: P('exit alive'|'poisoned')={exit_probability}",
            mode=write_mode
        )
        write_mode = "a"    # append after the first time
        print()


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

    print("Part (i-j)")
    part_ij(map_filepath_key, results_dir)
    print()

    print("Part (k)")
    part_k(map_filepath_key, results_dir)
    print()


if __name__ == "__main__":
    main()
