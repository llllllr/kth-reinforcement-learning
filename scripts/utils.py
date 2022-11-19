import numpy as np
from el2805.envs import Maze, PluckingBerries, MinotaurMaze
from el2805.envs.grid_world import Move


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


def minotaur_maze_exit_probability(env, agent):
    assert type(env) == MinotaurMaze
    n_episodes = 10000
    n_wins = 0
    for episode in range(1, n_episodes+1):
        done = False
        time_step = 0
        env.seed(episode)
        state = env.reset()
        while not done:
            action = agent.compute_action(state=state, time_step=time_step)
            state, _, done, _ = env.step(action)
            time_step += 1
        n_wins += 1 if env.won() else 0
    exit_probability = n_wins / n_episodes
    return exit_probability


def print_and_write_line(filepath, output, mode):
    print(output)
    with open(filepath, mode=mode) as f:
        f.write(output + "\n")
