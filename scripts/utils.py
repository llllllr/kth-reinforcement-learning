import numpy as np
from el2805.environments import Maze, PluckingBerries, MinotaurMaze
from el2805.environments.grid_world import Move


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
            action = agent.compute_action(state=state, time_step=time_step, episode=episode, explore=False)
            state, _, done, _ = environment.step(action)
            time_step += 1
        n_wins += 1 if environment.won() else 0
    exit_probability = n_wins / n_episodes
    return exit_probability


def train_rl_agent_one_episode(environment, agent, episode):
    done = False
    state = environment.reset()
    action = agent.compute_action(state=state, episode=episode)
    while not done:
        next_state, reward, done, _ = environment.step(action)
        next_action = agent.compute_action(state=next_state, episode=episode)
        agent.update(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)
        state = next_state
        action = next_action


def print_and_write_line(filepath, output, mode):
    print(output)
    with open(filepath, mode=mode) as f:
        f.write(output + "\n")
