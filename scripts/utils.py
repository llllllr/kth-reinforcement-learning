import numpy as np
from el2805.lab0.envs.grid_world import Move


def best_path(env, agent):
    best_path_ = np.full(len(env.states), fill_value=Move.NOP)
    done = False
    time_step = 0
    env.seed(1)
    state = env.reset()
    while not done:
        s = env.state_to_index(state)
        action = agent.compute_action(state, time_step)
        best_path_[s] = Move(action)
        state, _, done, _ = env.step(action)
        time_step += 1
    return best_path_


def print_write_line(filepath, output):
    print(output)
    with open(filepath, mode="a") as f:
        f.write(output + "\n")
