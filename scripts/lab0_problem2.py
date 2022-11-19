import numpy as np
from pathlib import Path
from el2805.envs import PluckingBerries
from el2805.agents import DynamicProgrammingAgent, ValueIterationAgent
from utils import best_maze_path


def main():
    map_filepath = Path(__file__).parent.parent / "data" / "plucking_berries.txt"
    horizons = np.arange(1, 31)

    # trick: instead of solving for every min_horizon<=T<=max_horizon, we solve only for T=max_horizon
    # then, we read the results by hacking the policy to consider the last T time steps
    max_horizon = horizons[-1]
    env = PluckingBerries(map_filepath=map_filepath, horizon=max_horizon)
    agent = DynamicProgrammingAgent(env=env)
    agent.solve()
    full_policy = agent.policy.copy()

    for horizon in horizons:
        print(f"Dynamic programming - Maximum value path with T={horizon}")
        agent.policy = full_policy[max_horizon - horizon:]   # trick
        env.horizon = horizon
        env.render(mode="policy", policy=best_maze_path(env, agent))
        print()

    print("Value iteration")
    env = PluckingBerries(map_filepath=map_filepath, discount=.99)
    agent = ValueIterationAgent(env=env, precision=1e-2)
    agent.solve()
    env.render(mode="policy", policy=agent.policy)
    print()


if __name__ == "__main__":
    main()
