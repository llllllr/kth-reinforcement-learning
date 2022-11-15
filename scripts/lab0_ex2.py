from pathlib import Path
from el2805.lab0.envs import PluckingBerries
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent
from utils import best_path


def main():
    map_filepath = Path(__file__).parent.parent / "data" / "plucking_berries.txt"

    min_horizon = 1
    max_horizon = 28

    # trick: instead of solving for every min_horizon<=T<=max_horizon, we solve only for T=max_horizon
    # then, we read the results by hacking the policy by considering the last T time steps
    env = PluckingBerries(map_filepath=map_filepath, horizon=max_horizon)
    agent = DynamicProgrammingAgent(env)
    agent.solve()
    full_policy = agent._policy.copy()
    for horizon in range(min_horizon, max_horizon+1):
        print(f"Dynamic programming - Maximum value path with T={horizon}")
        agent._policy = full_policy[max_horizon-horizon:]   # trick
        env.horizon = horizon
        env.render(mode="policy", policy=best_path(env, agent))
        print()

    print("Value iteration")
    env = PluckingBerries(map_filepath=map_filepath, discount=.99)
    agent = ValueIterationAgent(env)
    agent.solve()
    env.render(mode="policy", policy=agent.policy)
    print()


if __name__ == "__main__":
    main()
