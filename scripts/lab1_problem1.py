from pathlib import Path
from el2805.lab1.envs import MazeMinotaur
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent


def main():
    map_filepath = Path(__file__).parent.parent / "data" / "maze_minotaur.txt"

    env = MazeMinotaur(map_filepath=map_filepath, horizon=20)
    agent = DynamicProgrammingAgent(env)
    agent.solve()

    print("Dynamic programming - Play an episode")
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
    print()

    print("Value iteration - Stationary policy")
    env = MazeMinotaur(map_filepath=map_filepath, discount=29/30)
    agent = ValueIterationAgent(env)
    agent.solve()
    env.render(mode="policy", policy=agent._policy)
    print()


if __name__ == "__main__":
    main()
