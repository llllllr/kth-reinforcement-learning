from pathlib import Path
from el2805.environments import Maze
from el2805.agents.mdp import DynamicProgramming, ValueIteration
from utils import best_maze_path


def main():
    horizon = 20
    for map_filepath in [
        Path(__file__).parent.parent / "data" / "maze.txt",
        Path(__file__).parent.parent / "data" / "maze_delay.txt",
    ]:
        print(f"Map file: {map_filepath}")

        env = Maze(map_filepath=map_filepath, horizon=horizon)
        agent = DynamicProgramming(env=env)
        agent.solve()
        # for t in range(horizon):
        #     print(f"Dynamic programming - Policy with {horizon-t} remaining time steps")
        #     env.render(mode="policy", policy=agent.policy[t])
        #     print()

        print("Dynamic programming - Shortest path")
        env.render(mode="policy", policy=best_maze_path(env, agent))

        print("Value iteration - Stationary policy")
        env = Maze(map_filepath=map_filepath)
        agent = ValueIteration(env=env, discount=0.99, precision=1e-2)
        agent.solve()
        env.render(mode="policy", policy=agent.policy)
        print()


if __name__ == "__main__":
    main()
