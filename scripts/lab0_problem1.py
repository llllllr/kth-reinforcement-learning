import sys
# 把当前目录加入到模块搜索目录列表里, 这样就能导入模块el2805
sys.path.append('C:/Users/rlin9/Documents/AA_WS2324/小论文/kth-reinforcement-learning/')

from pathlib import Path
from el2805.envs import Maze
from el2805.agents.mdp import DynamicProgramming, ValueIteration
from utils import best_maze_path


def main():
    horizon = 20
    for map_filepath in [
        Path(__file__).parent.parent / "data" / "maze.txt",
        Path(__file__).parent.parent / "data" / "maze_delay.txt",
    ]:
        print(f"Map file: {map_filepath}")

        # 环境包含了: map包含所有枚举对象的二维 array, [[MazeCell.START, MazeCell.WALL, MazeCell.Empty ..], [Empty....]
        # states->list[Position]包括所有可能的state坐标,不包括墙, reward(state, action)->float, 
        # next_states(state, action) ->(array of possible states, array of probability) ,
        # valid_action(state)->list[action_int], state_index(state)->index_in_list,
        # 对每个state进行step(action) -> (下个state, reward, done(达到horizon/到达exit), info)
        environment = Maze(map_filepath=map_filepath, horizon=horizon)

        # agent包含了!!!!
        # compute_action -> action_int或array, 
        # q(state, action, value-function)->单个q-value, 
        # non-stationary DP.solve() -> 从倒数第一个state开始, 
        agent = DynamicProgramming(environment=environment)
        agent.solve()
        # for t in range(horizon):
        #     print(f"Dynamic programming - Policy with {horizon-t} remaining time steps")
        #     env.render(mode="policy", policy=agent.policy[t])
        #     print()

        print("Dynamic programming - Shortest path")
        environment.render(mode="policy", policy=best_maze_path(environment, agent))

        print("Value iteration - Stationary policy")
        environment = Maze(map_filepath=map_filepath)
        agent = ValueIteration(environment=environment, discount=0.99, precision=1e-2)
        agent.solve()
        environment.render(mode="policy", policy=agent.policy)
        print()


if __name__ == "__main__":
    main()
