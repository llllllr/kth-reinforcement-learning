from pathlib import Path
from el2805.lab0.envs import Maze
from el2805.lab0.agents import DynamicProgrammingAgent

# map_filepath = Path(__file__).parent.parent / "data" / "maze.txt"
map_filepath = Path(__file__).parent.parent / "data" / "maze_delay.txt"

env = Maze(map_filepath=map_filepath, horizon=13)
env.seed(1)

agent = DynamicProgrammingAgent(env)
agent.solve()
env.render(mode="policy", policy=agent.policy)

# done = False
# state = env.reset()
# while not done:
#     env.render()
#     action = agent.compute_action(state)
#     state, reward, done, _ = env.step(action)
# env.render()
