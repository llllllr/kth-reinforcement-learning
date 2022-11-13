from pathlib import Path
from maze import Maze
from dynamic_programming import DynamicProgrammingAgent

map_filepath = Path(__file__).parent.parent / "data" / "maze.txt"

env = Maze(map_filepath=map_filepath, horizon=10)
env.seed(1)

agent = DynamicProgrammingAgent(env)
agent.solve()
# env.render(mode="policy", policy=agent.policy)

done = False
state = env.reset()
while not done:
    env.render()
    action = agent.compute_action(state)
    state, reward, done, _ = env.step(action)
env.render()
