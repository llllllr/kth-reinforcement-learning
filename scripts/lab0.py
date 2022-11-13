from pathlib import Path
from el2805.lab0.envs import Maze
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent

# map_filepath = Path(__file__).parent.parent / "data" / "maze.txt"
map_filepath = Path(__file__).parent.parent / "data" / "maze_delay.txt"

print("Dynamic programming")
env = Maze(map_filepath=map_filepath, horizon=20)
agent = DynamicProgrammingAgent(env)
agent.solve()
env.render(mode="policy", policy=agent.policy)
print()

print("Value iteration")
env = Maze(map_filepath=map_filepath, discount=.99)
agent = ValueIterationAgent(env)
agent.solve()
env.render(mode="policy", policy=agent.policy)
print()

# env.seed(1)
# done = False
# state = env.reset()
# while not done:
#     env.render()
#     action = agent.compute_action(state)
#     state, reward, done, _ = env.step(action)
# env.render()
