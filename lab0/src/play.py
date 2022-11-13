from pathlib import Path
from envs.maze import Maze

map_filepath = Path(__file__).parent.parent / "data" / "maze.txt"

env = Maze(map_filepath=map_filepath, horizon=11)
env.seed(1)

done = False
state = env.reset()
while not done:
    env.render()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
