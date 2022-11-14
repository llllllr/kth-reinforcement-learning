from pathlib import Path
from el2805.lab0.envs import PluckingBerries
from el2805.lab0.agents import DynamicProgrammingAgent, ValueIterationAgent

map_filepath = Path(__file__).parent.parent / "data" / "plucking_berries.txt"

for horizon in range(12, 22):
    print(f"Dynamic programming T={horizon}")
    env = PluckingBerries(map_filepath=map_filepath, horizon=horizon)
    agent = DynamicProgrammingAgent(env)
    agent.solve()
    env.render(mode="policy", policy=agent.policy)
    print()

print("Value iteration")
env = PluckingBerries(map_filepath=map_filepath, discount=.99)
agent = ValueIterationAgent(env)
agent.solve()
env.render(mode="policy", policy=agent.policy)
print()
