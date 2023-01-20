# Reinforcement Learning

Assignments of the Reinforcement Learning course (EL2805) at KTH.

The repository provides a [Python package](el2805), which includes:

- [Dynamic programming algorithms for the exact solution of MDPs](el2805/agents/mdp)
- [RL algorithms](el2805/agents/rl)
- [Simulators for problems modeled as MDPs](el2805/envs)

You can install the Python package as follows:

```shell
pip install -e .
```

Additionally, the repository contains [documentation](docs) (instructions and reports), [scripts](scripts)
and [tests](tests). You can run scripts and tests as follows:

```shell
python scripts/<script_name>  # run a particular script
python -m unittest            # run all tests
```

Available algorithms:

- [x] [Dynamic programming](el2805/agents/mdp/dynamic_programming.py)
- [x] [Value iteration](el2805/agents/mdp/value_iteration.py)
- [x] [Tabular Q-learning](el2805/agents/rl/tabular/q_learning.py)
- [x] [Tabular SARSA](el2805/agents/rl/tabular/sarsa.py)
- [ ] λ-SARSA
- [x] [DQN](el2805/agents/rl/deep/dqn.py)
- [ ] DDPG
- [x] [PPO](el2805/agents/rl/deep/ppo.py)

The missing algorithms have been implemented by my lab partner and can be found
in [his repository](https://github.com/afcarzero1/ReinforcementLearning).
