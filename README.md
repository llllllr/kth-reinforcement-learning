# Reinforcement Learning

Assignments of the Reinforcement Learning course (EL2805) at KTH.

The repository provides a [Python package](el2805), which includes:
- [Simulators for problems modeled as MDPs](el2805/envs)
- [Algorithms for the exact solution of fully known MDPs](el2805/agents/mdp)
- [RL algorithms](el2805/agents/rl)

You can install the Python package as follows:
```shell
pip install -e .
```

Additionally, the repository contains [documentation](docs) (instructions and reports), [scripts](scripts) and [tests](tests). You can run scripts and tests as follows:
```shell
python scripts/<script_name>  # run a particular script
python -m unittest            # run all tests
```
