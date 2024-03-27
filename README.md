# Reinforcement Learning

1. 从windows打开anaconda, 选择Powershell Prompt的launch
2. 激活i2dl环境:  conda activate i2dl
3. cd到练习的文件夹:  cd .\Documents\SS23_I2DL\exercise_03\
4. 打开IDE: `jupyter notebook`或者 vscode : `code .`
5. 做练习期间不能关闭这个terminal!!!!
6. 下载需要的包:`pip install -r requirements.txt` 
   
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
- [x] [Q-learning](el2805/agents/rl/q_learning.py)
- [x] [SARSA](el2805/agents/rl/sarsa.py)
- [ ] λ-SARSA
- [x] [DQN](el2805/agents/rl/dqn.py)
- [ ] DDPG
- [x] [PPO](el2805/agents/rl/ppo.py)

The missing algorithms have been implemented by my lab partner and can be found
in [his repository](https://github.com/afcarzero1/ReinforcementLearning).
