import sys
import os

# 获取当前工作目录路径
current_path = os.getcwd()

# 将当前路径添加到 Python 搜索路径中
sys.path.append(current_path)


import gym
import torch
from pathlib import Path
from el2805.agents.rl import PPO
from el2805.agents.rl.utils import get_device
from utils import plot_training_stats, analyze_lunar_lander_agent, analyze_hyperparameter, compare_rl_agent_with_random
import pickle
import matplotlib.pyplot as plt
import numpy as np

SEED = 1
N_TRAIN_EPISODES = 1600
EARLY_STOP_REWARD = 250
AGENT_CONFIG = {
    "seed": SEED,
    "environment": gym.make("LunarLanderContinuous-v2"),
    "discount": .99,
    "n_epochs_per_step": 10,
    "epsilon": .2,
    "critic_learning_rate": 1e-3,
    "critic_hidden_layer_sizes": [400, 200],
    "critic_hidden_layer_activation": "relu",
    "actor_learning_rate": 1e-5,
    "actor_shared_hidden_layer_sizes": [400],
    "actor_mean_hidden_layer_sizes": [200],
    "actor_var_hidden_layer_sizes": [200],
    "actor_hidden_layer_activation": "relu",
    "gradient_max_norm": 1,
    "device": get_device(),
}


def task_c(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train agent
    agent = PPO(**AGENT_CONFIG)
    training_stats = agent.train(
        n_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD
    )

    # Save results
    agent.save(agent_path)
    torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    plot_training_stats(training_stats, results_dir)


def task_e2(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    analyze_hyperparameter(
        agent_class=PPO,
        agent_config=AGENT_CONFIG,
        hyperparameter_name="discount",
        hyperparameter_values=[0.5, 0.99, 1],
        n_train_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD,
        results_dir=results_dir
    )


def task_e3(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    analyze_hyperparameter(
        agent_class=PPO,
        agent_config=AGENT_CONFIG,
        hyperparameter_name="epsilon",
        hyperparameter_values=[0.01, 0.2, 0.5],
        n_train_episodes=N_TRAIN_EPISODES,
        early_stop_reward=EARLY_STOP_REWARD,
        results_dir=results_dir
    )


def task_f(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    agent = PPO.load(agent_path)

    def v(states):
        v_ = agent.critic(states)
        return v_

    def mean_side_engine(states):
        mean, _ = agent.actor(states)
        mean_side_engine_ = mean[:, 1]
        return mean_side_engine_

    analyze_lunar_lander_agent(
        agent_function=v,
        environment=agent.environment,
        z_label=r"$V_{\omega}(s)$",
        filepath=results_dir / "critic.pdf"
    )

    analyze_lunar_lander_agent(
        agent_function=mean_side_engine,
        environment=agent.environment,
        z_label=r"$\mu_{\theta,2}(s)$",
        filepath=results_dir / "actor.pdf"
    )


def task_g(results_dir, agent_path):
    results_dir.mkdir(parents=True, exist_ok=True)
    compare_rl_agent_with_random(
        agent_path=agent_path,
        agent_name="ppo",
        n_episodes=50,
        seed=SEED,
        results_dir=results_dir
    )

def reload_nn_and_test():


    nn_dir =Path(__file__).parent.parent / "results" / "lab2" / "problem3" / "task_c"
    with open(nn_dir / "ppo.pickle", "rb") as f:
        ppo_model = pickle.load(f)
        reward_per_episode = []
        length_per_episode = []
        for i in range(200):
                
            done = False
            state = ppo_model.environment.reset()
            # x_posi = [state[0]]
            # y_posi = [state[1]]
            total_reward = 0
            length = 0

            while not done:
                with torch.no_grad():
                    state = torch.as_tensor(
                    data=state.reshape((1,) + state.shape),
                    dtype=torch.float64,
                    device="cpu")
                    mean, var = ppo_model.actor(state)
                    mean, var = mean.reshape(-1), var.reshape(-1)  
                    action = torch.normal(mean, torch.sqrt(var))
                    action = action.numpy()

                next_state, reward, done, _ = ppo_model.environment.step(action)
                # x_posi.append(next_state[0])
                # y_posi.append(next_state[1])
                state = next_state
                total_reward += reward
                length +=1

            reward_per_episode.append(total_reward)
            length_per_episode.append(length)
            # if length % 5 == 0:
                # print(f"The current states are: {next_state}\n The current actions are: {action}. \n ")
                # print(f"The mean and var of the distribution, m: {mean.numpy()}, v: {var.numpy()}. \n")
        plt.subplot(2, 1, 1) 
        plt.plot(np.arange(len(reward_per_episode)), reward_per_episode, color='blue')
        plt.xlabel('episoden')
        plt.ylabel('reward_per_episode')
        # plt.title('With total reward: '+str(total_reward) + " totel length: "+str(length))

        plt.subplot(2, 1, 2) 
        plt.plot(np.arange(len(length_per_episode)), length_per_episode)
        plt.xlabel('Episoden')
        plt.ylabel('length ')


        plt.grid(True)
        plt.show()


def main():
    print("reload the pre-trained NN")
    reload_nn_and_test()
    print("end of programm")


    # results_dir = Path(__file__).parent.parent / "results" / "lab2" / "problem3"
    # agent_path = results_dir / "task_c" / "ppo.pickle"

    # print("Task (c)")
    # task_c(results_dir / "task_c", agent_path)
    # print()

    # print("Task (e2)")
    # task_e2(results_dir / "task_e2")
    # print()

    # print("Task (e3)")
    # task_e3(results_dir / "task_e3")
    # print()

    # print("Task (f)")
    # task_f(results_dir / "task_f", agent_path)
    # print()

    # print("Task (g)")
    # task_g(results_dir / "task_g", agent_path)
    # print()


if __name__ == "__main__":
    main()
