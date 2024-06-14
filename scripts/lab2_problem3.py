import gym
import torch
from pathlib import Path
from el2805.agents.rl import PPO
from el2805.agents.rl.utils import get_device
from utils import plot_training_stats, analyze_lunar_lander_agent, analyze_hyperparameter, compare_rl_agent_with_random
from MotionSimulationPlatform import MotionSimulationPlatform
import matplotlib.pyplot as plt
from el2805.agents.utils import running_average
import pickle
import numpy as np

SEED = 1
N_TRAIN_EPISODES = 1500
EARLY_STOP_REWARD = 250
MSPenv = MotionSimulationPlatform(total_time=10, dt=0.01)
AGENT_CONFIG = {
    "seed": SEED,
    # "environment": gym.make("LunarLanderContinuous-v2"),
    "environment" : MSPenv,
    "discount": .99,
    "n_epochs_per_step": 10,
    "epsilon": .2,

    "critic_learning_rate": 5*1e-4,
    
    "critic_hidden_layer_sizes": [64, 32],
    "critic_hidden_layer_activation": "relu",

    "actor_learning_rate": 5*1e-6,
    "actor_shared_hidden_layer_sizes": [64],
    "actor_mean_hidden_layer_sizes": [32],
    "actor_var_hidden_layer_sizes": [32],
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
    # torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    # torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    plot_training_stats(training_stats, results_dir)

def train_more(result_dir):
    with open(Path(__file__).parent.parent / "results_version1_with_limit_0614_small_learnrate" / "ppo.pickle", "rb") as f:
        ppo_model = pickle.load(f)
        training_stats = ppo_model.train(
        n_episodes=500,
        early_stop_reward=EARLY_STOP_REWARD
    )
        
    result_dir.mkdir(parents=True, exist_ok=True)
    ppo_model.save(result_dir  / "ppo_more_train.pickle")
    # torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    # torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    dir = result_dir / "more_train"
    dir.mkdir(parents=True, exist_ok=True)
    plot_training_stats(training_stats, dir)



def reload_nn_and_test():
    nn_dir = Path(__file__).parent.parent / "results_version1_with_limit_0614_small_learnrate_more_train" 
    ppo_model = PPO(**AGENT_CONFIG)
    # ppo_model.actor = torch.load(nn_dir / "actor_with_max_reward.pth")

    # ppo_model.critic = torch.load(nn_dir / "critic_with_max_reward.pth")
    
    with open(nn_dir / "ppo_more_train.pickle", "rb") as f:
        ppo_model = pickle.load(f)

        for i in range(5):
            done = False
            ref_acc, state = ppo_model.environment.reset()
            traj_acc = [state[0]]
            traj_vel = [state[1]]
            traj_pos = [state[2]]
            total_reward = 0
            while not done:

                action = ppo_model.compute_action_test(state=state)
                next_state, reward, done = ppo_model.environment.step(action)
                traj_acc.append(next_state[0])
                traj_vel.append(next_state[1])
                traj_pos.append(next_state[2])
                total_reward += reward
                state = next_state
            t = np.arange(0, 10.0, 0.01)
            plt.subplot(3, 5, (i+1))    
            traj_acc = running_average(traj_acc)
            plt.plot(t, ref_acc[:-1], color='blue', label='reference')
            plt.plot(t, traj_acc[:-1], color='red', label='trajectory')          
            plt.xlabel('Time')
            plt.ylabel('Acceleration')
            plt.title('Reward: '+str(total_reward))
            plt.grid(True)

            plt.subplot(3, 5, 5 + (i+1)) 
            plt.plot(t, traj_vel[:-1])
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.grid(True)

            plt.subplot(3, 5, 10 + (i+1)) 
            plt.plot(t, traj_pos[:-1])
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.grid(True)
        plt.show()
        plt.savefig(nn_dir / "testset_trajectory.jpg")



def main():
    # result_dir = Path(__file__).parent.parent / "results_version1_with_limit_0614_small_learnrate_more_train" 
    # train_more(result_dir)

    print("start of programm")
    reload_nn_and_test()
    print("end of programm")


    # results_dir = Path(__file__).parent.parent / "results_version1_with_limit_0614_small_learnrate" 
    # agent_path = results_dir  / "ppo.pickle"
    # print("Task (c)")
    # task_c(results_dir, agent_path)
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

# def task_e2(results_dir):
#     results_dir.mkdir(parents=True, exist_ok=True)
#     analyze_hyperparameter(
#         agent_class=PPO,
#         agent_config=AGENT_CONFIG,
#         hyperparameter_name="discount",
#         hyperparameter_values=[0.5, 0.99, 1],
#         n_train_episodes=N_TRAIN_EPISODES,
#         early_stop_reward=EARLY_STOP_REWARD,
#         results_dir=results_dir
#     )


# def task_e3(results_dir):
#     results_dir.mkdir(parents=True, exist_ok=True)
#     analyze_hyperparameter(
#         agent_class=PPO,
#         agent_config=AGENT_CONFIG,
#         hyperparameter_name="epsilon",
#         hyperparameter_values=[0.01, 0.2, 0.5],
#         n_train_episodes=N_TRAIN_EPISODES,
#         early_stop_reward=EARLY_STOP_REWARD,
#         results_dir=results_dir
#     )


# def task_f(results_dir, agent_path):
#     results_dir.mkdir(parents=True, exist_ok=True)
#     agent = PPO.load(agent_path)

#     def v(states):
#         v_ = agent.critic(states)
#         return v_

#     def mean_side_engine(states):
#         mean, _ = agent.actor(states)
#         mean_side_engine_ = mean[:, 1]
#         return mean_side_engine_

#     analyze_lunar_lander_agent(
#         agent_function=v,
#         environment=agent.environment,
#         z_label=r"$V_{\omega}(s)$",
#         filepath=results_dir / "critic.pdf"
#     )

#     analyze_lunar_lander_agent(
#         agent_function=mean_side_engine,
#         environment=agent.environment,
#         z_label=r"$\mu_{\theta,2}(s)$",
#         filepath=results_dir / "actor.pdf"
#     )


# def task_g(results_dir, agent_path):
#     results_dir.mkdir(parents=True, exist_ok=True)
#     compare_rl_agent_with_random(
#         agent_path=agent_path,
#         agent_name="ppo",
#         n_episodes=50,
#         seed=SEED,
#         results_dir=results_dir
#     )
