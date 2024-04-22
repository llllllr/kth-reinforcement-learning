import gym
import torch
from pathlib import Path
from el2805.agents.rl import PPO
from el2805.agents.rl.utils import get_device
from utils import plot_training_stats, analyze_lunar_lander_agent, analyze_hyperparameter, compare_rl_agent_with_random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from MotionSimulationPlatform import MotionSimulationPlatform
SEED = 1
N_TRAIN_EPISODES = 900
EARLY_STOP_REWARD = 250
MSPenv = MotionSimulationPlatform(total_time=10, dt=0.01)
AGENT_CONFIG = {
    "seed": SEED,
    # "environment": gym.make("LunarLanderContinuous-v2"),
    "environment" : MSPenv,
    "discount": .99,
    "n_epochs_per_step": 10,
    "epsilon": .2,

    "critic_learning_rate": 1e-3,
    
    "critic_hidden_layer_sizes": [64, 32],
    "critic_hidden_layer_activation": "relu",

    "actor_learning_rate": 1e-5,
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

    # with open(Path(__file__).parent.parent / "results_version1" / "withvariantAccelerations" / "ppo_MSP"/ "task_c" / "ppo.pickle", "rb") as f:
    #     agent = pickle.load(f)
    #     training_stats = agent.train(
    #     n_episodes=N_TRAIN_EPISODES,
    #     early_stop_reward=EARLY_STOP_REWARD
    # )


    # Save results
    agent.save(agent_path)
    torch.save(agent.actor, results_dir / "neural-network-3-actor.pth")
    torch.save(agent.critic, results_dir / "neural-network-3-critic.pth")
    plot_training_stats(training_stats, results_dir)


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

def reload_nn_and_test():
    nn_dir = Path(__file__).parent.parent / "results_version1" / "withvariantAccelerations" / "ppo_MSP"/ "task_c"
    with open(nn_dir / "ppo.pickle", "rb") as f:
        ppo_model = pickle.load(f)
        done = False
        ref_acc, state = ppo_model.environment.reset()
        traj_acc = [state[0]]
        traj_vel = [state[1]]
        traj_pos = [state[2]]
        total_reward = 0
        while not done:
            action = ppo_model.compute_action(state=state)
            next_state, reward, done = ppo_model.environment.step(action)
            traj_acc.append(next_state[0])
            traj_vel.append(next_state[1])
            traj_pos.append(next_state[2])
            total_reward += reward
            state = next_state
        plt.subplot(3, 1, 1)        
        plt.plot(np.arange(0, 10.0, 0.01), ref_acc[:-1], color='blue', label='reference')
        plt.plot(np.arange(0, 10.0, 0.01), traj_acc[:-1], color='red', label='trajectory')          
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Acceleration difference with reward: '+str(total_reward))

        plt.subplot(3, 1, 2) 
        plt.plot(np.arange(0, 10.0, 0.01), traj_pos[:-1])
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.subplot(3, 1, 3) 
        plt.plot(np.arange(0, 10.0, 0.01), traj_vel[:-1])
        plt.xlabel('Time')
        plt.ylabel('Velocity')

        plt.grid(True)
        plt.show()







def main():
    results_dir = Path(__file__).parent.parent / "results_version1_more_episoden" / "withvariantAccelerations" / "ppo_MSP"
    agent_path = results_dir / "task_c" / "ppo.pickle"

    print("Task (c)")
    task_c(results_dir / "task_c", agent_path)
    print()

    # reload_nn_and_test()



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
