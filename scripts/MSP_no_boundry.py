import numpy as np
import gym
import random

class MSP(gym.Env):
    # 速度和位置 初始随机的范围应该是多少???
    # 到底用什么来限制速度和位置的范围? 通过学习? 通过设置observation space的上下限? 
    def __init__(self, total_time=10.0, dt=0.01):
        self.total_time = total_time
        self.dt = dt
        self._rng = None
        self.seed() 
        self.current_state = None

        self.position = 0.0
        self.velocity = 0.0
        self.timesteps =  np.arange(0, self.total_time, self.dt)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.ref_accelerations = self.compute_random_reference()
        state = np.zeros(4) # [0] accel, [1] vel, [2] position
        state[3] = self.ref_accelerations[self.current_step + 1]
        # state[1] = random.uniform(-2.0, 2.0)  # randomrize the initial vel of each episode
        # state[2] = random.uniform(-2.0, 2.0)  # randomrize the initial pos of each episode  
        self.current_state = state

        return self.ref_accelerations, state
    
    def step(self, action):
        self.current_step += 1
        next_state = np.zeros(4)
        next_state[0] = self.current_state[0] + action[0]
        next_state[1] = self.current_state[1] + self.current_state[0] * self.dt
        next_state[2] = self.current_state[2] + self.current_state[1] * self.dt
        next_state[3] = self.ref_accelerations[self.current_step + 1]

        # define reward for current state, 这里只使用加速度的差值, 以确保加速度一定verfolgen
        weight_accel = 0.9
        reward = 1 - weight_accel * abs(self.current_state[0] - self.ref_accelerations[self.current_step - 1]) - 0
        self.current_state = next_state
        if self.current_step >= len(self.timesteps):
            self.done = True
        return next_state, reward, self.done
    
    def compute_random_reference(self):
        start_time = round(random.uniform(0.0, 7.9), 2)
        direction = random.choice([1, -1])
        ref_acceleration = 0
        ref_accelerations = np.zeros(len(self.timesteps)+2)
        # delta_accel = random.uniform(0.04, 0.06)
        delta_accel = 0.05
        for i, t in enumerate(self.timesteps):
            if t >= start_time and t < start_time + 0.5:
                ref_acceleration += delta_accel 
            elif t >= start_time + 0.5 and t < start_time + 1.5:
                ref_acceleration -= delta_accel
            elif t >= start_time + 1.5 and t < start_time + 2.0 :
                ref_acceleration += delta_accel
            ref_accelerations[i] = ref_acceleration        

        return ref_accelerations*direction

    def seed(self, seed: int | None = None) -> list[int]: # 返回一个包含seed的列表?
        pass