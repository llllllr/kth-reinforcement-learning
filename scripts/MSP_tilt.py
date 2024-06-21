import numpy as np
import gym
import random
import math

class MSP_tilt(gym.Env):
    # 速度和位置 初始随机的范围应该是多少???
    # 到底用什么来限制速度和位置的范围? 通过学习? 通过设置observation space的上下限? 
    def __init__(self, total_time=10.0, dt=0.01):
        self.total_time = total_time
        self.dt = dt
        self._rng = None
        self.seed() 
        self.current_state = None

        self.timesteps =  np.arange(0, self.total_time, self.dt)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.ref_spec_force, self.ref_angular_vel = self.compute_random_reference()
        # [0] accel, [1] vel, [2] position, 
        # [3] angular-vel, [4] angle
        # [5] specidic force, [6] specific force on next step
        state = np.zeros(7)
        # state[1] = random.uniform(-2.0, 2.0)  #  initial vel 
        # state[2] = random.uniform(-2.0, 2.0)  #  initial pos   
        # state[4] = random.uniform(-math.pi/6, math.pi/6 ) # initial angle along y (pitch])
        state[5] = state[0] + 9.81*math.sin(state[4])
        state[6] = self.ref_spec_force[self.current_step + 1]

        self.current_state = state

        return self.ref_spec_force, state
    
    def step(self, action):
        self.current_step += 1
        next_state = np.zeros(7)
        next_state[0] = self.current_state[0] + action[0]
        next_state[1] = self.current_state[1] + self.current_state[0] * self.dt
        next_state[2] = self.current_state[2] + self.current_state[1] * self.dt
        next_state[3] = self.current_state[3] + action[1]
        next_state[4] = self.current_state[4] + self.current_state[3] * self.dt
        next_state[5] = self.current_state[0] + 9.81*math.sin(self.current_state[4])
        
        next_state[6] = self.ref_spec_force[self.current_step + 1]

        # define reward for current state, 这里只使用加速度的差值, 以确保加速度一定verfolgen
        weight_spec_force = 0.9
        weight_ang_vel = 0.9
        weight_action = np.array([0.1, 0.1])

        reward_x_lim = 0
        if (abs(next_state[2]) > 5.0):
            reward_x_lim = -10000

        reward = 1 - weight_spec_force * abs(self.current_state[5] - self.ref_spec_force[self.current_step - 1]) + reward_x_lim - weight_ang_vel * abs(self.current_state[3] - self.ref_angular_vel[self.current_step - 1]) - np.dot(weight_action, abs(action))

        self.current_state = next_state
        if self.current_step >= len(self.timesteps):
            self.done = True
        return next_state, reward, self.done
    
    def compute_random_reference(self):
        start_time = round(random.uniform(0.0, 7.9), 2)
        direction = random.choice([1, -1])
        ref_acceleration = 0
        ref_spec_forces = np.zeros(len(self.timesteps)+2)
        # delta_accel = random.uniform(0.04, 0.06)
        delta_accel = 0.05
        for i, t in enumerate(self.timesteps):
            if t >= start_time and t < start_time + 0.5:
                ref_acceleration += delta_accel 
            elif t >= start_time + 0.5 and t < start_time + 1.5:
                ref_acceleration -= delta_accel
            elif t >= start_time + 1.5 and t < start_time + 2.0 :
                ref_acceleration += delta_accel
            ref_spec_forces[i] = ref_acceleration        

        ref_angular_velocities =  np.zeros(len(self.timesteps)+2)
        ref_ang_vel = 0
        delta_ang_vel = random.uniform(0.04, 0.06)/35
        for i, t in enumerate(self.timesteps):
            if t >= start_time and t < start_time + 1/3:
                ref_ang_vel += delta_ang_vel
            elif t >= start_time + 1/3 and t < start_time + 1:
                ref_ang_vel -= delta_ang_vel
            elif t >= start_time + 1 and t < start_time + 5/3 :
                ref_ang_vel += delta_ang_vel
            elif t >= start_time + 5/3 and t < start_time + 2.02:
                ref_ang_vel -= delta_ang_vel
            ref_angular_velocities[i] = ref_ang_vel 
        return ref_spec_forces*direction, ref_angular_velocities * direction

    def seed(self, seed: int | None = None) -> list[int]: # 返回一个包含seed的列表?
        pass