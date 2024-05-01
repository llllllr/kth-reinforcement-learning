import numpy as np
import matplotlib.pyplot as plt
import random
from gym.utils.seeding import np_random
import gym


class MotionSimulationPlatform(gym.Env):
                  
    def __init__(self, total_time=10.0, dt=0.01):
        self.total_time = total_time
        self.dt = dt
        self._rng = None
        self.seed() 
        self.current_state = None

        # initial state
        self.position = 0.0
        self.velocity = 0.0
        self.timesteps =  np.arange(0, self.total_time, self.dt)

    def reset(self):
        # reset the env to a random initial state, type: nparray
        state = np.zeros(3) 
        state[1] = random.uniform(-2.0, 2.0)  # randomrize the initial vel of each episode
        state[2] = random.uniform(-2.0, 2.0)  # randomrize the initial pos of each episode
        self.current_state = state
        self.current_step = 0
        self.ref_accelerations = self.compute_random_reference()
        self.done = False
        return self.ref_accelerations, state


    # next_state, reward, done, _ = self.environment.step(action), 
    def step(self, action):
        # states are [0] accel, [1] vel, [2] position
        # action: change of the acceleration, delte_a
        next_state = np.zeros(3)
        next_state[0] = self.current_state[0] + action[0]
        next_state[1] = self.current_state[1] + self.current_state[0] * self.dt
        next_state[2] = self.current_state[2] + self.current_state[1] * self.dt

        # compute the reward, w.r.t the reference-accelaration & angular-velocity
        weight_accel = 10
        weight_posi = 0.9
        weight_action = 0.9  # weight_action = np.array([0.9, 0.9])
        reward = 1 - weight_accel*abs(next_state[0] - self.ref_accelerations[self.current_step+1]) \
                    - weight_posi*abs(next_state[2]) - weight_action * abs(action)
        # reward_x_lim = 0
        # if (abs(next_state[2] > 5.0)):
        #     reward_x_lim = - 10000
        # reward = 1 - weight_accel*abs(next_state[0] - self.ref_accelerations[self.current_step+1]) \
        #              - weight_action * abs(action) + reward_x_lim
        # np.dot(weight_action, action)

        self.current_step += 1
        self.current_state = next_state
        if self.current_step >= len(self.timesteps):
            self.done = True
        return next_state, reward, self.done
            

    def compute_random_reference(self):
        start_time = round(random.uniform(0.0, 7.9), 2)
        direction = random.choice([1, -1])
        ref_acceleration = 0
        ref_accelerations = np.zeros(len(self.timesteps)+1)
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

        """Sets the seed of the environment's internal RNG.

        :param seed: seed
        :type seed: int, optional
        """
        pass
        # self._rng, seed = np_random(seed)
        # return [seed]
    


        # we have 8 dimensions: in x-direction: x-accel, y-accel,
        #                                       x-vel, y-vel,                         
        #                                       x-pos, y-pos, 
        #                                       w-yaw, phi-yaw
        
        # # update states: [0:2] x-accel, y-accel, [2:4] x-vel, y-vel, [4:6]  x-pos, y-pos, 
        # # [6] angular velocity in yaw, [7] yaw-angular 
        # next_state = np.zeros(8)
        # next_state[0:2] = self.current_state[0:2] + actions[0:2]
        # next_state[2:4] = self.current_state[2:4] + self.current_state[0:2] * self.dt
        # next_state[4:6] = self.current_state[4:6] + self.current_state[2:4] * self.dt
        # next_state[6] = self.current_state[6] + actions[2]
        # next_state[7] = self.current_state[7] + self.current_state[6]