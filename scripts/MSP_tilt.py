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
        # [5] specific force, [6] specific force on next step, [7] ang_vel on next step
        state = np.zeros(8)
        # state[1] = random.uniform(-2.0, 2.0)  #  initial vel 
        # state[2] = random.uniform(-2.0, 2.0)  #  initial pos   
        # state[4] = random.uniform(-math.pi/6, math.pi/6 ) # initial angle along y (pitch])
        state[5] = state[0] + 9.81*math.sin(state[4])
        state[6] = self.ref_spec_force[self.current_step + 1]
        state[7] = self.ref_angular_vel[self.current_step + 1]

        self.current_state = state

        return self.ref_spec_force, self.ref_angular_vel, state
    
    def step(self, action):
        self.current_step += 1
        next_state = np.zeros(8)
        next_state[0] = self.current_state[0] + action[0]
        next_state[1] = self.current_state[1] + self.current_state[0] * self.dt
        next_state[2] = self.current_state[2] + self.current_state[1] * self.dt
        next_state[3] = self.current_state[3] + action[1]
        next_state[4] = self.current_state[4] + self.current_state[3] * self.dt
        next_state[5] = self.current_state[0] + 9.81*math.sin(self.current_state[4])
        
        next_state[6] = self.ref_spec_force[self.current_step + 1]
        next_state[7] = self.ref_angular_vel[self.current_step + 1]

        # define reward for current state, 这里只使用加速度的差值, 以确保加速度一定verfolgen
        weight_spec_force = 0.9
        weight_ang_vel = 0.9
        weight_action = np.array([0.1, 0.1])

        reward_x_lim = 0
        if (abs(next_state[2]) > 5.0):
            reward_x_lim = -10000

        reward = 1 \
            - weight_spec_force * abs(self.current_state[5] - self.ref_spec_force[self.current_step - 1]) \
            + reward_x_lim \
            - np.dot(weight_action, abs(action)) \
            - weight_ang_vel * abs(self.current_state[3] - self.ref_angular_vel[self.current_step - 1]) \
            

        self.current_state = next_state
        if self.current_step >= len(self.timesteps):
            self.done = True
        return next_state, reward, self.done
    
    def compute_random_reference(self):
        # start_time = round(random.uniform(0.0, 7.9), 2)
        start_time = 2.6
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



class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        # self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        # if self.is_vector_env:
        #     self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        # else:
        #     self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.obs_rms = RunningMeanStd(shape=(8,))
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, done = self.env.step(action)
        if self.is_vector_env:
            obs_norm = self.normalize(obs)
        else:
            obs_norm = self.normalize(np.array([obs]))[0]
        return obs_norm, obs, rews, done

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        ref_f, ref_ang_vel, obs = self.env.reset(**kwargs)

        if self.is_vector_env:
            return ref_f, ref_ang_vel, self.normalize(obs), obs
        else:
            return ref_f, ref_ang_vel, self.normalize(np.array([obs]))[0], obs

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

