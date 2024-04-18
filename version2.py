class MotionSimulationPlatform(gym.Env):
                  
    def __init__(self, total_time=10.0, dt=0.01):
        self.total_time = total_time
        self.dt = dt
        self._rng = None
        self.seed() 
        self.current_state = None

        # initial state
        self.timesteps =  np.arange(0, self.total_time, self.dt)

    def reset(self):
        # reset the env to a random initial state, type: nparray
        state = np.zeros(6) 
        state[1] = random.uniform(-2.0, 2.0)  # randomrize the initial vel of each episode
        state[2] = random.uniform(-2.0, 2.0)  # randomrize the initial pos of each episode
        state[4] = random.uniform(-math.pi/6, math.pi/6 )
        state[5] = state[0] - 9.81*math.sin(state[4])  
        # at t=0, the specific force on the pilot is known 
        self.current_state = state
        self.current_step = 0
        self.ref_specific_force, self.angular_velocities = self.compute_random_reference()
        self.done = False
        return state


    # next_state, reward, done, _ = self.environment.step(action), 
    def step(self, action):
        # version2 add tilt-coordinate: states : 5, actions : 2
        # states are [0] accel的变化, [1] vel, [2] position, ALONG-X-AXIS , 
        # [3] angular-velocity的变化, [4] angular-position, [5] specific force 
        # action: [0] change of the acceleration delte_a, [1] change of angular vel, delta_w
        next_state = np.zeros(6)
        next_state[0] = self.current_state[0] + action[0] 
        next_state[1] = self.current_state[1] + self.current_state[0] * self.dt
        next_state[2] = self.current_state[2] + self.current_state[1] * self.dt
        next_state[3] = self.current_state[3] + action[1]
        next_state[4] = self.current_state[4] + self.current_state[3] * self.dt
        next_state[5] = next_state[0] - 9.81*math.sin(next_state[4])

        # compute the reward, w.r.t the reference-accelaration & angular-velocity
        weight_specific_force = 100
        weight_angular = 10
        weight_posi = 0.1
        weight_action = np.array([0.1, 0.1])
        reward_lim_x = 0
        reward_lim_ang = 0
        if (self.current_state[2] > 20 or self.current_state[2] < -20):
            reward_lim_x = - 1e9
            self.done = True
        if (self.current_state[4] > math.pi/3 or self.current_state[4] < -math.pi/3):
            reward_lim_ang = -1e9
            self.done = True
        reward_early_stop = 0
        if (self.done == True and (self.current_step < 1000)):
            reward_early_stop = -3*1e9

        reward = 1 - weight_specific_force*abs(self.current_state[5] - self.ref_specific_force[self.current_step]) \
                    - np.dot(weight_action, abs(action)) \
                    - weight_angular*abs(self.current_state[3] - self.angular_velocities[self.current_step]) \
                    + reward_lim_x + reward_lim_ang 
    # - weight_posi*abs(self.current_state[2])

        self.current_step += 1
        self.current_state = next_state

        if self.current_step >= len(self.timesteps):
            self.done = True
        return next_state, reward, self.done
    

    def compute_random_reference(self):
        start_time = round(random.uniform(0.0, 7.9), 2)
        direction = random.choice([1, -1])
        ref_acceleration = 0
        ref_accelerations = np.zeros(len(self.timesteps))
        delta_accel = random.uniform(0.04, 0.06)
        for i, t in enumerate(self.timesteps):
            if t >= start_time and t < start_time + 0.5:
                ref_acceleration += delta_accel 
            elif t >= start_time + 0.5 and t < start_time + 1.5:
                ref_acceleration -= delta_accel
            elif t >= start_time + 1.5 and t < start_time + 2.0 :
                ref_acceleration += delta_accel
            ref_accelerations[i] = ref_acceleration  

        ref_angular_velocities =  np.zeros(len(self.timesteps))
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


        return ref_accelerations*direction, ref_angular_velocities * direction