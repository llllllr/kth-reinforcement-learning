import gym
import numpy as np
import torch
from collections import defaultdict
from el2805.agents.rl.rl_agent import RLAgent
from el2805.agents.rl.utils import Experience, MultiLayerPerceptron, normal_pdf
import MotionSimulationPlatform

class PPO(RLAgent):
    """PPO (Proximal Policy Optimization) agent."""

    def __init__(
            self,
            *,
            environment: gym.Env,
            # environment: MotionSimulationPlatform(total_time=10, dt=0.01), 
            discount: float,
            n_epochs_per_step: int,
            epsilon: float,
            critic_learning_rate: float,
            critic_hidden_layer_sizes: list[int],
            critic_hidden_layer_activation: str,

            actor_learning_rate: float,
            actor_shared_hidden_layer_sizes: list[int],
            actor_mean_hidden_layer_sizes: list[int],
            actor_var_hidden_layer_sizes: list[int],
            actor_hidden_layer_activation: str,
            gradient_max_norm: float,

            device: str,
            seed: int | None = None
    ):
        super().__init__(environment=environment, seed=seed)
        self.discount = discount
        self.n_epochs_per_step = n_epochs_per_step
        self.epsilon = epsilon
        self.critic_learning_rate = critic_learning_rate
        self.critic_hidden_layer_sizes = critic_hidden_layer_sizes
        self.critic_hidden_layer_activation = critic_hidden_layer_activation
        self.actor_learning_rate = actor_learning_rate
        self.actor_shared_hidden_layer_sizes = actor_shared_hidden_layer_sizes
        self.actor_mean_hidden_layer_sizes = actor_mean_hidden_layer_sizes
        self.actor_var_hidden_layer_sizes = actor_var_hidden_layer_sizes
        self.actor_hidden_layer_activation = actor_hidden_layer_activation
        self.gradient_max_norm = gradient_max_norm
        self.device = device

# conti state & conti action 
        state_dim = 6
        self._action_dim = 2

        # assert isinstance(environment.observation_space, gym.spaces.Box)
        # state_dim = len(environment.observation_space.low)

        # assert isinstance(environment.action_space, gym.spaces.Box)
        # self._action_dim = len(environment.action_space.low)


# input is the current real state(8 dimension), N samples in batch. 
# Output is the single value: V(s)
        self.critic = PPOCritic(
            state_dim=state_dim,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
            hidden_layer_activation=self.critic_hidden_layer_activation
        ).double().to(self.device)

        self.actor = PPOActor(
            state_dim=state_dim,
            action_dim=self._action_dim,
            shared_hidden_layer_sizes=self.actor_shared_hidden_layer_sizes,
            mean_hidden_layer_sizes=self.actor_mean_hidden_layer_sizes,
            var_hidden_layer_sizes=self.actor_var_hidden_layer_sizes,
            hidden_layer_activation=self.actor_hidden_layer_activation
        ).double().to(self.device)

        self._critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self._actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self._episodic_buffer = []

    def update(self) -> dict:
        stats = defaultdict(list)

        # Skip update if the episode has not terminated, 
        if not self._episodic_buffer[-1].done:
            return stats

        # Unpack experiences
        n = len(self._episodic_buffer)
        # torch.size = ([1000])
        rewards = torch.as_tensor(
            data=np.asarray([e.reward for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )
        
        # torch.size = ([1000, 6])
        states = torch.as_tensor(
            data=np.asarray([e.state for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )

        max_state, idx_max  = torch.max(states, 0)
        min_state, idx_min = torch.min(states, 0)
        mean_state = torch.mean(states, 0)
        var_state = torch.var(states, 0)
        # print(f"max value in 1000 experiences: {max_state}, min of 1000 states: {min_state}")
        # print(f"mean value in 1000 experiences: {mean_state}, var of 1000 states: {var_state}")
        # actions: torch.size = ([1000, 2])
        actions = torch.as_tensor(
            data=np.asarray([e.action for e in self._episodic_buffer]),
            dtype=torch.float64,
            device=self.device
        )
        print(f"The lastet state : {states[-5, :]}")
        print(f"The lastet action : {actions[-5, :]}")

        # Compute Monte Carlo targets, 
        g = []
        discounted_reward = 0
        for r in reversed(rewards):  # for every collocted reward, we reverse the raward list, to get RETURN from s_T...s_0
            discounted_reward = r + self.discount * discounted_reward
            g.append(discounted_reward)
        g = torch.as_tensor(
            np.asarray(g[::-1]),    # reverse the G-list, from s0 to s_T
            dtype=torch.float64,
            device=self.device
        ) 
        # print(f"the length of monto-carlo-target of an episode: {g.shape}")  # size = 1000, 是一个list
        # print(f"the length of rewards of an episode: {rewards.shape}")
        # print(f"the length of states of an episode: {states.shape}")
        # print(f"the length of actions of an episode: {actions.shape}")

        g = g.reshape(-1)  # size = (1000, )
        assert g.shape == (n, )

        with torch.no_grad():  # no_grad, because we only compute the forward-result, no need for gradient-compute
            # Compute advantages values for each state in episode
            self.critic.eval()
            v = self.critic(states)  # states的size: torch.Size([1000, 6])
            assert v.shape == (n, 1)  # result size should be (1000, 1)
            v = v.reshape(-1)  # reshape to 1 dimension tensor (1000, )
            psi = g - v  

            # Compute action likelihood from old policy, for each abserved pair (s_i, a_i) , i = 0...T 
            self.actor.eval()
            pi_old = self._compute_actions_likelihood(states, actions)

        for _ in range(self.n_epochs_per_step):
            # Forward pass by critic
            self.critic.train()
            v = self.critic(states)
            assert v.shape == (n, 1)
            v = v.reshape(-1) # become to (1000, )
            critic_loss = torch.nn.functional.mse_loss(g, v) # 是标量

            # Backward pass by critic, reset the gradiant 
            self._critic_optimizer.zero_grad()
            critic_loss.backward()  # compute gradiant w.r.t critic-loss
            # clip the gradient of the NN
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_max_norm)
            self._critic_optimizer.step()  # update grad w.r.t loss

            # Forward pass by actor
            # what is the probability density value of given states and actions? 
            # pi.shape = (1000, )
            self.actor.train()
            pi = self._compute_actions_likelihood(states, actions)
            r = pi / pi_old # r就是 现在从actor得到的两个分布中抽中当前action的可能性/旧的actor分布抽中的可能性, 其中pi_old是常数
            assert r.shape == (n,)
            r_clipped = r.clip(min=1-self.epsilon, max=1 + self.epsilon) # 剪裁掉超出范围的一部部分, 那么相当于有些r_clipped是一个常数, 不是通过圣经网络计算得到
            assert r_clipped.shape == (n,)
            actor_loss = - torch.minimum(r * psi, r_clipped * psi).mean() # loss如果没被剪裁就是 真r(含actor网络参数)*state对应的advantage_value, 要么就是一个常数*state对应的advantage_value
            # 结果是一个标量!!!! 本来所有的都是(1000, ), mean之后变成了一个值

            # Backward pass by actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()   # ? actor_loss经过了一些变化才变成actor_loss, 为什么可以直接backward? --> 因为auto-grad
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_max_norm)
            self._actor_optimizer.step()

            # Save stats, for each episode, for each epoch(update) of current-episode, save the loss-value
            # print(critic_loss.item())
            # print(actor_loss.item())  # for every epoches, obtain one critic-loss, one actor-loss
            stats["critic_loss"].append(critic_loss.item())   # 一个标量, 每个epoch得到一个标量, 每个episode得到m个向量
            stats["actor_loss"].append(actor_loss.item())     # 一个标量

        # Clear episodic buffer for new episode, List
        self._episodic_buffer = []

        return stats

    def record_experience(self, experience: Experience) -> None:
        self._episodic_buffer.append(experience)

# class Experience(NamedTuple): episode: int, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,done: bool

    def compute_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        
        with torch.no_grad():
            state = torch.as_tensor(
                data=state.reshape((1,) + state.shape),    # reshape to (1, )+(6, ) = (1, 6)
                dtype=torch.float64,
                device=self.device
            )
            # mean.shape = [1, 2], var.shape = [1, 2]
            self.actor.eval()
            mean, var = self.actor(state)
            # 
            mean, var = mean.reshape(-1), var.reshape(-1)
            # for every action-dim, we should choose a action-value, we shoose it from a normal disturbution
            action = torch.normal(mean, torch.sqrt(var))
            action = action.numpy()
        return action

    def _compute_actions_likelihood(self, states, actions):
        assert len(states) == len(actions)
        n = len(states)
        # compute the action disturbution of current actor
        mean, var = self.actor(states)
        # compute the probability density value of current real action, how possible I take this action under current actor-policy?
        # def normal_pdf(x, mean, var):
        #     pdf = 1 / torch.sqrt(2 * torch.pi * var) * torch.exp(-1/2 * (x - mean)**2 / var)
        #     return pdf
        # actions(1000, 2), mean/var(1000,2) , pi = (1000, 2).prod = (1000, 1). 
        pi = normal_pdf(actions, mean, var).prod(dim=1)     # assumption: independent action dimensions
        assert mean.shape == (n, self._action_dim) and var.shape == (n, self._action_dim) and pi.shape == (n,)

        return pi


class PPOCritic(MultiLayerPerceptron):
    def __init__(
            self,
            *,
            state_dim: int,
            hidden_layer_sizes: list[int],
            hidden_layer_activation: str,
    ):
        super().__init__(
            input_size=state_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_activation=hidden_layer_activation,
            output_size=1, # only one output, indicate the V(s) value
            include_top=True # here indicate that there is no activate-funcion in output-layer
        )
        for m in self._hidden_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                m.bias.data.fill_(0.01)


    def forward(self, x):
        x = x.to(torch.float64)
        return super().forward(x)


class PPOActor(torch.nn.Module):
    def __init__(
            self,
            *,
            state_dim: int,
            action_dim: int,
            shared_hidden_layer_sizes: list[int],
            mean_hidden_layer_sizes: list[int],
            var_hidden_layer_sizes: list[int],
            hidden_layer_activation: str
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_hidden_layer_sizes = shared_hidden_layer_sizes

        self.mean_hidden_layer_sizes = mean_hidden_layer_sizes
        self.var_hidden_layer_sizes = var_hidden_layer_sizes
        self.hidden_layer_activation = hidden_layer_activation

# first multiLayer are shared for both output. Input is the states(6 dimensions)
        self._shared_layers = MultiLayerPerceptron(
            input_size=self.state_dim,
            hidden_layer_sizes=self.shared_hidden_layer_sizes,
            # hidden_layer_activation=self.hidden_layer_activation,
            hidden_layer_activation="relu",
            include_top=False
        )
        for m in self._shared_layers._hidden_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                m.bias.data.fill_(0.01)
        # input-size for further NN
        input_size = self.shared_hidden_layer_sizes[-1]

        # The randomized policy is modeled as a multi-variate Gaussian distribution
        # Key assumption: independent action dimensions
        # Therefore, the randomized policy is parametrized with mean and variance of each action dimension
# output_size should be dimension of action, here 3, indicate the mean-value of distribution of auch action dimension
        # range  [-1, 1]
        self._mean_head = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.mean_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,
            output_layer_activation="tanh",
            include_top=True
        )
        for m in self._mean_head._hidden_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        # should be dimension of action, here 3, indicate the varianc-value of distribution of auch action dimension
        # range [0, 1]
        self._var_head = MultiLayerPerceptron(
            input_size=input_size,
            hidden_layer_sizes=self.var_hidden_layer_sizes,
            hidden_layer_activation=self.hidden_layer_activation,
            output_size=self.action_dim,    # assumption: independent action dimensions
            output_layer_activation="sigmoid",
            include_top=True
        )
        for m in self._var_head._hidden_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.to(torch.float64)
        x = self._shared_layers(x)
        mean = self._mean_head(x)
        var = self._var_head(x)
        return mean, var
