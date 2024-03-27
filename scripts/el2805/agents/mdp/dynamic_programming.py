import numpy as np
from typing import Any
from el2805.agents.mdp.mdp_agent import MDPAgent
from el2805.envs import TabularMDP


class DynamicProgramming(MDPAgent):
    def __init__(self, environment: TabularMDP):
        super().__init__(environment=environment)
        assert self.environment.finite_horizon()

    def solve(self) -> None:
        n_states = len(self.environment.states) # 一共有几个states, 42个
        u = np.zeros(n_states)  # 每个state对应的最优action是什么?
        # for every time_step, all states are possible to be in that state, array(20 steps, 42 states)
        self.policy = np.zeros((self.environment.horizon, n_states), dtype=np.int32)  # optimal policy (non-stationary)

        for t in range(self.environment.horizon - 1, -1, -1): # 从t=19 , t = 18...开始计算
            last_time_step = (t == self.environment.horizon - 1)              # terminal case? t 是不是等于19? 
            u_next = u.copy() if not last_time_step else None       # u*_{t+1}

            for s, state in enumerate(self.environment.states): # 对每个位置state(i,j), 以及它对应的index进行遍历
                # Q_t(s,a) for each a in A_s
                valid_actions = self.environment.valid_actions(state) # 获取当前state 的valid_actions
                # 如果是第19步,计算每个action对应的 reward, 只包含当前这一步
                if last_time_step:
                    q = np.asarray([self.environment.reward(state, action, mean=True) for action in valid_actions])
                else:
                    # 如果不是最后一步, 那么包含下一步的value-function: q = r +  0.9* (transition proba一一对应乘以v(s') )
                    q = np.asarray([self.q(state, action, u_next) for action in valid_actions])

                # u*_t(s), u的每个state_index都对应一个最大q,相当于v_max(s), 每个状态对应的最大valuefunction值
                u[s] = max(q)

                # store optimal policy (non-stationary, optimal at this time step)
                a_best = q.argmax()     # index of best action for valid actions in this state, 并保存对应q最大时候的action值
                self.policy[t, s] = valid_actions[a_best]

    def compute_action(self, *, state: Any, time_step: int, **kwargs) -> int:
        _ = kwargs
        assert self.policy is not None
        # 对每个state, 对应的index:s, 获取对应的 time-step, index的action
        s = self.environment.state_index(state)
        action = self.policy[time_step, s]  # 从policy中读取最优action
        return action
