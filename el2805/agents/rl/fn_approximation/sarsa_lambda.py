import numpy as np
import gym
import pickle
from typing import Any
from el2805.agents.rl.rl_agent import QAgent
from el2805.bases import Basis, FourierBasis


class SarsaLambda(QAgent):
    def __init__(
            self,
            *,
            environment: gym.Env,
            discount: float,
            learning_rate: float | str,
            basis: Basis,
            trace_decay: float,
            momentum: float | None = None,
            clip: float | None = None,
            seed: int | None = None
    ):
        super().__init__(environment=environment, discount=discount, learning_rate=learning_rate, seed=seed)
        if not isinstance(basis, FourierBasis):
            raise NotImplementedError
        self.basis: FourierBasis = basis
        self.trace_decay = trace_decay
        self.momentum = momentum
        self.clip = clip

        self._w = np.zeros((self.basis.output_size, self.env.action_space.n))
        self._z = None
        self._v = None

    def q(self, state: Any, action: int) -> float:
        phi = self.basis(state)
        q = self._w[action].dot(phi)
        return q

    def update(self, *, state: Any, action: int, reward: float, next_state: Any, next_action: int, **kwargs) -> None:
        _ = kwargs
        phi = self.basis(state)
        q = self.q(state, action)
        q_next = self.q(next_state, next_action)
        delta = reward + self.discount * q_next - q     # TD error

        self._z *= self.discount * self.trace_decay
        self._z[a] += phi
        self._z = np.clip(self._z, -self.clip, self.clip)

        learning_rates = self.learning_rate / np.linalg.norm(self.basis.eta, axis=0)
        self._v = self.momentum * self._v + learning_rates * delta * self._z
        self._w += self.momentum * self._v + learning_rates * delta * self._z

    def reset(self):
        self._z = np.zeros(self._w.shape)
        self._v = np.zeros(self._w.shape)

    def save(self, filepath):
        with open(f"{filepath}.pkl", "wb") as file:
            content = {"W": self._w, "N": self.basis.eta}
            pickle.dump(content, file)
