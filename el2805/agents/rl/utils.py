import numpy as np
from typing import NamedTuple


def get_epsilon(
        epsilon: float | str,
        *,
        episode: int | None = None,
        epsilon_max: float | None = None,
        epsilon_min: float | None = None,
        epsilon_decay_duration: int | None = None,
        delta: float | None = None
) -> float:
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = epsilon
    elif epsilon == "delta":
        assert delta is not None
        epsilon = 1 / (episode ** delta)
    elif epsilon == "linear":
        assert epsilon_max is not None and epsilon_min is not None and epsilon_decay_duration is not None
        epsilon = max(
            epsilon_min,
            epsilon_max - (epsilon_max - epsilon_min) * (episode - 1) / (epsilon_decay_duration - 1)
        )
    elif epsilon == "exponential":
        assert epsilon_max is not None and epsilon_min is not None and epsilon_decay_duration is not None
        epsilon = max(
            epsilon_min,
            epsilon_max * (epsilon_min / epsilon_max) ** ((episode - 1) / (epsilon_decay_duration - 1))
        )
    else:
        raise NotImplementedError
    return epsilon


class Experience(NamedTuple):
    episode: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
