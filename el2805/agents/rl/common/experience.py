import numpy as np
from typing import NamedTuple


class Experience(NamedTuple):
    episode: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
