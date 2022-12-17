import numpy as np
from typing import NamedTuple


class Experience(NamedTuple):
    episode: int
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool
