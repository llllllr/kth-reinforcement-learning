import numpy as np
from typing import NamedTuple


class Experience(NamedTuple):
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool


def random_decide(rng, probability):
    return rng.binomial(n=1, p=probability) == 1


def running_average(data, window_length):
    if len(data) >= window_length:
        averages = np.copy(data)
        averages[window_length - 1:] = np.convolve(data, np.ones((window_length,)) / window_length, mode='valid')
    else:
        averages = np.zeros_like(data)
    return averages
