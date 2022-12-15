import numpy as np


def running_average(data, window_length: int = 50):
    if len(data) >= window_length:
        averages = np.copy(data)
        averages[window_length - 1:] = np.convolve(data, np.ones((window_length,)) / window_length, mode='valid')
    else:
        averages = np.zeros_like(data)
    return averages
