import numpy as np


def running_average(data, window_length: int = 50):
    overlap_length = np.concatenate((np.arange(1, window_length), window_length * np.ones(len(data))))
    window = np.ones(window_length)
    averages = (np.convolve(data, window) / overlap_length)[:-(window_length-1)]
    assert len(averages) == len(data)
    return averages
