import numpy as np
import torch


def running_average(data, window_length: int = 50):
    overlap_length = np.concatenate((np.arange(1, window_length), window_length * np.ones(len(data))))
    window = np.ones(window_length)
    averages = (np.convolve(data, window) / overlap_length)[:-(window_length-1)]
    assert len(averages) == len(data)
    return averages


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.has_mps:
    #     device = "mps"
    else:
        device = "cpu"
    return device
