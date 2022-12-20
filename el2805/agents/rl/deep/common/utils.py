import torch


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.has_mps:
    #     device = "mps"
    else:
        device = "cpu"
    return device


def normal_pdf(x, mean, var):
    pdf = 1 / torch.sqrt(2 * torch.pi * var) * torch.exp(-1/2 * (x - mean)**2 / var)
    return pdf
