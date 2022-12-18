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
    distribution = torch.distributions.Normal(loc=mean, scale=torch.sqrt(var))
    log_pdf = distribution.log_prob(x)
    pdf = torch.exp(log_pdf)
    return pdf
