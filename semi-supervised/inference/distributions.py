import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution evaluated at x.
    :param x: point to evaluated
    :return: log pdf(x)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log pdf(x)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=1)


def log_standard_categorical(x):
    """
    Calculates the entropy between a categorical vector
    and a standard (uniform) categorical distribution.

    :param x: (torch.autograd.Variable)
    :return: (torch.autograd.Variable) entropy
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(x), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(torch.mul(x, torch.log(prior + 1e-8)), dim=1)

    # Alternatively
    # [_, n] = x.size()
    # cross_entropy = -math.log(1/n) * torch.sum(x, dim=1)

    return cross_entropy