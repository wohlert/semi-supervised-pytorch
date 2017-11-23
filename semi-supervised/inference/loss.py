import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

EPSILON = 1e-7


def log_gaussian(x, mu, log_var):
    """
    Returns the log CDF of a normal distribution.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log cdf(x)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu) ** 2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=1)


def log_standard_gaussian(x):
    """
    Evaluates the log CDF of a standard normal distribution
    :param x: point to evaluated
    :return: log cdf(x)
    """
    return torch.sum(- 0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=1)


def log_standard_categorical(x):
    """
    Calculates the entropy between a categorical vector
    and a standard (uniform) categorical distribution.

    :param x: (torch.autograd.Variable)
    :return: (torch.autograd.Variable) entropy
    """
    [batch_size, n] = x.size()

    # Uniform prior over y
    prior = (1. / n) * torch.ones(batch_size, n)
    prior = Variable(prior)

    if x.is_cuda:
        prior = prior.cuda()

    prior = F.softmax(prior, dim=-1)

    cross_entropy = -torch.sum(x * torch.log(prior + EPSILON), dim=1)

    return -cross_entropy


class VariationalInference(nn.Module):
    """
    Variational autoencoder loss function
    as described in (Kingma, 2013).

    :param reconstruction: (function) Autoencoder reconstruction loss
    :param kl_div: (function) KL-divergence function
    """
    def __init__(self, reconstruction):
        super(VariationalInference, self).__init__()
        self.reconstruction = reconstruction

    def forward(self, r, x, latent):
        """
        Compute loss.
        :param r: (torch.Tensor) reconstruction
        :param x: (torch.Tensor) original
        :param mu: (torch.Tensor) mean of z
        :param log_var: (torch.Tensor) log variance of z
        :return: (torch.Tensor) loss
        """
        z, mu, log_var = latent
        log_likelihood = self.reconstruction(r, x)
        kl_divergence = log_standard_gaussian(z) - log_gaussian(z, mu, log_var)

        return log_likelihood, kl_divergence


class VariationalInferenceWithLabels(VariationalInference):
    """
    Loss function for labelled data points
    as described in (Kingma, 2014).
    """
    def __init__(self, reconstruction):
        super(VariationalInferenceWithLabels, self).__init__(reconstruction)

    def forward(self, r, x, y, latent):
        """
        Compute loss.
        :param r: reconstruction
        :param x: original
        :param y: label
        :param mu: mean of z
        :param log_var: log variance of z
        :return: loss
        """
        log_prior_y = log_standard_categorical(y)
        log_likelihood = self.reconstruction(r, x)
        kl_divergence = torch.cat([log_standard_gaussian(z) - log_gaussian(z, mu, log_var) for z, mu, log_var in latent])

        return log_likelihood, kl_divergence, log_prior_y