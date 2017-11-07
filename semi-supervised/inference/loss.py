import math
import torch
from torch import nn
import torch.nn.functional as F


EPSILON = 1e-7


def kl_divergence_normal(mu, log_var):
    """
    Returns the KL-divergence between an [isotropic] normal
    distribution with parameters mu and log_var and a
    standard normal, equivalent to KL(N(mu, var) || N(0, I))

    :param mu: (torch.Tensor) mean of distribution
    :param log_var: (torch.Tensor) log variance of distribution
    :return: (torch.Tensor) KL(N(mu, var) || N(0, I))
    """
    return 0.5 * (1. + log_var - mu**2 - torch.exp(log_var))


def discrete_uniform_prior(x):
    """
    Calculates the cross entropy between a categorical
    vector and a uniform prior.

    :param x: (torch.autograd.Variable)
    :return: (torch.autograd.Variable) entropy
    """
    [batch_size, n] = x.size()

    # Uniform prior over y
    prior = (1. / n) * torch.ones(batch_size, n)
    prior = F.softmax(prior)

    cross_entropy = -torch.sum(x * torch.log(prior + EPSILON), dim=1)

    return -cross_entropy


class VariationalInference(nn.Module):
    """
    Variational autoencoder loss function
    as described in (Kingma, 2013).

    :param reconstruction: (function) Autoencoder reconstruction loss
    :param kl_div: (function) KL-divergence function
    """
    def __init__(self, reconstruction, kl_div):
        super(VariationalInference, self).__init__()
        self.reconstruction = reconstruction
        self.kl_div = kl_div

    def forward(self, r, x, mu, log_var):
        """
        Compute loss.
        :param r: (torch.Tensor) reconstruction
        :param x: (torch.Tensor) original
        :param mu: (torch.Tensor) mean of z
        :param log_var: (torch.Tensor) log variance of z
        :return: (torch.Tensor) loss
        """
        log_likelihood = self.reconstruction(r, x)
        kl_divergence = torch.sum(self.kl_div(mu, log_var))

        return log_likelihood - kl_divergence


class VariationalInferenceWithLabels(VariationalInference):
    """
    Loss function for labelled data points
    as described in (Kingma, 2014).

    :param prior_y (function) function to calculate the
        entropy between y and some discrete categorical
        distribution.
    """
    def __init__(self, reconstruction, kl_div, prior_y):
        super(VariationalInferenceWithLabels, self).__init__(reconstruction, kl_div)
        self.prior_y = prior_y

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
        log_prior_y = self.prior_y(y)
        log_likelihood = self.reconstruction(r, x)
        kl_divergence = [torch.sum(self.kl_div(mu, log_var), dim=-1) for _, mu, log_var in latent]

        return log_likelihood + log_prior_y + sum(kl_divergence)