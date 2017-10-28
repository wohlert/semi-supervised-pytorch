import torch
from torch import nn
import torch.nn.functional as F


def kl_divergence_normal(mu, log_var):
    r"""
    Returns the KL-divergence between an [isotropic] normal
    distribution with parameters mu and log_var and a
    standard normal ~ KL(N(mu, var) || N(0, I))
    :param mu: (torch.Tensor) mean of distribution
    :param log_var: (torch.Tensor) log variance of distribution
    :return: (torch.Tensor) KL(N(mu, var) || N(0, I))
    """
    return 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1.)


def uniform_prior(x):
    [_, n] = x.size()

    prior = (1. / n) * torch.ones(x.size())
    prior = F.softmax(prior)

    return -F.cross_entropy(prior, x)


class VariationalInference(nn.Module):
    """
    Variational autoencoder loss function
    as described by (Kingma 2013).

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
        log_likelihood = self.reconstruction(r, x, size_average=False)
        kl_divergence = self.kl_div(mu, log_var)

        return log_likelihood + kl_divergence


class VariationalInferenceWithLabels(VariationalInference):
    """
    Loss function for labelled data points
    as described by (Kingma 2014).

    :param prior_y (function) function to calculate the
        entropy between y and some discrete categorical
        distribution.
    """
    def __init__(self, reconstruction, kl_div, prior_y):
        super(VariationalInferenceWithLabels, self).__init__(reconstruction, kl_div)
        self.label_prior = prior_y

    def forward(self, r, x, y, z, z_mu, z_log_var):
        """
        Compute loss.
        :param r: reconstruction
        :param x: original
        :param y: label
        :param z: latent variable
        :param z_mu: mean of z
        :param z_log_var: log variance of z
        :return: loss
        """
        log_prior_y = self.prior_y(y)
        log_likelihood = self.reconstruction(r, x)
        kl_divergence = self.kl_div(z_mu, z_log_var)

        return log_likelihood + kl_divergence + log_prior_y