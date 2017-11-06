import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class StochasticGaussian(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, input_dim, z_dim):
        super(StochasticGaussian, self).__init__()
        self.mu = nn.Linear(input_dim, z_dim)
        self.log_var = nn.Linear(input_dim, z_dim)

    def forward(self, x):
        """
        Performs the reparametrisation trick as described
        by (Kingma 2013) in order to backpropagate through
        stochastic units.
        :param mu: (torch.autograd.Variable) mean of normal distribution
        :param log_var: (torch.autograd.Variable) log variance of normal distribution
        :return: (torch.autograd.Variable) a sample from the distribution
        """
        mu = self.mu(x)
        log_var = self.log_var(x)

        epsilon = Variable(torch.randn(*mu.size()))
        std = torch.exp(0.5 * log_var)

        z = mu + std * epsilon

        return z, mu, log_var


class StochasticMultinomial(nn.Module):
    """

    """
    def __init__(self, input_dim, output_dim):
        super(StochasticMultinomial, self).__init__()

    def forward(self, x):
        pi = x
        x = F.softmax(pi)
        return x