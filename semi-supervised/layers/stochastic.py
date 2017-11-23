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
        :param x: (torch.autograd.Variable) input tensor
        :return: (torch.autograd.Variable) a sample from the distribution
        """
        mu = self.mu(x)
        log_var = self.log_var(x)

        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if x.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul_(std, epsilon)

        if not self.training:
            z = mu

        return z, mu, log_var
