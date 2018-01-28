import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Stochastic(nn.Module):
    """
    Base stochastic layer.
    """
    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul_(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Performs the reparametrisation trick as described
        by (Kingma 2013) in order to backpropagate through
        stochastic units.
        :param x: (torch.autograd.Variable) input tensor
        :return: (torch.autograd.Variable) a sample from the distribution
        """
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var


class GaussianMerge(GaussianSample):
    """
    Precision weighted merging of two Gaussian
    distributions.
    Uses an GaussianSample layer to merge
    information from z into the given
    mean and log variance and produces
    a sample from this new distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianMerge, self).__init__(in_features, out_features)

    def forward(self, z, mu1, log_var1):
        # Calculate precision of each distribution
        # (inverse variance)
        log_var2 = F.softplus(self.log_var(z))
        precision1, precision2 = (1/torch.exp(log_var1), 1/torch.exp(log_var2))

        # Merge distributions into a single new
        # distribution
        mu2 = self.mu(z)
        mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)

        var = 1 / (precision1 + precision2)
        log_var = torch.log(var + 1e-8)

        return self.reparametrize(mu, log_var), mu, log_var