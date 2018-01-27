import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class GaussianSample(nn.Module):
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

        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if x.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul_(std, epsilon)

        # When evaluating get the MAP estimate
        if not self.training:
            z = mu

        return z, mu, log_var


class GaussianMerge(nn.Module):
    def forward(self, mu1, mu2, logvar1, logvar2):
        precision1, precision2 = (1/logvar1, 1/logvar2)

        mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)
        logvar = 1 / (precision1 + precision2) #TODO: check that this should be log

        return mu, logvar