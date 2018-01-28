from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F

from utils import enumerate_choices, log_sum_exp, generate_label
from .distributions import log_standard_categorical


class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        self.mc = mc
        self.iw = iw

    def repeat(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(-1, self.mc, self.iw, 1)
        elbo = torch.mean(log_sum_exp(elbo, dim=2, sum_op=torch.mean), dim=1)
        return elbo


class DeterministicWarmup(object):
    """
    Linear deterministic warmup as described in both
    [Maaløe, Sønderby].
    """
    def __init__(self, init=0, end=1, inc=0.01):
        self.t = init
        self.end = end
        self.inc = inc

    def __iter__(self):
        return self

    def __next__(self):
        if self.t > self.end:
            return self.end
        else:
            self.t += self.inc
            return self.t


# Shorthand cross_entropy till fix in next version of PyTorch
def cross_entropy(logits, y):
    return -torch.sum(y * torch.log(logits + 1e-6), dim=1)


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, likelihood=F.binary_cross_entropy, is_supervised=True, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model:
        :param likelihood:
        :param beta: warmup/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta
        self.is_supervised = is_supervised

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Increase sampling dimension
        x = self.sampler.repeat(x)

        # Enumerate choices of label
        if self.is_supervised:
            logits = self.model.classifier(x)

            if is_labelled:
                y = self.sampler.repeat(y)
            else:
                y = enumerate_choices(x, self.model.y_dim)
                x = x.repeat(self.model.y_dim, 1)

            r, kl_divergence = self.model(x, y)
        else:
            r, kl_divergence = self.model(x)

        likelihood = -self.likelihood(r, x)

        if self.is_supervised:
            likelihood += -log_standard_categorical(y)

        elbo = likelihood - next(self.beta) * kl_divergence
        elbo = self.sampler(elbo)

        if self.is_supervised:
            if is_labelled:
                elbo = elbo + self.model.alpha * -cross_entropy(logits, y)
            else:
                elbo = elbo.view(logits.size())
                elbo = torch.sum(torch.mul(logits, elbo - torch.log(logits + 1e-8)), dim=1)

        return -torch.mean(elbo)