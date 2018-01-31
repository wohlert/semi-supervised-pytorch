from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F

from utils import log_sum_exp, enumerate_discrete
from .distributions import log_standard_categorical

class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def repeat(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(-1, self.mc, self.iw)
        elbo = torch.mean(log_sum_exp(elbo, dim=2, sum_op=torch.mean), dim=1)
        return elbo


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
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


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Increase sampling dimension
        xs = self.sampler.repeat(x)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)
        else:
            ys = self.sampler.repeat(y)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)
        if not is_labelled:
            logits = self.model.classify(x)

            L = L.view_as(logits.t()).t()

            # Calculate entropy H(q(y|x)) and sum over all labels
            H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
            L = torch.sum(torch.mul(logits, L), dim=-1)

            # Equivalent to -U(x)
            U = L + H
            return torch.mean(U)

        return torch.mean(L)