from itertools import cycle

import torch
from torch.autograd import Variable

from utils import generate_label, log_sum_exp


# Shorthand cross_entropy till fix in next version of PyTorch
def cross_entropy(logits, y):
    return -torch.sum(y * torch.log(logits + 1e-8), dim=1)


class SemiSupervisedTrainer:
    def __init__(self, model, objective, optimizer, logger=print, cuda=False, args=None):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        if cuda: self.model.cuda()

        if args is None:
            self.args = {"iw": 1, "eq": 1, "temperature": 1}
        else:
            self.args = args

    def _calculate_loss(self, x, y=None):
        pass

    def train(self, labelled, unlabelled, n_epochs):
        pass


class VAETrainer(SemiSupervisedTrainer):
    def __init__(self, model, objective, optimizer, logger=print, cuda=False, args=None):
        super(VAETrainer, self).__init__(model, objective, optimizer, logger, cuda, args)

    def _calculate_loss(self, x, y=None):
        """
        Given a unsupervised problem.
        :param x: Features
        :returns ELBO
        """
        x = Variable(x)

        # Increase sampling dimension for importance weighting
        x = x.repeat(self.args["eq"] * self.args["iw"], 1)

        if self.cuda:
            x = x.cuda()

        # Compute lower bound (the same as -L)
        reconstruction, z = self.model(x)
        log_likelihood, kl_divergence = self.objective(reconstruction, x, z)

        ELBO = log_likelihood - self.args["temperature"] * kl_divergence

        # Inner mean over IW samples and outer mean of E_q samples
        ELBO = ELBO.view(-1, self.args["eq"], self.args["iw"], 1)
        ELBO = torch.mean(log_sum_exp(ELBO, dim=2, sum_op=torch.mean), dim=1)

        loss = torch.mean(ELBO)

        return loss, log_likelihood, kl_divergence

    def train(self, labelled, unlabelled, n_epochs):
        """
        Trains a VAE model based on some data.
        :param labelled: Labelled data loader (unused)
        :param unlabelled: Unlabelled data loader
        :param n_epochs: Number of epochs
        """
        for epoch in range(n_epochs):
            # Go through all unlabelled data
            for (u, _) in unlabelled:
                U, log_likelihood, kl_divergence = self._calculate_loss(u)

                U.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger({"epoch": epoch, "NLL": torch.mean(log_likelihood).data[0], "KL": torch.mean(kl_divergence).data[0]})


class DGMTrainer(VAETrainer):
    def __init__(self, model, objective, optimizer, logger=print, cuda=False, args=None):
        super(DGMTrainer, self).__init__(model, objective, optimizer, logger, cuda, args)

    def _calculate_loss(self, x, y=None):
        """
        Given a semi-supervised problem (x, y) pair where y
        is only occasionally observed, calculates the
        associated loss.
        :param x: Features
        :param y: Labels (optional)
        :returns L_alpha if labelled, U if unlabelled.
        """
        is_unlabelled = True if y is None else False

        # Increase sampling dimension for importance weighting
        x = x.repeat(self.args["eq"] * self.args["iw"], 1)

        x = Variable(x)

        if self.cuda:
            x = x.cuda()

        logits = self.model(x)

        # If the data is unlabelled, sum over all classes
        if is_unlabelled:
            [batch_size, *_] = x.size()
            x = x.repeat(self.model.y_dim, 1)
            y = torch.cat([generate_label(batch_size, i, self.model.y_dim) for i in range(self.model.y_dim)])
        else:
            # Increase sampling dimension
            y = y.repeat(self.args["eq"] * self.args["iw"], 1)

        y = Variable(y.type(torch.FloatTensor))

        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        # Compute lower bound (the same as -L)
        reconstruction, _, z = self.model(x, y)
        log_likelihood, kl_divergence, log_prior_y = self.objective(reconstruction, x, y, z)

        # - L(x, y)
        ELBO = log_likelihood + log_prior_y + self.args["temperature"] * kl_divergence

        # Inner mean over IW samples and outer mean of E_q samples
        ELBO = ELBO.view(-1, self.args["eq"], self.args["iw"], 1)
        ELBO = torch.mean(log_sum_exp(ELBO, dim=2, sum_op=torch.mean), dim=1)

        if is_unlabelled:
            # In the unlabelled case calculate the entropy H and return U(x)
            ELBO = ELBO.view(logits.size())
            loss = torch.sum(torch.mul(logits, ELBO - torch.log(logits)), -1)
            loss = -torch.mean(loss)
        else:
            # In the case of labels add cross entropy and return L_alpha
            loss = ELBO + self.model.alpha * -cross_entropy(logits, y)
            loss = -torch.mean(loss)

        return loss, log_likelihood, kl_divergence

    def train(self, labelled, unlabelled, n_epochs):
        """
        Trains a DGM model based on some data.
        :param labelled: Labelled data loader
        :param unlabelled: Unlabelled data loader
        :param n_epochs: Number of epochs
        """
        for epoch in range(n_epochs):
            for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
                U, *_ = self._calculate_loss(u)
                L, *_ = self._calculate_loss(x, y)

                J = L + U

                J.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger({"epoch": epoch, "Unlabelled loss": U.data[0], "Labelled loss": L.data[0]})
