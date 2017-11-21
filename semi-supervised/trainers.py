from itertools import cycle

import torch
from torch.autograd import Variable

from utils import generate_label


# Shorthand cross_entropy till fix in next version of PyTorch
def cross_entropy(logits, y):
    return -torch.sum(y * torch.log(logits + 1e-8), dim=1)


class SemiSupervisedTrainer:
    def __init__(self, model, objective, optimizer, logger=print, cuda=False):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        if cuda: model.cuda()

    def _calculate_loss(self, x, y=None):
        pass

    def train(self, labelled, unlabelled, n_epochs):
        pass


class VAETrainer(SemiSupervisedTrainer):
    def __init__(self, model, objective, optimizer, logger=print, cuda=False):
        super(VAETrainer, self).__init__(model, objective, optimizer, logger, cuda)

    def _calculate_loss(self, x, y=None):
        """
        Given a unsupervised problem problem.
        :param x: Features
        :returns ELBO
        """
        x = Variable(x)

        if self.cuda:
            x = x.cuda()

        # Compute lower bound (the same as -L)
        reconstruction, z = self.model(x)
        elbo = self.objective(reconstruction, x, z)

        return torch.mean(elbo)

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
                u = self._calculate_loss(u)

                self.optimizer.zero_grad()
                u.backward()
                self.optimizer.step()

            self.logger(epoch, {"epoch": epoch, "Loss": u.data[0]})


class DGMTrainer(VAETrainer):
    def __init__(self, model, objective, optimizer, logger=print, cuda=False):
        super(DGMTrainer, self).__init__(model, objective, optimizer, logger, cuda)

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
        x = x.repeat(self.objective.iw * self.objective.eq, 1)

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
            y = y.repeat(self.objective.iw * self.objective.eq, 1)

        y = Variable(y.type(torch.FloatTensor))

        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        # Compute lower bound (the same as -L)
        reconstruction, _, z = self.model(x, y)
        elbo = self.objective(reconstruction, x, y, z)

        # In the unlabelled case calculate the entropy H and return U
        if is_unlabelled:
            elbo = elbo.view(logits.size())
            loss = torch.sum(torch.mul(logits, elbo - torch.log(logits)), -1)
            loss = -torch.mean(loss)

        # In the case of labels add cross entropy and return L_alpha
        else:
            loss = elbo + self.model.beta * -cross_entropy(logits, y)
            loss = -torch.mean(loss)

        return loss

    def train(self, labelled, unlabelled, n_epochs):
        """
        Trains a DGM model based on some data.
        :param labelled: Labelled data loader
        :param unlabelled: Unlabelled data loader
        :param n_epochs: Number of epochs
        """
        for epoch in range(n_epochs):
            for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
                u = self._calculate_loss(u)
                l = self._calculate_loss(x, y)

                j = l + u

                self.optimizer.zero_grad()
                j.backward()
                self.optimizer.step()

            self.logger(epoch, {"epoch": epoch, "Unlabelled loss": u.data[0], "Labelled loss": l.data[0]})
