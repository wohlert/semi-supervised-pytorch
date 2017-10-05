"""
M2 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from vae import reparametrize


class Classifier(nn.Module):
    """
    Two layer classifier
    with softmax output.
    """
    def __init__(self, dims):
        super(Classifier, self).__init__()
        x, h, y = dims
        self.dense1 = nn.Linear(x, h)
        self.dense2 = nn.Linear(h, y)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        return F.softmax(self.dense2(x))


class Encoder(nn.Module):
    """
    Encodes x and y into a
    latent representation z.
    """
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        self.dense_x = nn.Linear(x_dim, h_dim)
        self.dense_y = nn.Linear(1, h_dim)
        self.dense   = nn.Linear(h_dim, 2*z_dim)

    def forward(self, x, y):
        x = self.dense_x(x)
        y = self.dense_y(y)
        z = self.dense(torch.add(x, y))
        return F.relu(z)


class Decoder(nn.Module):
    """
    Decodes latent representation z
    and label y into an original
    representation x.
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        self.dense_y = nn.Linear(1, z_dim)
        self.dense1  = nn.Linear(z_dim, h_dim)
        self.dense2  = nn.Linear(h_dim, x_dim)

    def forward(self, z, y):
        y = self.dense_y(y)
        x = F.relu(self.dense1(torch.add(z, y)))
        x = F.sigmoid(self.dense2(x))
        return x


class DeepGenerativeModel(nn.Module):
    """
    Deep generative model consisting of an
    encoder/decoder pair along with a classifier
    in order to perform semi-supervised learning.
    """
    def __init__(self, ratio, dims):
        """
        Initialise a new generative model
        :param ratio: ratio between labelled and unlabelled data
        :param dims: dimensions of x, y, z and hidden layer.
        """
        super(DeepGenerativeModel, self).__init__()
        self.alpha = 0.1
        self.beta = self.alpha * ratio

        [self.x_dim, self.y_dim, self.z_dim, self.h_dim] = dims

        self.encoder = Encoder([self.x_dim, self.h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim, self.h_dim, self.x_dim])
        self.classifier = Classifier([self.x_dim, self.h_dim, self.y_dim])

    def forward(self, x, y):
        # Classify the data point
        y_logits = self.classifier(x)

        # Add label and data and generate latent variable
        latent = self.encoder(x, y)
        z_mu, z_log_var = torch.chunk(latent, 2, dim=1)
        z = reparametrize(z_mu, z_log_var)

        # Reconstruct data point from latent data and label
        reconstruction = self.decoder(z, y)

        return reconstruction, y_logits, [z, z_mu, z_log_var]

    def generate(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param y: label
        :return: x
        """
        y = y.type(torch.FloatTensor)
        return self.decoder(z, y)


def separate(x, y, labels=(0,)):
    x, y = x.numpy(), y.numpy()
    x = np.vstack([x[y == i] for i in labels])
    y = np.hstack([y[y == i] for i in labels])

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y


class LabelledLoss(nn.Module):
    """
    Loss function for labelled data points
    as described by (Kingma 2014).
    """
    def __init__(self, n_labels):
        super(LabelledLoss, self).__init__()
        self.n_labels = n_labels

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
        [batch_size, *_] = y.size()

        # Uniform prior over y
        prior_y = (1. / self.n_labels) * torch.ones(batch_size, self.n_labels)
        prior_y = F.softmax(prior_y)
        log_prior_y = -F.cross_entropy(prior_y, y, size_average=False)

        # Binary cross entropy as log likelihood of r
        log_likelihood = -F.binary_cross_entropy(r, x, size_average=False)

        # Gaussian variational distribution over z
        log_post_z = -0.5 * torch.sum(z_log_var - z_mu**2 - torch.exp(z_log_var) + 1)

        return torch.mean(log_likelihood + log_prior_y - log_post_z)


def generate_label(batch_size, label):
    """
    Generates a `torch.Tensor` of size batch_size x 1 of
    the given label.

    Example: generate_label(5, 1) #=> torch.Tensor([[1, 1, 1, 1, 1]])
    :param batch_size: number of labels
    :param label: label to generate
    """
    return (torch.ones(batch_size, 1) * label).type(torch.LongTensor)


def train_dgm(model, dataloader, labelled, optimizer, objective, labels=[0], epochs=100):
    """
    Trains a deep generative model end to end.
    :param model: Object of class `DeepGenerativeModel`
    :param dataloader: Dataloader for unlabelled data
    :param labelled: [x, y] pairs of labelled data
    :param optimizer: A PyTorch optimizer
    :param objective: Loss function for labelled data, e.g. `LabelledLoss`
    :param labels: Labels in the dataset, e.g. `[0, 1, 2]`
    :param epochs: Number of epochs to run the training loop for
    """
    [x, _] = labelled

    [n_labelled, *_] = x.size()
    n_unlabelled = len(dataloader.dataset)

    for epoch in range(epochs):
        [x, y] = labelled

        # Bernoulli transform for binary cross entropy
        x = torch.bernoulli(x)

        x = Variable(x.view(n_labelled, -1), requires_grad=True)
        y = Variable(y.view(n_labelled, 1), requires_grad=False)

        optimizer.zero_grad()

        # Labelled data
        reconstruction, y_logits, (z, z_mu, z_log_var) = model(x, y.type(torch.FloatTensor))

        # Batchwise loss
        y = y.view(-1)
        L = objective(reconstruction, x, y, z, z_mu, z_log_var)

        # Unlabelled data
        U = Variable(torch.FloatTensor([0]))

        for u, l in dataloader:
            u = torch.bernoulli(u)
            u, _ = separate(u, l, labels)
            [batch_size, *_] = u.size()
            u = Variable(u.view(batch_size, -1), requires_grad=True)

            u_logits = model.classifier(u)

            # Gather unlabelled loss in a single tensor per batch
            loss_tensor = Variable(torch.zeros(batch_size, len(labels)))
            for i, label in enumerate(labels):
                label = Variable(generate_label(batch_size, label))
                reconstruction, _, (z, z_mu, z_log_var) = model(u, label.type(torch.FloatTensor))

                label = label.view(-1)
                loss = torch.stack([objective(*data) for data in zip(reconstruction, u, label, z, z_mu, z_log_var)], dim=1)
                loss_tensor[:, i] = loss.view(-1)

            # Weighted loss + entropy
            U += torch.mean(torch.mul(u_logits, loss_tensor - torch.log(u_logits)))

        L -= model.beta * F.cross_entropy(y_logits, y)
        J = -(L + U)

        J.backward()
        optimizer.step()

        print("Epoch: {}\t labelled loss: {}, unlabelled loss: {}".format(epoch, float(L.data.numpy()), float(U.data.numpy())))
