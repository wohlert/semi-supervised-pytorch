"""
M2 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vae import reparametrize


class Classifier(nn.Module):
    """
    Two layer classifier
    with softmax output.
    """
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.prediction = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.prediction(x))
        return x


class Encoder(nn.Module):
    """
    Encodes x and y into a
    latent representation z.
    """
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, y_dim, h_dim, z_dim] = dims
        self.transform_x = nn.Linear(x_dim, h_dim)
        self.transform_y = nn.Linear(y_dim, h_dim)
        self.latent = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):
        x = self.transform_x(x)
        y = self.transform_y(y)
        z = F.relu(self.latent(x + y))
        return self.mu(z), F.softplus(self.log_var(z))


class Decoder(nn.Module):
    """
    Decodes latent representation z
    and label y into an original
    representation x.
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()
        [z_dim, h_dim, y_dim, x_dim] = dims
        self.transform = nn.Linear(y_dim, z_dim)
        self.dense = nn.Linear(z_dim, h_dim)
        self.reconstruction = nn.Linear(h_dim, x_dim)

    def forward(self, z, y):
        y = self.transform(y)
        x = F.relu(self.dense(z + y))
        x = F.sigmoid(self.reconstruction(x))
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

        self.encoder = Encoder([self.x_dim, self.y_dim, self.h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim, self.h_dim, self.y_dim, self.x_dim])
        self.classifier = Classifier([self.x_dim, self.h_dim, self.y_dim])

    def forward(self, x, y):
        # Classify the data point
        y_logits = self.classifier(x)

        # Add label and data and generate latent variable
        z_mu, z_log_var = self.encoder(x, y)
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

        log_prior_y = torch.sum(y * torch.log(prior_y), dim=1)

        # Binary cross entropy as log likelihood of r
        log_likelihood = torch.sum((x * torch.log(r) + (1 - x) * torch.log(1 - r)), dim=-1)

        # Gaussian variational distribution over z
        kl_divergence = torch.sum(-0.5 * (z_log_var - z_mu**2 - torch.exp(z_log_var) + 1), dim=-1)

        return log_likelihood + log_prior_y - kl_divergence


def generate_label(batch_size, label, nlabels=2):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param batch_size: number of labels
    :param label: label to generate
    :param nlabels: number of total labels
    """
    labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
    y = torch.zeros((batch_size, nlabels))
    y.scatter_(1, labels, 1)
    return y


def train_dgm(model, unlabelled, labelled, optimizer, objective, cuda=False):
    """
    Trains a deep generative model end to end.
    :param model: Object of class `DeepGenerativeModel`
    :param unlabelled: Dataloader for unlabelled data
    :param labelled: [x, y] pairs of labelled data
    :param optimizer: A PyTorch optimizer
    :param objective: Loss function for labelled data, e.g. `LabelledLoss`
    :param cuda: Optional parameter whether to use CUDA accelation
    """
    [x, y] = labelled
    _, n = y.size()

    x = Variable(x)
    y = Variable(y.type(torch.FloatTensor))

    if cuda:
        x = x.cuda()
        y = y.cuda()

    for u, _ in unlabelled:
        if cuda: u = u.cuda()
        u = Variable(u)

        ### Labelled data ###

        # Calculate loss for labelled
        reconstruction, y_logits, (z, z_mu, z_log_var) = model(x, y)
        L = objective(reconstruction, x, y, z, z_mu, z_log_var) + model.beta * torch.sum(y * torch.log(y_logits), dim=1)
        L = torch.mean(L)

        ### Unlabelled data ###
        u_logits = model.classifier(u)

        # Sum over all labels in the dataset
        [batch_size, *_] = u.size()
        u = u.repeat(n, 1)

        targets = torch.cat([generate_label(batch_size, i, n) for i in range(n)])
        targets = Variable(targets.type(torch.FloatTensor))

        if cuda:
            u = u.cuda()
            targets = targets.cuda()

        reconstruction, _, (z, z_mu, z_log_var) = model(u, targets)
        loss = objective(reconstruction, u, targets, z, z_mu, z_log_var)
        loss = loss.view(u_logits.size())

        # Weighted loss + entropy
        U = torch.sum(torch.mul(u_logits, loss - torch.log(u_logits)), -1)
        U = torch.mean(U)

        U = -U
        L = -L
        J = (L + U)

    J.backward()
    optimizer.step()
    optimizer.zero_grad()

    return L, U